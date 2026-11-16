"""Synthetic trajectory data, conditioned transformer, and flow-matching utilities."""
import copy
import math
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

T = 16


def trajectories(n, seed):
    rng = np.random.default_rng(seed)
    r = rng.uniform(.5, 1.5, (n, 1))
    phi = rng.uniform(0, 2 * np.pi, (n, 1))
    a = rng.uniform(.3, 1, (n, 1))
    theta = 2 * np.pi * np.arange(T)[None, :] / T + phi
    xy = np.stack((r * np.cos(theta), r * a * np.sin(theta)), axis=-1)
    return torch.tensor(xy + .02 * rng.standard_normal(xy.shape), dtype=torch.float32)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(64, elementwise_affine=False), nn.LayerNorm(64, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(64, 4, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(64, 256), nn.GELU(), nn.Linear(256, 64))
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(64, 256))

    def forward(self, x, cond, mask):
        s1, c1, s2, c2 = self.mod(cond).chunk(4, dim=-1)
        h = self.norm1(x) * (1 + c1) + s1
        x = x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        return x + self.ff(self.norm2(x) * (1 + c2) + s2)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(2, 64)
        self.pos = nn.Parameter(torch.randn(1, T, 64) * .02)
        self.time = nn.Sequential(nn.Linear(64, 256), nn.SiLU(), nn.Linear(256, 64))
        self.blocks = nn.ModuleList([Block() for _ in range(4)])
        self.norm = nn.LayerNorm(64, elementwise_affine=False)
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(64, 128))
        self.output = nn.Linear(64, 2)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.head = nn.Sequential(nn.Linear(64, 128), nn.SiLU(), nn.Linear(128, 64))

    def forward(self, x, t, groups=None):
        freq = torch.exp(-math.log(10000) * torch.arange(32, device=x.device) / 31)
        angles = t[..., None] * freq * 1000
        cond = self.time(torch.cat((angles.sin(), angles.cos()), dim=-1))
        mask = None
        if groups is not None:
            mask = (groups[:, :, None] != groups[:, None, :]).repeat_interleave(4, dim=0)
        h = self.input(x) + self.pos
        early = late = None
        for i, block in enumerate(self.blocks):
            h = block(h, cond, mask)
            if i == 0: early = h
            if i == 2: late = h
        shift, scale = self.mod(cond).chunk(2, dim=-1)
        return self.output(self.norm(h) * (1 + scale) + shift), early, late


def train(data, seed, schedule, attention, alpha, rho, steps, budget_sec):
    torch.manual_seed(seed)
    model = Model()
    teacher = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0)
    # A separate, identically seeded generator gives paired conditions identical draws.
    gen = torch.Generator().manual_seed(seed + 100000)
    started = time.monotonic()
    for step in range(steps):
        if len(data) < 128:
            idx = torch.randint(len(data), (128,), generator=gen)
        else:
            idx = torch.randperm(len(data), generator=gen)[:128]
        x0 = data[idx]
        x1 = torch.randn(x0.shape, generator=gen)
        order = torch.rand((128, T), generator=gen).argsort(dim=1)
        groups = (order >= int(alpha * T)).long()
        ts = torch.rand((128, 2), generator=gen)
        offsets = torch.rand((128, 2), generator=gen) * .2
        # Always draw the same random numbers for paired conditions.
        mix_order = torch.rand(128, generator=gen).argsort()
        if schedule == 'single': ts[:, 1] = ts[:, 0]
        if rho:
            selected = mix_order[:int(128 * rho)]
            groups[selected] = 0
        t = ts.gather(1, groups)
        teacher_t = (ts - offsets).clamp_min(0).gather(1, groups)
        xt = (1 - t[..., None]) * x0 + t[..., None] * x1
        teacher_x = (1 - teacher_t[..., None]) * x0 + teacher_t[..., None] * x1
        group_mask = groups if attention == 'sep' else None
        velocity, early, _ = model(xt, t, group_mask)
        with torch.no_grad():
            _, _, target = teacher(teacher_x, teacher_t, group_mask)
        loss = F.mse_loss(velocity, x1 - x0) + .5 * (1 - F.cosine_similarity(model.head(early), target, dim=-1)).mean()
        if not torch.isfinite(loss): raise RuntimeError(f'Non-finite loss at step {step + 1}')
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            for p, q in zip(teacher.parameters(), model.parameters()): p.lerp_(q, .01)
        if time.monotonic() - started > budget_sec:
            raise RuntimeError(f'Training exceeded time limit at step {step + 1}')
    return teacher, time.monotonic() - started


@torch.no_grad()
def sample(model, seed):
    gen = torch.Generator().manual_seed(seed + 200000)
    noise = torch.randn((2000, T, 2), generator=gen)
    out = []
    for batch in noise.split(128):
        x = batch
        for i in range(50):
            t = torch.full((len(x), T), 1 - i / 50)
            v, _, _ = model(x, t)
            x = x - v / 50
        out.append(x)
    result = torch.cat(out)
    if not torch.isfinite(result).all(): raise RuntimeError('Non-finite samples')
    return result


def directions():
    rng = np.random.default_rng(777)
    d = rng.standard_normal((32, 256)).astype('float32')
    return torch.tensor(d / np.linalg.norm(d, axis=0, keepdims=True))


def swd(x, y, dirs):
    xp = (x.reshape(len(x), -1) @ dirs).sort(dim=0).values
    yp = (y.reshape(len(y), -1) @ dirs).sort(dim=0).values
    return (xp - yp).abs().mean().item()


def memory_ratio(x, held, train_data):
    def nearest(z):
        return torch.cat([torch.cdist(b.flatten(1), train_data.flatten(1)).amin(dim=1) for b in z.split(128)])
    return (nearest(x).median() / nearest(held).median()).item()


def baselines(train_data, held, seed, dirs):
    real = trajectories(2000, seed + 400000)
    gen = torch.Generator().manual_seed(seed + 500000)
    flat = train_data.flatten(1)
    gaussian = torch.randn((2000, 32), generator=gen) * flat.std(dim=0, unbiased=False) + flat.mean(dim=0)
    replay = train_data[torch.randint(len(train_data), (2000,), generator=gen)]
    return {'real': swd(real, held, dirs), 'gaussian': swd(gaussian, held, dirs), 'replay': swd(replay, held, dirs)}
