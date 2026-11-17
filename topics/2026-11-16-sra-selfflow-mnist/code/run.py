#!/usr/bin/env python3
"""MNIST DiT experiment. Run --help for the smoke and formal interfaces."""
import argparse
import copy
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path('/home/pc/lab/coding_agent_repro')
sys.path.insert(0, str(ROOT / 'tools'))
import figstyle
import matplotlib.pyplot as plt

DEVICE = torch.device('cuda')
CONDITIONS = [('single', 'full'), ('dual', 'full'), ('single', 'sep'), ('dual', 'sep')]
STEPS = (5000, 10000, 20000)
CTX = (0.8, 0.6, 0.4, 0.2, 0.0)
DEADLINE = None


def check_budget():
    if DEADLINE is not None and time.perf_counter() >= DEADLINE:
        raise RuntimeError('Approved topic budget reached; stopping')


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def finite(value, label):
    if not torch.isfinite(value).all():
        raise RuntimeError(f'Non-finite {label}; stopping')


def sinusoid(values, width):
    frequencies = torch.exp(-math.log(10000) * torch.arange(width // 2, device=values.device).float() / (width // 2))
    angles = values.float().unsqueeze(-1) * frequencies
    return torch.cat((angles.cos(), angles.sin()), dim=-1)


def positions():
    y, x = torch.meshgrid(torch.arange(14), torch.arange(14), indexing='ij')
    return torch.cat((sinusoid(y.flatten(), 96), sinusoid(x.flatten(), 96)), -1)[None]


def patches(x):
    return F.unfold(x, kernel_size=2, stride=2).transpose(1, 2)


def unpatch(x):
    return F.fold(x.transpose(1, 2), (28, 28), kernel_size=2, stride=2)


def groups(batch, device):
    # Exactly 49 uniformly sampled positions, independently for every image.
    order = torch.rand(batch, 196, device=device).argsort(dim=1)
    mask = torch.zeros(batch, 196, device=device, dtype=torch.bool)
    return mask.scatter_(1, order[:, :49], True)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(192, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(192, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(192, 576)
        self.proj = nn.Linear(192, 192)
        self.mlp = nn.Sequential(nn.Linear(192, 768), nn.GELU(approximate='tanh'), nn.Linear(768, 192))
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(192, 1152))
        nn.init.zeros_(self.mod[-1].weight)
        nn.init.zeros_(self.mod[-1].bias)

    def forward(self, x, c, mask):
        s1, a1, g1, s2, a2, g2 = self.mod(c).chunk(6, -1)
        h = self.norm1(x) * (1 + a1) + s1
        qkv = self.qkv(h).reshape(h.shape[0], 196, 3, 6, 32).permute(2, 0, 3, 1, 4)
        h = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], attn_mask=mask)
        h = h.transpose(1, 2).reshape(x.shape[0], 196, 192)
        x = x + g1 * self.proj(h)
        return x + g2 * self.mlp(self.norm2(x) * (1 + a2) + s2)


class DiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(4, 192)
        self.register_buffer('pos', positions(), persistent=False)
        self.time = nn.Sequential(nn.Linear(192, 768), nn.SiLU(), nn.Linear(768, 192))
        self.labels = nn.Embedding(11, 192)
        self.blocks = nn.ModuleList(Block() for _ in range(8))
        self.head = nn.Sequential(nn.Linear(192, 384), nn.SiLU(), nn.Linear(384, 192))
        self.norm = nn.LayerNorm(192, elementwise_affine=False, eps=1e-6)
        self.final_mod = nn.Sequential(nn.SiLU(), nn.Linear(192, 384))
        self.output = nn.Linear(192, 4)
        nn.init.zeros_(self.final_mod[-1].weight)
        nn.init.zeros_(self.final_mod[-1].bias)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x, t, labels, group, attention='full', feature=None):
        c = self.time(sinusoid(t * 1000, 192)) + self.labels(labels)[:, None, :]
        x = self.input(x) + self.pos
        mask = (group[:, None, :, None] == group[:, None, None, :]) if attention == 'sep' else None
        intermediate = None
        for index, block in enumerate(self.blocks, 1):
            x = block(x, c, mask)
            if index == feature:
                intermediate = x
                if feature == 6:
                    return intermediate
        shift, scale = self.final_mod(c).chunk(2, -1)
        output = self.output(self.norm(x) * (1 + scale) + shift)
        return output, intermediate


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc = nn.Linear(64 * 7 * 7, 64)
        self.out = nn.Linear(64, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        feature = F.relu(self.fc(x.flatten(1)))
        return self.out(feature), feature


def data(path, download=False):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    return (datasets.MNIST(path, train=True, transform=transform, download=download),
            datasets.MNIST(path, train=False, transform=transform, download=download))


def loader(ds, batch=256, shuffle=False, drop_last=False):
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, drop_last=drop_last, num_workers=2, pin_memory=True)


@torch.no_grad()
def features(model, batches):
    fs, preds = [], []
    for x in batches:
        logits, h = model(x.to(DEVICE))
        finite(h, 'classifier feature')
        fs.append(h.float().cpu().numpy())
        preds.append(logits.argmax(1).cpu().numpy())
    return np.concatenate(fs), np.concatenate(preds)


def fid(a, b):
    ma, mb = a.mean(0), b.mean(0)
    ca, cb = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    # Symmetric PSD formulation avoids unstable complex sqrtm on singular covariances.
    ev, u = np.linalg.eigh(ca)
    root = (u * np.sqrt(np.maximum(ev, 0))) @ u.T
    cross = np.linalg.eigvalsh(root @ cb @ root)
    result = float((ma - mb) @ (ma - mb) + np.trace(ca) + np.trace(cb) - 2 * np.sqrt(np.maximum(cross, 0)).sum())
    if not np.isfinite(result):
        raise RuntimeError('Non-finite FID; stopping')
    return max(0.0, result)


def classifier(train, test, path):
    model = Classifier().to(DEVICE)
    if path.exists():
        state = torch.load(path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state['model'])
    else:
        seed_all(0)
        # Reinitialize after setting the seed.
        model = Classifier().to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for epoch in range(2):
            model.train()
            for x, y in loader(train, 128, True):
                check_budget()
                x, y = x.to(DEVICE), y.to(DEVICE)
                loss = F.cross_entropy(model(x)[0], y)
                finite(loss, 'classifier loss')
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            print(f'classifier epoch {epoch+1}/2', flush=True)
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader(test):
            pred = model(x.to(DEVICE))[0].argmax(1).cpu()
            correct += (pred == y).sum().item()
            total += len(y)
    acc = correct / total
    print(f'classifier test accuracy={acc:.6f}', flush=True)
    if acc < .985:
        raise RuntimeError(f'Classifier accuracy {acc:.6f} below 0.985; stopping')
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'model': model.state_dict(), 'test_accuracy': acc}, path)
    return model, acc


def update_ema(ema, student):
    with torch.no_grad():
        for e, s in zip(ema.parameters(), student.parameters()):
            e.lerp_(s, 0.001)


def train_step(model, ema, optimizer, scaler, x, labels, schedule, attention):
    batch = len(x)
    x0 = patches(x)
    x1 = torch.randn_like(x0)
    group = groups(batch, DEVICE)
    t0 = torch.rand(batch, 1, device=DEVICE)
    t1 = torch.rand(batch, 1, device=DEVICE) if schedule == 'dual' else t0
    t = torch.where(group, t0, t1)
    teacher_t = torch.where(group, (t0 - torch.rand_like(t0) * .2).clamp_min(0),
                            (t1 - torch.rand_like(t1) * .2).clamp_min(0))
    student_x = (1 - t[..., None]) * x0 + t[..., None] * x1
    teacher_x = (1 - teacher_t[..., None]) * x0 + teacher_t[..., None] * x1
    labels = torch.where(torch.rand(batch, device=DEVICE) < .1, 10, labels)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.float16):
        pred, feat = model(student_x, t, labels, group, attention, 3)
        with torch.no_grad():
            target_feat = ema(teacher_x, teacher_t, labels, group, attention, 6)
        velocity = F.mse_loss(pred.float(), (x1 - x0).float())
        align = (1 - F.cosine_similarity(model.head(feat).float(), target_feat.float(), dim=-1)).mean()
        loss = velocity + .5 * align
    finite(loss, 'training loss')
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    checks = [torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
    overflow = not torch.stack(checks).all().item()
    gradient_nan = overflow and torch.stack([torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None]).any().item()
    scale_before = scaler.get_scale()
    # GradScaler.step skips optimizer.step on non-finite gradients; it normally
    # returns None even on success, so the finite check, not its return, gates EMA.
    scaler.step(optimizer)
    scaler.update()
    scale_after = scaler.get_scale()
    if not overflow:
        update_ema(ema, model)
    result = {'loss': float(loss.detach()), 'overflow': overflow, 'skip_reason': 'gradient_nan' if gradient_nan else ('gradient_inf' if overflow else None),
              'scale_before': scale_before, 'scale_after': scale_after}
    if overflow:
        print(f'AMP {result["skip_reason"]}: loss={result["loss"]:.8g} scale={scale_before:g}->{scale_after:g}; optimizer and EMA skipped', flush=True)
        if gradient_nan:
            raise RuntimeError('NaN gradient; stopping after safe optimizer skip')
        if scale_before <= 1:
            raise RuntimeError('Non-finite gradient persists at loss scale <= 1; stopping')
    return result


@torch.no_grad()
def sample(model, n, batch=128):
    model.eval()
    # A fresh generator fixes all conditions to exactly the same noise and classes.
    gen = torch.Generator(device=DEVICE).manual_seed(12345)
    noise = torch.randn((n, 196, 4), generator=gen, device=DEVICE)
    labels = torch.arange(n, device=DEVICE) // (n // 10)
    images = []
    for start in range(0, n, batch):
        check_budget()
        x = noise[start:start+batch].clone()
        lab = labels[start:start+batch]
        group = torch.zeros(len(x), 196, dtype=torch.bool, device=DEVICE)
        for step in range(50):
            t = torch.full((len(x), 196), 1 - step / 50, device=DEVICE)
            with torch.autocast('cuda', dtype=torch.float16):
                pred, _ = model(x, t, lab, group)
            x = x - pred.float() / 50
        finite(x, 'sample')
        images.append(unpatch(x).cpu())
    return torch.cat(images), labels.cpu().numpy()


def eval_sample(model, cnn, real, count):
    start = time.perf_counter()
    images, labels = sample(model, count)
    torch.cuda.synchronize()
    sampling = time.perf_counter() - start
    start = time.perf_counter()
    fs, pred = features(cnn, (images[i:i+256] for i in range(0, count, 256)))
    score = fid(fs, real)
    return {'fid': score, 'cond_acc': float((pred == labels).mean()),
            'sample_sec': sampling, 'feature_fid_sec': time.perf_counter() - start}, images


def train_condition(schedule, attention, seed, train, cnn, real, out, smoke, activation_checkpoint=False):
    seed_all(seed)
    model = DiT().to(DEVICE)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    if activation_checkpoint:
        # Recompute deterministic student blocks; EMA teacher remains unchanged.
        from torch.utils.checkpoint import checkpoint
        for block in model.blocks:
            original = block.forward
            def wrapped(x, c, mask, original=original):
                return checkpoint(original, x, c, mask, use_reentrant=False, preserve_rng_state=True)
            block.forward = wrapped
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0)
    scaler = torch.amp.GradScaler('cuda')
    name = f'{schedule}+{attention}'
    points = (200,) if smoke else STEPS
    metrics, train_seconds = {}, 0.0
    skipped_updates, successful_updates, overflow_events = 0, 0, []
    progress_path = out / 'weights' / f'{schedule}_{attention}_{seed}_training_state.pt'
    start_step = 0
    if not smoke and progress_path.exists():
        # These checkpoints are produced locally by this run, never downloaded input.
        state = torch.load(progress_path, map_location='cpu', weights_only=False)
        model.load_state_dict(state['student'])
        ema.load_state_dict(state['ema'])
        opt.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler'])
        start_step, metrics, train_seconds = state['step'], state['metrics'], state['train_sec']
        skipped_updates = state.get('skipped_updates', 0)
        successful_updates = state.get('successful_updates', start_step - skipped_updates)
        overflow_events = state.get('overflow_events', [])
        random.setstate(state['random'])
        np.random.set_state(state['numpy'])
        torch.set_rng_state(state['torch_cpu'].cpu())
        torch.cuda.set_rng_state(state['torch_cuda'].cpu())
        del state
        print(f'Resuming {name} at step {start_step}', flush=True)

    # Each epoch has an independent deterministic permutation. A restart can
    # reconstruct the exact next batch without replaying data-loader iterations.
    per_epoch = len(train) // 256
    iteration, loaded_epoch = None, -1
    for step in range(start_step + 1, points[-1] + 1):
        check_budget()
        torch.cuda.synchronize()
        start = time.perf_counter()
        epoch, batch_index = divmod(step - 1, per_epoch)
        if epoch != loaded_epoch:
            permutation = torch.randperm(len(train), generator=torch.Generator().manual_seed(seed * 1000000 + epoch))
            indices = permutation[batch_index * 256:per_epoch * 256].tolist()
            subset = torch.utils.data.Subset(train, indices)
            gen = torch.Generator().manual_seed(seed * 1000000 + epoch + 123456)
            iteration = iter(DataLoader(subset, batch_size=256, num_workers=2, pin_memory=True, generator=gen))
            loaded_epoch = epoch
        x, y = next(iteration)
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        try:
            step_result = train_step(model, ema, opt, scaler, x, y, schedule, attention)
        except RuntimeError as exc:
            print(f'{name} seed={seed} step={step} stopping: {exc}', flush=True)
            raise
        if step_result['overflow']:
            skipped_updates += 1
            overflow_events.append({'step': step, **step_result})
            print(f'{name} step={step} skipped_updates={skipped_updates}', flush=True)
        else:
            successful_updates += 1
        torch.cuda.synchronize()
        train_seconds += time.perf_counter() - start
        if step % 100 == 0:
            print(f'{name} seed={seed} step={step} loss={step_result["loss"]:.5f}', flush=True)
        if step in points:
            path = out / 'weights' / f'{schedule}_{attention}_{seed}_{step}.pt'
            path.parent.mkdir(parents=True, exist_ok=True)
            save_start = time.perf_counter()
            torch.save(ema.state_dict(), path)
            train_seconds += time.perf_counter() - save_start
            metrics[str(step)], images = eval_sample(ema, cnn, real, 1000 if smoke else 10000)
            if not smoke and step == 20000:
                # Save all generated outputs for uncurated samples and audit.
                torch.save(images, out / f'samples_{schedule}_{attention}_{seed}.pt')
            print(f'{name} step={step} {metrics[str(step)]}', flush=True)
        if not smoke and (step % 500 == 0 or step in points):
            state = {'step': step, 'student': model.state_dict(), 'ema': ema.state_dict(),
                     'optimizer': opt.state_dict(), 'scaler': scaler.state_dict(), 'metrics': metrics,
                     'train_sec': train_seconds, 'skipped_updates': skipped_updates,
                     'successful_updates': successful_updates, 'overflow_events': overflow_events,
                     'random': random.getstate(), 'numpy': np.random.get_state(),
                     'torch_cpu': torch.get_rng_state(), 'torch_cuda': torch.cuda.get_rng_state()}
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = progress_path.with_suffix('.tmp')
            torch.save(state, temporary)
            temporary.replace(progress_path)
    return {'checkpoints': metrics, 'train_sec': train_seconds, 'train_steps': points[-1],
            'skipped_updates': skipped_updates, 'successful_updates': successful_updates, 'overflow_events': overflow_events}


def baselines(cnn, train, test):
    real, _ = features(cnn, (x for x, _ in loader(test)))
    gen = torch.Generator().manual_seed(12345)
    indices = torch.randperm(len(train), generator=gen)[:10000]
    chosen = torch.stack([train[int(i)][0] for i in indices])
    actual, _ = features(cnn, (chosen[i:i+256] for i in range(0, len(chosen), 256)))
    # Per-pixel moments are measured over the full training set.
    all_train = train.data.float().div(127.5).sub(1)
    mean, std = all_train.mean(0), all_train.std(0)
    normal = torch.randn((10000, 28, 28), generator=gen) * std + mean
    noise, _ = features(cnn, (normal[i:i+256, None] for i in range(0, 10000, 256)))
    return real, {'real_train_vs_test_fid': fid(actual, real), 'independent_gaussian_fid': fid(noise, real)}


def plot_e1(metrics, base, out, figs):
    figstyle.setup()
    fig, ax = plt.subplots()
    for i, (s, a) in enumerate(CONDITIONS):
        label = f'{s}+{a}'
        ax.plot(STEPS, [metrics[label]['checkpoints'][str(k)]['fid'] for k in STEPS], marker='o', label=label, color=figstyle.SERIES[i])
    ax.axhline(base['real_train_vs_test_fid'], color='gray', ls='--', label='Real data reference')
    ax.set(xlabel='Training step', ylabel='Feature FID', yscale='log', title='Feature FID over training')
    ax.legend()
    figstyle.save_fig(fig, figs / 'E1_curve.png')
    fig, ax = plt.subplots()
    vals = [metrics[f'{s}+{a}']['checkpoints']['20000']['fid'] for s, a in CONDITIONS]
    for i, val in enumerate(vals):
        ax.scatter(i, val, color=figstyle.SERIES[i], s=60)
    ax.hlines(vals[2], 2.7, 3.3, color='gray', linestyles='--', label='No interaction expectation')
    ax.set_xticks(range(4), [f'{s}\n{a}' for s, a in CONDITIONS])
    ax.set(ylabel='Feature FID', title='Final feature FID across conditions')
    ax.legend()
    figstyle.save_fig(fig, figs / 'E1_final.png')
    fig, axes = plt.subplots(4, 20, figsize=(16, 3.5))
    for row, (s, a) in enumerate(CONDITIONS):
        samples = torch.load(out / f'samples_{s}_{a}_0.pt', map_location='cpu', weights_only=True)
        for col in range(20):
            ax = axes[row, col]
            ax.imshow(samples[(col // 2)*1000 + col % 2, 0], cmap='gray', vmin=-1, vmax=1)
            ax.axis('off')
        axes[row, 0].set_ylabel(f'{s}+{a}')
        axes[row, 0].text(-.2, .5, f'{s}+{a}', transform=axes[row, 0].transAxes, ha='right', va='center', fontsize=8)
    fig.suptitle('Fixed noise: first two samples per class')
    figstyle.save_fig(fig, figs / 'E1_samples.png')


@torch.no_grad()
def e2(test, out, figs):
    seed_all(9876)
    x0 = patches(torch.stack([test[i][0] for i in range(2000)]).to(DEVICE))
    labels = torch.tensor([test[i][1] for i in range(2000)], device=DEVICE)
    group = groups(2000, DEVICE)
    noise = torch.randn_like(x0)
    target = noise - x0
    lines = [('dual+full', 'dual', 'full', 'full'), ('single+full', 'single', 'full', 'full'),
             ('dual+sep masked', 'dual', 'sep', 'sep'), ('dual+sep full', 'dual', 'sep', 'full')]
    results = {}
    for name, schedule, train_attn, eval_attn in lines:
        model = DiT().to(DEVICE).eval()
        model.load_state_dict(torch.load(out / 'weights' / f'{schedule}_{train_attn}_0_20000.pt', map_location=DEVICE, weights_only=True))
        values = []
        for context in CTX:
            check_budget()
            total, n = 0.0, 0
            for start in range(0, 2000, 128):
                sl = slice(start, start + 128)
                t = torch.where(group[sl], .8, context)
                x = (1 - t[..., None]) * x0[sl] + t[..., None] * noise[sl]
                with torch.autocast('cuda', dtype=torch.float16):
                    pred, _ = model(x, t, labels[sl], group[sl], eval_attn)
                err = (pred.float() - target[sl]).square().mean(-1)[group[sl]]
                finite(err, 'E2 error')
                total += err.sum().item()
                n += err.numel()
            values.append(total / n)
        results[name] = {'mse': values, 'relative_drop': 1 - values[-1] / values[0]}
        if name == 'dual+sep masked' and (max(values) - min(values)) / values[0] > .01:
            raise RuntimeError(f'E2 masked control varies by >1%; values={values}; stopping')
    figstyle.setup()
    fig, ax = plt.subplots()
    for i, (name, *_) in enumerate(lines):
        ax.plot(CTX, results[name]['mse'], marker='o', label=name, color=figstyle.SERIES[i])
    ax.axhline(results['dual+full']['mse'][0], ls='--', color='gray', label='No interaction expectation')
    ax.invert_xaxis()
    ax.set(xlabel='Context noise time', ylabel='Minority token velocity MSE', title='Velocity error as context becomes cleaner')
    ax.legend()
    figstyle.save_fig(fig, figs / 'E2_main.png')
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--exp', choices=['smoke', 'E1', 'E2'], required=True)
    p.add_argument('--seeds', type=int, nargs='+', default=[0])
    p.add_argument('--data', '--data-root', dest='data', type=Path, default=Path('/home/pc/lab/assets/datasets/mnist'),
                   help='Existing torchvision root, MNIST directory, or MNIST/raw directory')
    p.add_argument('--download', action='store_true')
    p.add_argument('--out', type=Path, required=True, help='Run-local directory for weights and results')
    p.add_argument('--classifier', type=Path, required=True, help='Shared classifier weights from smoke')
    p.add_argument('--already-sec', type=float, default=0, help='Prior topic GPU time to include in budget estimate')
    p.add_argument('--budget-sec', type=int, default=14400, help='Approved total topic budget; higher limits require a plan marker')
    p.add_argument('--activation-checkpoint', action='store_true', help='Recompute student blocks to reduce training memory')
    p.add_argument('--forecast-training-steps', type=int, default=80000,
                   help='Smoke budget projection only; does not change formal training steps')
    args = p.parse_args()
    if not 1 <= args.forecast_training_steps <= 80000:
        p.error('--forecast-training-steps must be between 1 and 80000')
    if args.exp != 'smoke' and args.forecast_training_steps != 80000:
        p.error('--forecast-training-steps is only for smoke')
    if args.seeds != [0]:
        p.error('The approved plan revision allows seed 0 only')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; stopping')
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    global DEADLINE
    if args.budget_sec > 14400:
        plan = ROOT / 'topics/2026-11-16-sra-selfflow-mnist/PLAN.md'
        marker = f'<!-- approved-gpu-budget-seconds: {args.budget_sec} -->'
        if marker not in plan.read_text():
            raise RuntimeError('Budget increase requires an approved plan revision')
    if args.budget_sec <= 0 or args.already_sec < 0 or args.already_sec >= args.budget_sec:
        raise RuntimeError('Invalid already-used time or topic budget exhausted')
    DEADLINE = started + args.budget_sec - args.already_sec
    args.out.mkdir(parents=True, exist_ok=True)
    if args.data.name == 'raw' and args.data.parent.name == 'MNIST':
        args.data = args.data.parent.parent
    elif args.data.name == 'MNIST' and (args.data / 'raw').is_dir():
        args.data = args.data.parent
    train, test = data(args.data, args.download)
    cnn, accuracy = classifier(train, test, args.classifier)
    figs = args.out / 'figures'
    if args.exp == 'E2':
        metrics = e2(test, args.out, figs)
        dump(args.out / 'results' / 'E2.json', {'exp': 'E2', 'status': 'done', 'seeds': [0], 'representative_seed': 0,
              'metrics': metrics, 'runtime_sec': time.perf_counter() - started,
              'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
              'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved()})
        return
    real, base = baselines(cnn, train, test)
    conditions = [('single', 'full'), ('dual', 'sep')] if args.exp == 'smoke' else CONDITIONS
    metrics = {}
    for s, a in conditions:
        metrics[f'{s}+{a}'] = train_condition(s, a, 0, train, cnn, real, args.out, args.exp == 'smoke', args.activation_checkpoint)
    elapsed = time.perf_counter() - started
    result = {'exp': args.exp, 'status': 'done', 'seeds': [0], 'representative_seed': 0,
              'metrics': metrics, 'baselines': base, 'classifier_accuracy': accuracy, 'runtime_sec': elapsed,
              'activation_checkpoint': args.activation_checkpoint,
              'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
              'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved()}
    if args.exp == 'smoke':
        rates = [v['train_sec'] / v['train_steps'] for v in metrics.values()]
        sampling = [v['checkpoints']['200']['sample_sec'] for v in metrics.values()]
        feature = [v['checkpoints']['200']['feature_fid_sec'] for v in metrics.values()]
        # Use the slower measured condition for a conservative extrapolation; include E2's plan budget.
        setup_sec = max(0.0, elapsed - sum(v['train_sec'] + v['checkpoints']['200']['sample_sec'] + v['checkpoints']['200']['feature_fid_sec'] for v in metrics.values()))
        forecast = args.already_sec + elapsed + setup_sec + args.forecast_training_steps*max(rates) + 12*10*(max(sampling)+max(feature)) + 600
        result['forecast'] = {'already_sec': args.already_sec, 'smoke_elapsed_sec': elapsed,
            'train_sec_per_step': rates, 'sampling_1000_sec': sampling, 'feature_fid_1000_sec': feature,
            'forecast_training_steps': args.forecast_training_steps,
            'formal_train_sec': args.forecast_training_steps*max(rates), 'formal_eval_sec': 120*(max(sampling)+max(feature)),
            'formal_setup_allowance_sec': setup_sec, 'e2_allowance_sec': 600, 'total_sec': forecast,
            'budget_sec': args.budget_sec, 'within_budget': forecast <= args.budget_sec}
        dump(args.out / 'results' / 'smoke.json', result)
        print(json.dumps(result['forecast'], indent=2), flush=True)
        if forecast > args.budget_sec:
            raise RuntimeError('Forecast exceeds approved GPU budget; stopping before formal run')
    else:
        plot_e1(metrics, base, args.out, figs)
        dump(args.out / 'results' / 'E1.json', result)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        # Preserve the stop reason even when validation fails before final results exist.
        if '--out' in sys.argv and '--exp' in sys.argv:
            out = Path(sys.argv[sys.argv.index('--out') + 1])
            exp = sys.argv[sys.argv.index('--exp') + 1]
            dump(out / 'results' / f'{exp}_failure.json', {'exp': exp, 'status': 'failed',
                 'seeds': [0], 'representative_seed': 0, 'error': str(exc),
                 'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
                 'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None})
        raise
