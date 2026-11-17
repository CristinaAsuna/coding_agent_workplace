#!/usr/bin/env python3
"""Read-only checkpoint audit and bounded replay; never saves training state."""
import argparse
import copy
import importlib.util
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

RUN = Path('/home/pc/lab/runs/2026-11-16-sra-selfflow-mnist/E1-20261116-071540-pc')
CODE = RUN / 'topics/2026-11-16-sra-selfflow-mnist/code/run.py'


def load_run(verify_fix=False):
    code = Path(__file__).with_name('run.py') if verify_fix else CODE
    spec = importlib.util.spec_from_file_location('snapshot_run', code)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tensor_audit(value, path='', issues=None):
    if issues is None:
        issues = []
    if torch.is_tensor(value) and (value.is_floating_point() or value.is_complex()):
        if not torch.isfinite(value).all().item():
            issues.append(path)
    elif isinstance(value, dict):
        for key, child in value.items():
            tensor_audit(child, f'{path}.{key}', issues)
    elif isinstance(value, (tuple, list)):
        for i, child in enumerate(value):
            tensor_audit(child, f'{path}[{i}]', issues)
    return issues


def gradient_report(model):
    result = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        g = parameter.grad.detach()
        nans = int(torch.isnan(g).sum().item())
        infs = int(torch.isinf(g).sum().item())
        if nans or infs:
            result.append({'parameter': name, 'nan': nans, 'inf': infs, 'numel': g.numel()})
    return result


def make_batch(m, x, labels):
    x0 = m.patches(x)
    noise = torch.randn_like(x0)
    group = m.groups(len(x), m.DEVICE)
    t0 = torch.rand(len(x), 1, device=m.DEVICE)
    t = torch.where(group, t0, t0)
    teacher_t = torch.where(group, (t0 - torch.rand_like(t0) * .2).clamp_min(0),
                            (t0 - torch.rand_like(t0) * .2).clamp_min(0))
    x_student = (1 - t[..., None]) * x0 + t[..., None] * noise
    x_teacher = (1 - teacher_t[..., None]) * x0 + teacher_t[..., None] * noise
    labels = torch.where(torch.rand(len(x), device=m.DEVICE) < .1, 10, labels)
    return x_student, x_teacher, t, teacher_t, labels, group, noise - x0


def forward(model, ema, batch, amp):
    x, tx, t, tt, labels, group, target = batch
    with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
        pred, feat = model(x, t, labels, group, 'full', 3)
        with torch.no_grad():
            teacher = ema(tx, tt, labels, group, 'full', 6)
        velocity = F.mse_loss(pred.float(), target.float())
        align = (1 - F.cosine_similarity(model.head(feat).float(), teacher.float(), dim=-1)).mean()
        loss = velocity + .5 * align
    return loss, velocity, align


def replay(args, state):
    m = load_run(args.verify_fix)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    m.seed_all(0)
    model = m.DiT().to(m.DEVICE)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    if args.activation_checkpoint:
        from torch.utils.checkpoint import checkpoint
        for block in model.blocks:
            original = block.forward
            def wrapped(x, c, mask, original=original):
                return checkpoint(original, x, c, mask, use_reentrant=False, preserve_rng_state=True)
            block.forward = wrapped
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0)
    scaler = torch.amp.GradScaler('cuda')
    model.load_state_dict(state['student'])
    ema.load_state_dict(state['ema'])
    optimizer.load_state_dict(state['optimizer'])
    scaler.load_state_dict(state['scaler'])
    saved_step = int(state['step'])
    random.setstate(state['random'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'].cpu())
    torch.cuda.set_rng_state(state['torch_cuda'].cpu())
    del state
    train, _ = m.data(args.data_root, False)
    per_epoch = len(train) // 256
    loaded_epoch, iteration = -1, None
    started = time.monotonic()
    result = {'checkpoint_step': saved_step, 'event': None, 'samples': [],
              'activation_checkpoint': args.activation_checkpoint, 'verify_fix': args.verify_fix}
    post_event_successes = 0
    for step in range(saved_step + 1, saved_step + args.max_steps + 1):
        if time.monotonic() - started >= args.max_seconds:
            result['stop'] = 'time_limit'
            break
        epoch, batch_index = divmod(step - 1, per_epoch)
        if epoch != loaded_epoch:
            permutation = torch.randperm(len(train), generator=torch.Generator().manual_seed(epoch))
            subset = Subset(train, permutation[batch_index * 256:per_epoch * 256].tolist())
            gen = torch.Generator().manual_seed(epoch + 123456)
            iteration = iter(DataLoader(subset, batch_size=256, num_workers=2, pin_memory=True, generator=gen))
            loaded_epoch = epoch
        x, labels = next(iteration)
        if args.verify_fix:
            # The diagnosed event was step 3143; retain a temporary reference
            # only at that step to check both model and EMA remain unchanged.
            before = ([p.detach().clone() for p in model.parameters()], [p.detach().clone() for p in ema.parameters()]) if step == 3143 else None
            r = m.train_step(model, ema, optimizer, scaler, x.to(m.DEVICE, non_blocking=True),
                             labels.to(m.DEVICE, non_blocking=True), 'single', 'full')
            if r['overflow']:
                if result['event'] is not None:
                    raise RuntimeError('Unexpected second overflow during short verification')
                result['event'] = {'step': step, **r,
                    'student_unchanged': before is not None and all(torch.equal(a, b) for a, b in zip(before[0], model.parameters())),
                    'ema_unchanged': before is not None and all(torch.equal(a, b) for a, b in zip(before[1], ema.parameters()))}
            elif result['event'] is not None:
                post_event_successes += 1
                if post_event_successes >= 20:
                    result['stop'] = 'verified_20_subsequent_updates'
                    break
            if step % 100 == 0:
                print(f'verification step={step} loss={r["loss"]:.8g}', flush=True)
            continue
        batch = make_batch(m, x.to(m.DEVICE, non_blocking=True), labels.to(m.DEVICE, non_blocking=True))
        optimizer.zero_grad(set_to_none=True)
        loss, velocity, align = forward(model, ema, batch, True)
        vals = {'step': step, 'loss': float(loss.detach()), 'velocity': float(velocity.detach()),
                'align': float(align.detach()), 'scale': scaler.get_scale()}
        if not torch.isfinite(loss).item():
            result['event'] = {**vals, 'kind': 'nonfinite_forward_loss'}
            break
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        # Match training's single aggregate check; report details only at the event.
        finite = torch.stack([torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]).all().item()
        if not finite:
            event = {**vals, 'kind': 'nonfinite_gradient', 'original_gradients': gradient_report(model)}
            # Keep parameters unchanged for both retries; no optimizer step.
            for label, amp in [('fp16_scale_1', True), ('fp32_scale_1', False)]:
                try:
                    optimizer.zero_grad(set_to_none=True)
                    retry_loss, retry_v, retry_a = forward(model, ema, batch, amp)
                    retry_loss.backward()
                    event[label] = {'loss': float(retry_loss.detach()), 'velocity': float(retry_v.detach()),
                                    'align': float(retry_a.detach()), 'bad_gradients': gradient_report(model)}
                except RuntimeError as exc:
                    event[label] = {'error': str(exc)}
                    optimizer.zero_grad(set_to_none=True)
                    if 'out of memory' in str(exc).lower():
                        torch.cuda.empty_cache()
                    else:
                        raise
            result['event'] = event
            break
        scaler.step(optimizer)
        scaler.update()
        m.update_ema(ema, model)
        if step == saved_step + 1:
            print(json.dumps({'first_step': step, 'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved()}), flush=True)
        if step % 25 == 0:
            print(json.dumps(vals), flush=True)
        if step % 100 == 0:
            result['samples'].append(vals)
    else:
        result['stop'] = 'step_limit'
    torch.cuda.synchronize()
    result['runtime_sec'] = time.monotonic() - started
    result['peak_cuda_reserved_bytes'] = torch.cuda.max_memory_reserved()
    if args.verify_fix:
        event = result['event'] or {}
        result['post_event_successful_updates'] = post_event_successes
        result['verified'] = (event.get('step') == 3143 and event.get('skip_reason') == 'gradient_inf'
                              and event.get('scale_before') == 131072 and event.get('scale_after') == 65536
                              and event.get('student_unchanged') and event.get('ema_unchanged')
                              and post_event_successes == 20)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=Path('/mnt/d/Datasets'))
    parser.add_argument('--replay', action='store_true')
    parser.add_argument('--activation-checkpoint', action='store_true', help='Recompute deterministic student blocks to reduce memory')
    parser.add_argument('--verify-fix', action='store_true', help='Replay with edited training step and verify skip/EMA behavior')
    parser.add_argument('--max-steps', type=int, default=500)
    parser.add_argument('--max-seconds', type=int, default=600)
    args = parser.parse_args()
    if not 1 <= args.max_steps <= 500 or not 1 <= args.max_seconds <= 600:
        parser.error('Diagnostic limits are 500 steps and 600 seconds')
    start = time.monotonic()
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    audit = {'step': state['step'], 'scaler': state['scaler'],
             'nonfinite_tensors': {name: tensor_audit(state[name]) for name in ('student', 'ema', 'optimizer')},
             'runtime_sec': time.monotonic() - start}
    print(json.dumps({'audit': audit}, indent=2), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    if args.replay:
        if any(audit['nonfinite_tensors'].values()):
            raise RuntimeError('Checkpoint already contains non-finite values; replay stopped')
        result = replay(args, state)
        target = 'fix_verification.json' if args.verify_fix else 'replay.json'
        (args.out / target).write_text(json.dumps(result, indent=2, allow_nan=True) + '\n')
        print(json.dumps(result, indent=2, allow_nan=True), flush=True)
        if args.verify_fix and not result['verified']:
            raise RuntimeError('Fix verification did not satisfy all invariants')


if __name__ == '__main__':
    main()
