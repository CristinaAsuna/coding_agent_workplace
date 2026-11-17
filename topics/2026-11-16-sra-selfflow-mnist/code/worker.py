#!/usr/bin/env python3
"""Detached, auditable E1 -> E2 worker. Execute only after launch approval."""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path('/home/pc/lab/coding_agent_repro')
TOPIC = '2026-11-16-sra-selfflow-mnist'
REL = Path('topics') / TOPIC
PY = '/home/pc/lab/envs/main/bin/python'
BUDGET = 36000
SMOKE = 174.3929354090069
JOB = 'sra-mnist-formal'


def command(args, check=True):
    print('+', ' '.join(map(str, args)), flush=True)
    return subprocess.run([str(a) for a in args], cwd=ROOT, check=check, text=True)


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2) + '\n')
    tmp.replace(path)


def status(value, summary):
    table = ROOT / 'EXPERIMENTS.md'
    lines = table.read_text().splitlines()
    found = False
    for index, line in enumerate(lines):
        if line.startswith('|') and f'| {TOPIC} |' in line:
            cells = line.split('|')
            cells[5] = f' {value} '
            cells[6] = f' {summary} '
            lines[index] = '|'.join(cells)
            found = True
    if not found:
        raise RuntimeError('Topic row missing in EXPERIMENTS.md')
    table.write_text('\n'.join(lines) + '\n')


def run_start(prior_used, source_run):
    # lab.py validates and archives only committed topic code.
    args = [PY, 'tools/lab.py', 'run', 'start', str(REL), '--exp', 'E1', '--job', JOB,
            '--', PY, str(REL / 'code/run.py'), '--exp', 'E1', '--seeds', '0',
            '--data-root', '/mnt/d/Datasets', '--out', 'outputs', '--classifier',
            str(ROOT / REL / 'runs/smoke/classifier.pt'), '--already-sec', str(prior_used),
            '--budget-sec', str(BUDGET)]
    if source_run:
        old_record = ROOT / REL / 'results' / 'runs' / f'{source_run.name}.json'
        if not old_record.is_file():
            raise RuntimeError('Source run record missing')
        source_commit = json.loads(old_record.read_text())['commit']
        args[args.index('--')] = '--note'
        args.insert(args.index('--note') + 1, f'Checkpoint inherited from {source_run.name}, snapshot {source_commit}')
        args.insert(args.index('--note') + 2, '--')
    print('+', ' '.join(args), flush=True)
    result = subprocess.run(args, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout, end='', flush=True)
    match = re.search(r'运行 (E1-\S+)：机器', result.stdout)
    if result.returncode or not match:
        raise RuntimeError('Could not start/register run; inspect log and records before retrying')
    run_id = match.group(1)
    record = json.loads((ROOT / REL / 'results' / 'runs' / f'{run_id}.json').read_text())
    return run_id, Path(record['run_dir'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--resume', type=Path, help='Existing registered run directory to resume')
    p.add_argument('--from-run', type=Path, help='Prior run directory containing the 3000-step checkpoint')
    p.add_argument('--prior-used-sec', type=float, default=SMOKE, help='Cumulative topic time before this run')
    args = p.parse_args()
    if args.resume and args.from_run:
        p.error('--resume and --from-run are mutually exclusive')
    if args.from_run and args.prior_used_sec < 1369.0671829122875:
        p.error('Prior runs, diagnostics and verification consumed at least 1369.07 seconds')
    if not 0 <= args.prior_used_sec < BUDGET:
        p.error('Invalid cumulative prior budget usage')
    plan = (ROOT / REL / 'PLAN.md').read_text()
    if '<!-- approved-gpu-budget-seconds: 36000 -->' not in plan:
        raise RuntimeError('Approved 10-hour budget marker missing')
    # A failed workspace check blocks transition, including unrelated errors.
    command([PY, 'tools/lab.py', 'check', str(REL)])
    query = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True)
    free = int(query.splitlines()[0].strip())
    print(f'GPU free: {free} MiB; smoke peak reserved: 6258 MiB', flush=True)
    if free < 4096:
        raise RuntimeError('Less than PLAN minimum 4096 MiB free; stopping')
    if args.resume:
        run_dir = args.resume.resolve()
        state_path = run_dir / 'worker_state.json'
        state = json.loads(state_path.read_text())
        if state['status'] != 'running':
            raise RuntimeError('Only an interrupted running record may be resumed')
        if state.get('active_since'):
            # Includes downtime after an interruption: conservative budget accounting.
            state['used_sec'] += max(0, time.time() - state['active_since'])
            state['active_since'] = None
    else:
        status('running', '正式计算运行中')
        summary = ROOT / REL / 'SUMMARY.md'
        previous = summary.read_text()
        summary.write_text('> 正式计算运行中；以下为先前 smoke 预算停止时的历史记录。\n\n' + previous)
        command(['git', 'add', 'EXPERIMENTS.md', str(REL / 'SUMMARY.md')])
        command(['git', 'commit', '-m', f'run[pc]({TOPIC}): begin formal run'])
        try:
            run_id, run_dir = run_start(args.prior_used_sec, args.from_run)
        except Exception:
            status('blocked', '运行登记失败，详见后台日志')
            command(['git', 'add', 'EXPERIMENTS.md'], check=False)
            command(['git', 'commit', '-m', f'run[pc]({TOPIC}): record registration failure'], check=False)
            raise
        state = {'run_id': run_id, 'status': 'running', 'phase': 'E1', 'used_sec': args.prior_used_sec,
                 'active_since': None, 'source_run': str(args.from_run.resolve()) if args.from_run else None}
        state_path = run_dir / 'worker_state.json'
        save(state_path, state)
    out = run_dir / 'outputs'
    snapshot = run_dir / REL / 'code' / 'run.py'
    classifier = ROOT / REL / 'runs/smoke/classifier.pt'
    try:
        if args.from_run:
            source = args.from_run.resolve() / 'outputs/weights/single_full_0_training_state.pt'
            destination = out / 'weights/single_full_0_training_state.pt'
            if not source.is_file() or destination.exists():
                raise RuntimeError('Source checkpoint missing or destination already exists')
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            print(f'Copied source checkpoint {source} to new run {destination}', flush=True)
        for phase in ('E1', 'E2'):
            if phase == 'E1' and state['phase'] == 'E2':
                continue
            if state['used_sec'] >= BUDGET:
                raise RuntimeError('Cumulative 10-hour budget exhausted')
            state.update(phase=phase, active_since=time.time())
            save(state_path, state)
            call = [PY, snapshot, '--exp', phase, '--seeds', '0', '--data-root', '/mnt/d/Datasets',
                    '--out', out, '--classifier', classifier, '--already-sec', str(state['used_sec']),
                    '--budget-sec', str(BUDGET)]
            try:
                command(call)
            finally:
                state['used_sec'] += max(0, time.time() - state['active_since'])
                state['active_since'] = None
                save(state_path, state)
            if phase == 'E1':
                state['phase'] = 'E2'
                save(state_path, state)
        for exp in ('E1', 'E2'):
            source = out / 'results' / f'{exp}.json'
            data = json.loads(source.read_text())
            if data.get('status') != 'done' or not data.get('metrics'):
                raise RuntimeError(f'{exp} did not produce completed metrics')
        expected = ['E1_curve.png', 'E1_final.png', 'E1_samples.png', 'E2_main.png']
        for name in expected:
            if not (out / 'figures' / name).is_file():
                raise RuntimeError(f'Missing figure: {name}')
        (ROOT / REL / 'figures').mkdir(parents=True, exist_ok=True)
        for exp in ('E1', 'E2'):
            shutil.copy2(out / 'results' / f'{exp}.json', ROOT / REL / 'results' / f'{exp}.json')
        for name in expected:
            shutil.copy2(out / 'figures' / name, ROOT / REL / 'figures' / name)
        # Human visual inspection and interpretive summary remain outstanding.
        summary = ROOT / REL / 'SUMMARY.md'
        e1 = json.loads((out / 'results/E1.json').read_text())
        e2 = json.loads((out / 'results/E2.json').read_text())
        values = {name: result['checkpoints']['20000']['fid'] for name, result in e1['metrics'].items()}
        drops = {name: result['relative_drop'] for name, result in e2['metrics'].items()}
        text = (f'# From SRA to Self-Flow — MNIST L2：执行记录\n\n'
                f'topic：{TOPIC} · [计划](PLAN.md) · 状态：计算完成，等待人工验图和核对（running）\n\n'
                '**一句话结论**：后台计算已完成；图像尚未人工查看，正式结论待核对。\n\n'
                '## 结论先行\n\n- 不预判结果；等待逐图检查、核对指标和补写解释。\n\n'
                '## E1：双时间步与注意力消融\n\n'
                f'20000 步 FID 原始值：`{values}`。完整结果见 [E1.json](results/E1.json)。\n\n'
                '## E2：训练时跨噪声交互\n\n'
                f'误差下降比例原始值：`{drops}`。完整结果见 [E2.json](results/E2.json)。\n\n'
                '## 意外与反例\n\n待人工核对。\n\n'
                '## 这个 toy 能说明什么、不能说明什么\n\n'
                '解释待核对；MNIST 和自训分类器 FID 不等同于论文实验。单 seed 无法估计训练随机性；'
                '10% 阈值未经校准，接近阈值时不能下结论。\n\n'
                '## 下一步\n\n逐张人工验图，核对 JSON 与摘要，补写读图说明后再检查 executed。\n\n'
                '## 复现\n\n'
                f'代码快照、命令与权重见运行记录 `{state["run_id"]}` 和 `{out}`；'
                '数据只读复用 `/mnt/d/Datasets/MNIST/raw/`。\n\n'
                '## 执行记录\n\n'
                f'累计 {state["used_sec"]:.2f} 秒，含先前 smoke 174.39 秒。原 4 小时预算因 smoke '
                '外推 8.31 小时而触发停止，用户随后批准上限提高至 10 小时。'
                '四张图已经生成但未人工查看，不标记 executed。\n')
        summary.write_text(text)
        status('running', '计算完成，等待人工验图和结果核对')
        state.update(status='done', phase='awaiting_visual_review')
        save(state_path, state)
        command([PY, 'tools/lab.py', 'check', str(REL)])
        command([PY, 'tools/lab.py', 'run', 'end', state['run_id'], '--status', 'done', '--outputs', str(out),
                 '--note', 'Computation done; topic awaits human figure inspection'])
        command(['git', 'add', 'EXPERIMENTS.md', str(REL / 'SUMMARY.md'), str(REL / 'results/E1.json'),
                 str(REL / 'results/E2.json'), str(REL / 'figures')])
        command(['git', 'commit', '-m', f'run[pc]({TOPIC}): save results awaiting visual review'])
        command([PY, 'tools/lab.py', 'sync'])
        print('Computation complete; manual visual inspection remains required.', flush=True)
    except Exception as exc:
        print(f'Worker stopped: {exc}', file=sys.stderr, flush=True)
        state.update(status='failed', error=str(exc), active_since=None)
        save(state_path, state)
        status('blocked', '后台计算停止，详见执行记录')
        for exp in ('E1', 'E2'):
            result_path = out / 'results' / f'{exp}.json'
            if result_path.is_file():
                shutil.copy2(result_path, ROOT / REL / 'results' / f'{exp}.json')
        summary = ROOT / REL / 'SUMMARY.md'
        summary.write_text(f'# From SRA to Self-Flow — MNIST L2：执行记录\n\n'
                           f'状态：blocked；后台阶段 {state["phase"]} 停止。\n\n'
                           f'累计 {state["used_sec"]:.2f} 秒（含 smoke）；原因：`{exc}`。'
                           f'运行数据位于 `{out}`。未标记 executed，结果和图尚待人工核对。\n\n'
                           '原 4 小时预算在 smoke 外推为 8.31 小时后触发停止，'
                           '用户随后批准提高至 10 小时。数据只读复用 `/mnt/d/Datasets/MNIST/raw/`。\n')
        command([PY, 'tools/lab.py', 'run', 'end', state['run_id'], '--status', 'failed', '--outputs', str(out),
                 '--note', str(exc)], check=False)
        command(['git', 'add', 'EXPERIMENTS.md', str(REL / 'SUMMARY.md'), str(REL / 'results/E1.json'),
                 str(REL / 'results/E2.json')], check=False)
        command(['git', 'commit', '-m', f'run[pc]({TOPIC}): record interrupted formal run'], check=False)
        command([PY, 'tools/lab.py', 'sync'], check=False)
        raise


if __name__ == '__main__':
    main()
