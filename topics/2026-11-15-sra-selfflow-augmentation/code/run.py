"""Run the frozen SRA/Self-Flow toy plan, with a first-condition budget gate."""
import argparse
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
import toy


def load_style(root):
    path = root / 'tools' / 'figstyle.py'
    spec = importlib.util.spec_from_file_location('figstyle', path)
    if spec is None or spec.loader is None: raise RuntimeError(f'Cannot load {path}')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parser():
    p = argparse.ArgumentParser()
    p.add_argument('--exp', required=True, choices=['E1', 'E2', 'E3'])
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    p.add_argument('--steps', type=int, default=1500)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--workspace-root', type=Path, default=Path(__file__).resolve().parents[3])
    p.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parents[1])
    return p


def conditions(exp):
    if exp == 'E1':
        return [dict(n=256, schedule=s, attention=a, alpha=.25, rho=0.) for s, a in [('single','full'), ('dual','full'), ('single','sep'), ('dual','sep')]]
    if exp == 'E2':
        return [dict(n=n, schedule=s, attention=a, alpha=.25, rho=0.) for n in [64,256,1024,4096] for s,a in [('single','full'),('dual','full'),('dual','sep')]]
    return [dict(n=256, schedule='dual', attention='sep', alpha=a, rho=r) for a in [.25,.375,.5] for r in [0.,.25]]


def label(c): return c['schedule'] + '+' + c['attention']


def plot(exp, rows, seeds, baselines, samples, style, out):
    import matplotlib.pyplot as plt
    style.setup()
    out.mkdir(parents=True, exist_ok=True)
    def get(**kwargs):
        return [r for r in rows if all(r[k] == v for k,v in kwargs.items())]
    def matrix(xkey, xs, metric, **kwargs):
        return np.array([[next(r[metric] for r in get(seed=s, **kwargs) if r[xkey] == x) for x in xs] for s in seeds])
    def refs(ax, n=256):
        for key, ls in [('real', ':'), ('gaussian', '-.'), ('replay', '--')]:
            ax.axhline(np.mean([baselines[str(n)][str(s)][key] for s in seeds]), color='gray', ls=ls, lw=1, label=key)
    if exp == 'E1':
        fig, ax = plt.subplots(figsize=(8,5))
        labels = ['single+full','dual+full','single+sep','dual+sep']
        for i, lab in enumerate(labels):
            vals = [r['swd'] for r in rows if label(r) == lab]
            ax.scatter(np.full(len(vals),i), vals, s=25, color=style.SERIES[i], alpha=.5)
            ax.scatter(i, np.mean(vals), s=85, color=style.SERIES[i], zorder=4)
        ax.plot([2.8,3.2], [np.mean([r['swd'] for r in rows if label(r)=='single+sep'])]*2, '--', color='gray', label='naive expectation')
        refs(ax); ax.set_xticks(range(4), labels, rotation=15); ax.set_yscale('log'); ax.set_ylabel('SWD'); ax.set_title('SWD across schedule and attention'); ax.legend(fontsize=8)
        style.save_fig(fig, out/'E1_main.png')
        representative = style.representative_seed([next(r['swd'] for r in get(seed=s, schedule='dual', attention='sep')) for s in seeds], seeds)
        fig, axes = plt.subplots(5, 12, figsize=(14, 6), sharex=True, sharey=True)
        grids = [samples[(representative,lab)] for lab in labels] + [toy.trajectories(2000, representative+300000)]
        for i, data in enumerate(grids):
            for j in range(12):
                xy = data[j].numpy(); xy = np.concatenate((xy,xy[:1]))
                axes[i,j].plot(xy[:,0],xy[:,1],lw=.7); axes[i,j].set_aspect('equal'); axes[i,j].axis('off')
            axes[i,0].text(-.1,.5, (labels+['held-out'])[i], transform=axes[i,0].transAxes, ha='right', va='center', fontsize=8)
        fig.suptitle('Generated and held-out trajectories'); style.save_fig(fig, out/'E1_samples.png')
    elif exp == 'E2':
        xs = [64,256,1024,4096]; labs = [('single','full'),('dual','full'),('dual','sep')]
        fig, (ax, gain) = plt.subplots(1,2,figsize=(11,4),sharex=True)
        base = matrix('n', xs, 'swd', schedule='single', attention='full')
        for i,(s,a) in enumerate(labs):
            vals = matrix('n',xs,'swd',schedule=s,attention=a)
            style.seed_lines(ax,xs,vals,label=s+'+'+a,color=style.SERIES[i])
            if s == 'dual': style.seed_lines(gain,xs,base-vals,label=s+'+'+a,color=style.SERIES[i])
        for key,ls in [('real',':'),('replay','--'),('gaussian','-.')]:
            ax.plot(xs,[np.mean([baselines[str(n)][str(s)][key] for s in seeds]) for n in xs], color='gray',ls=ls,lw=1,label=key)
        g0=(base-matrix('n',xs,'swd',schedule='dual',attention='full'))[:,0].mean()
        gain.axhline(0,color='gray',lw=.8); gain.axhline(g0,color='gray',ls='--',label='naive expectation')
        for a in (ax,gain): a.set_xscale('log'); a.set_xlabel('Training samples'); a.legend(fontsize=8)
        ax.set_yscale('log'); ax.set_ylabel('SWD'); gain.set_ylabel('Paired SWD gain'); ax.set_title('SWD across dataset sizes'); gain.set_title('Paired gain across dataset sizes')
        style.save_fig(fig,out/'E2_main.png')
        fig,ax=plt.subplots(figsize=(6,4))
        for i,(s,a) in enumerate(labs): style.seed_lines(ax,xs,matrix('n',xs,'mem',schedule=s,attention=a),label=s+'+'+a,color=style.SERIES[i])
        ax.set_xscale('log'); ax.set_xlabel('Training samples'); ax.set_ylabel('Memory ratio'); ax.set_title('Memory ratio across dataset sizes'); ax.legend(); style.save_fig(fig,out/'E2_mem.png')
    else:
        xs=[.25,.375,.5]; fig,ax=plt.subplots(figsize=(6,4))
        for i,rho in enumerate([0.,.25]): style.seed_lines(ax,xs,matrix('alpha',xs,'swd',rho=rho),label=f'rho={rho}',color=style.SERIES[i])
        ax.axhline(matrix('alpha',xs,'swd',rho=0.)[:,0].mean(),color='gray',ls='--',label='naive expectation'); refs(ax)
        ax.set_yscale('log'); ax.set_xlabel('Alpha'); ax.set_ylabel('SWD'); ax.set_title('SWD across masking fractions'); ax.legend(fontsize=8); style.save_fig(fig,out/'E3_main.png')


def main():
    args=parser().parse_args()
    if args.steps < 1 or args.threads < 1 or not args.seeds: raise ValueError('Positive steps, threads, and at least one seed required')
    # Prevent a short run from being mistaken for a formal result.
    if args.steps != 1500 and args.output_dir.resolve() == Path(__file__).resolve().parents[1]:
        raise ValueError('Nonstandard step counts require an explicit separate --output-dir')
    torch.set_num_threads(args.threads)
    style=load_style(args.workspace_root.resolve())
    start=time.monotonic(); rows=[]; baseline_values={}; samples={}; dirs=toy.directions()
    conds=conditions(args.exp)
    output=args.output_dir.resolve()
    def stop_budget(reason, estimate=None):
        (output/'results').mkdir(parents=True,exist_ok=True)
        partial={'exp':args.exp,'status':'partial','seeds':args.seeds,'metrics':rows,'representative_seed':None,'runtime_sec':time.monotonic()-start,'stop_reason':reason,'estimated_total_sec':estimate,'parameters':{'steps':args.steps,'threads':args.threads},'environment':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'matplotlib':matplotlib.__version__,'platform':platform.platform()}}
        (output/'results'/f'{args.exp}.json').write_text(json.dumps(partial,indent=2,allow_nan=False)+'\n')
        raise RuntimeError(reason)
    print(f'Experiment {args.exp}: {len(conds)*len(args.seeds)} trainings, {args.steps} steps, {args.threads} CPU threads', flush=True)
    for c in conds:
        for seed in args.seeds:
            n=c['n']; data=toy.trajectories(n,seed); held=toy.trajectories(2000,seed+300000)
            print(f'Training {c} seed={seed}', flush=True)
            # Limit each run to 1.5 times the plan's per-condition experiment allocation.
            limit={'E1':360/12,'E2':720/36,'E3':360/18}[args.exp] * 1.5
            try:
                model, elapsed=toy.train(data,seed,c['schedule'],c['attention'],c['alpha'],c['rho'],args.steps,limit)
            except RuntimeError as exc:
                stop_budget(str(exc))
            estimate=elapsed / args.steps * 1500 * 66
            print(f'Training finished: {elapsed:.2f}s for {args.steps} steps; estimated 66 x 1500-step training: {estimate:.1f}s ({estimate/60:.1f} min), excluding evaluation',flush=True)
            if estimate > 1800:
                stop_budget('Budget gate: estimated 66 trainings exceed 30 minutes; stopping before evaluation', estimate)
            evaluation_start=time.monotonic()
            generated=toy.sample(model,seed)
            if args.exp=='E1': samples[(seed,label(c))]=generated
            bn=baseline_values.setdefault(str(n),{})
            if str(seed) not in bn: bn[str(seed)]=toy.baselines(data,held,seed,dirs)
            rows.append(dict(c,seed=seed,swd=toy.swd(generated,held,dirs),mem=toy.memory_ratio(generated,held,data),training_sec=elapsed))
            eval_sec=time.monotonic()-evaluation_start
            total_estimate=(elapsed / args.steps * 1500 + eval_sec)*66
            print(f'Evaluation: {eval_sec:.2f}s; estimated 66 training and evaluation runs: {total_estimate:.1f}s ({total_estimate/60:.1f} min)',flush=True)
            if total_estimate > 1800: stop_budget('Budget gate: estimated total including evaluation exceeds 30 minutes', total_estimate)
            print(f"SWD={rows[-1]['swd']:.5f} mem={rows[-1]['mem']:.5f}", flush=True)
    if args.exp=='E2':
        for row in rows:
            reference=next(r for r in rows if r['n']==row['n'] and r['seed']==row['seed'] and label(r)=='single+full')
            row['gain']=reference['swd']-row['swd']
    target=[r for r in rows if (label(r)=='dual+sep' and (args.exp!='E2' or r['n']==256) and (args.exp!='E3' or (r['alpha']==.25 and r['rho']==0)))]
    rep=style.representative_seed([next(r['swd'] for r in target if r['seed']==s) for s in args.seeds],args.seeds)
    result={'exp':args.exp,'status':'done' if args.steps==1500 and len(args.seeds)>=3 else 'partial','seeds':args.seeds,'metrics':rows,'baselines':baseline_values,'representative_seed':rep,'runtime_sec':time.monotonic()-start,'parameters':{'steps':args.steps,'threads':args.threads,'batch_size':128,'ema_decay':.99,'sampling_steps':50,'directions':256},'environment':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'matplotlib':matplotlib.__version__,'platform':platform.platform()}}
    output=args.output_dir.resolve(); (output/'results').mkdir(parents=True,exist_ok=True)
    (output/'results'/f'{args.exp}.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    plot(args.exp,rows,args.seeds,baseline_values,samples,style,output/'figures')
    print(f'Output: {output}; runtime {time.monotonic()-start:.1f}s',flush=True)


if __name__=='__main__':
    try: main()
    except Exception as exc:
        print(f'STOP: {exc}',file=sys.stderr,flush=True)
        sys.exit(1)
