"""Run the frozen SRA/Self-Flow toy plan, with a first-condition budget gate."""
import argparse
import importlib.util
import hashlib
import os
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
    p.add_argument('--topic-budget-minutes', type=float, default=30)
    p.add_argument('--experiment-budget-minutes', type=float)
    p.add_argument('--prior-runtime-sec', type=float, default=0, help='Topic time excluding this experiment, including smoke')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--plot-only', action='store_true', help='Redraw completed results without training')
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
    original_save = style.save_fig
    def save_fig(fig,path,**kwargs):
        path=Path(path)
        if path.name=='E1_main.png':
            fig.axes[0].set_title('Separating attention raises SWD in this toy')
        elif path.name=='E1_samples.png':
            old=fig._suptitle.get_text()
            fig.suptitle('Ellipse-like trajectories in all settings\n'+old)
        elif path.name=='E2_main.png':
            left,right=fig.axes
            left.set_title('Full attention improves as data grow')
            right.set_title('Dual+full gains do not decrease monotonically')
            point=next(r for r in rows if r['n']==4096 and r['seed']==0 and r['schedule']=='dual' and r['attention']=='sep')
            base=next(r for r in rows if r['n']==4096 and r['seed']==0 and r['schedule']=='single' and r['attention']=='full')
            for ax,y,offset in [(left,point['swd'],(-120,-30)),(right,base['swd']-point['swd'],(-120,12))]:
                ax.scatter([4096],[y],s=28,facecolors='none',edgecolors=style.SERIES[3],zorder=7)
                ax.annotate('Counterexample: seed 0',xy=(4096,y),xytext=offset,textcoords='offset points',fontsize=8,ha='right',arrowprops={'arrowstyle':'->','color':style.INK_MUTED,'lw':.8})
        elif path.name=='E2_mem.png':
            ax=fig.axes[0]
            ax.set_title('No consistent small-data memory advantage')
            point=next(r for r in rows if r['n']==4096 and r['seed']==0 and r['schedule']=='dual' and r['attention']=='sep')
            ax.scatter([4096],[point['mem']],s=28,facecolors='none',edgecolors=style.SERIES[3],zorder=7)
            ax.annotate('Counterexample: seed 0',xy=(4096,point['mem']),xytext=(-20,-25),textcoords='offset points',fontsize=8,ha='right',arrowprops={'arrowstyle':'->','color':style.INK_MUTED,'lw':.8})
        elif path.name=='E3_main.png':
            means={rho:sum(r['swd'] for r in rows if r['alpha']==.5 and r['rho']==rho)/3 for rho in (0.,.25)}
            fig.axes[0].set_title(f'At alpha=0.5, rho={min(means,key=means.get):g} has lower mean SWD')
        for ax in fig.axes:
            legend=ax.get_legend()
            if legend:
                handles, labels = ax.get_legend_handles_labels()
                ncols = 4 if path.name == 'E1_main.png' else 3
                legend = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, -.28), ncols=ncols, fontsize=8)
                legend.set_frame_on(True)
                frame=legend.get_frame()
                frame.set_facecolor(style.SURFACE)
                frame.set_alpha(1)
                frame.set_linewidth(0)
                frame.set_edgecolor('none')
                legend.set_zorder(20)
        return original_save(fig,path,**kwargs)
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
        refs(ax); ax.set_xticks(range(4), labels, rotation=15); ax.set_yscale('log'); ax.set_ylabel('SWD'); ax.set_title('Lowest mean SWD: '+labels[int(np.argmin([np.mean([r['swd'] for r in rows if label(r)==lab]) for lab in labels]))]); ax.legend(fontsize=8)
        save_fig(fig, out/'E1_main.png')
        representative = style.representative_seed([next(r['swd'] for r in get(seed=s, schedule='dual', attention='sep')) for s in seeds], seeds)
        fig, axes = plt.subplots(5, 12, figsize=(14, 6), sharex=True, sharey=True)
        grids = [samples[(representative,lab)] for lab in labels] + [toy.trajectories(2000, representative+300000)]
        for i, data in enumerate(grids):
            for j in range(12):
                xy = data[j].numpy(); xy = np.concatenate((xy,xy[:1]))
                axes[i,j].plot(xy[:,0],xy[:,1],lw=.7); axes[i,j].set_aspect('equal'); axes[i,j].axis('off')
            axes[i,0].text(-.1,.5, (labels+['held-out'])[i], transform=axes[i,0].transAxes, ha='right', va='center', fontsize=8)
        fig.suptitle(f'Generated and held-out trajectories: seed {representative}, first 12 (unselected)'); save_fig(fig, out/'E1_samples.png')
    elif exp == 'E2':
        xs = [64,256,1024,4096]; labs = [('single','full'),('dual','full'),('dual','sep')]
        fig, (ax, gain) = plt.subplots(1,2,figsize=(11,4),sharex=True)
        base = matrix('n', xs, 'swd', schedule='single', attention='full')
        for i,(s,a) in enumerate(labs):
            vals = matrix('n',xs,'swd',schedule=s,attention=a)
            style.seed_lines(ax,xs,vals,label=s+'+'+a,color=style.SERIES[[0,1,3][i]])
            if s == 'dual': style.seed_lines(gain,xs,base-vals,label=s+'+'+a,color=style.SERIES[[0,1,3][i]])
        for key,ls in [('real',':'),('replay','--'),('gaussian','-.')]:
            ax.plot(xs,[np.mean([baselines[str(n)][str(s)][key] for s in seeds]) for n in xs], color='gray',ls=ls,lw=1,label=key)
        g0=(base-matrix('n',xs,'swd',schedule='dual',attention='full'))[:,0].mean()
        gain.axhline(0,color='gray',lw=.8); gain.axhline(g0,color='gray',ls='--',label='naive expectation')
        for a in (ax,gain): a.set_xscale('log'); a.set_xlabel('Training samples'); a.legend(fontsize=8)
        ax.set_yscale('log'); ax.set_ylabel('SWD'); gain.set_ylabel('Paired SWD gain'); ax.set_title('Lowest SWD at n=4096: '+labs[int(np.argmin([matrix('n',xs,'swd',schedule=s,attention=a)[:,-1].mean() for s,a in labs]))][0]+'+'+labs[int(np.argmin([matrix('n',xs,'swd',schedule=s,attention=a)[:,-1].mean() for s,a in labs]))][1]); gain.set_title('Paired gain across dataset sizes')
        save_fig(fig,out/'E2_main.png')
        fig,ax=plt.subplots(figsize=(6,4))
        for i,(s,a) in enumerate(labs): style.seed_lines(ax,xs,matrix('n',xs,'mem',schedule=s,attention=a),label=s+'+'+a,color=style.SERIES[[0,1,3][i]])
        ax.set_xscale('log'); ax.set_xlabel('Training samples'); ax.set_ylabel('Memory ratio'); ax.set_title('Memory ratio across dataset sizes'); ax.legend(); save_fig(fig,out/'E2_mem.png')
    else:
        xs=[.25,.375,.5]; fig,ax=plt.subplots(figsize=(6,4))
        for i,rho in enumerate([0.,.25]): style.seed_lines(ax,xs,matrix('alpha',xs,'swd',rho=rho),label=f'rho={rho}',color=style.SERIES[i])
        ax.axhline(matrix('alpha',xs,'swd',rho=0.)[:,0].mean(),color='gray',ls='--',label='naive expectation'); refs(ax)
        ax.set_yscale('log'); ax.set_xlabel('Alpha'); ax.set_ylabel('SWD'); ax.set_title('Lowest mean SWD at alpha=0.5: rho='+str([0.,.25][int(np.argmin([matrix('alpha',xs,'swd',rho=r)[:,-1].mean() for r in [0.,.25]]))])); ax.legend(fontsize=8); save_fig(fig,out/'E3_main.png')


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


def main():
    args = parser().parse_args()
    if args.plot_only:
        topic = args.output_dir.resolve()
        result_path = topic / 'results' / f'{args.exp}.json'
        if not result_path.exists(): raise ValueError(f'Missing result: {result_path}')
        data = json.loads(result_path.read_text())
        if data.get('status') != 'done': raise ValueError(f'{result_path} is not done')
        style = load_style(args.workspace_root.resolve())
        samples = {}
        if args.exp == 'E1':
            matches = []
            for record_path in (topic / 'results' / 'runs').glob('E1-*.json'):
                record = json.loads(record_path.read_text())
                if record.get('exp') != 'E1' or record.get('status') != 'done': continue
                sample_dir = Path(record['run_dir']) / 'topics' / topic.name / 'runs'
                state_path = sample_dir / 'E1_state.json'
                if not state_path.exists(): continue
                state = json.loads(state_path.read_text())
                if state.get('status') == 'done' and state.get('config') == data.get('parameters'):
                    matches.append((sample_dir, state))
            if len(matches) != 1: raise ValueError(f'Expected one matching completed E1 run; found {len(matches)}')
            sample_dir, state = matches[0]
            for entry in state['entries'].values():
                row = entry['row']
                samples[(row['seed'], label(row))] = torch.load(sample_dir / entry['samples'], map_location='cpu', weights_only=True)
        plot(args.exp, data['metrics'], data['seeds'], data['baselines'], samples, style, topic / 'figures')
        print(f'Redrawn {args.exp} figures in {topic / "figures"}', flush=True)
        return
    if args.steps < 1 or args.threads < 1 or not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError('Positive steps, threads and unique seeds required')
    exp_budget = args.experiment_budget_minutes if args.experiment_budget_minutes is not None else {'E1':6., 'E2':12., 'E3':6.}[args.exp]
    if min(args.topic_budget_minutes, exp_budget) <= 0 or args.prior_runtime_sec < 0:
        raise ValueError('Budgets must be positive and prior runtime nonnegative')
    output = args.output_dir.resolve()
    if args.steps != 1500 and output == Path(__file__).resolve().parents[1]:
        raise ValueError('Nonstandard step counts require an explicit separate --output-dir')
    torch.set_num_threads(args.threads)
    style = load_style(args.workspace_root.resolve())
    conds = conditions(args.exp)
    code_hash = hashlib.sha256(Path(__file__).read_bytes() + Path(toy.__file__).read_bytes()).hexdigest()
    env = {'python':platform.python_version(),'torch':str(torch.__version__),'numpy':np.__version__,'matplotlib':matplotlib.__version__,'platform':platform.platform()}
    config = {'exp':args.exp,'seeds':args.seeds,'steps':args.steps,'threads':args.threads,'conditions':conds,'code_sha256':code_hash,'environment':env,'batch_size':128,'ema_decay':.99,'sampling_steps':50,'directions':256,'optimizer':'AdamW lr=0.001 weight_decay=0','held_out':2000,'generated':2000}
    state_path = output/'runs'/f'{args.exp}_state.json'
    result_path = output/'results'/f'{args.exp}.json'
    if state_path.exists():
        if not args.resume: raise ValueError('Existing state: pass --resume or use a new output directory')
        state = json.loads(state_path.read_text())
        if state['config'] != config: raise ValueError('Resume configuration, environment, or code differs from saved state')
    else:
        if args.resume: raise ValueError('No state exists to resume')
        state = {'config':config,'entries':{},'runtime_sec':0.0,'status':'partial'}
    previous_runtime = state['runtime_sec']
    start = time.monotonic()
    dirs = toy.directions()
    def runtime(): return previous_runtime + time.monotonic()-start
    def limits():
        return min(exp_budget*60*1.5-runtime(), args.topic_budget_minutes*60-args.prior_runtime_sec-runtime())
    def save(reason=None, estimate=None):
        state['runtime_sec']=runtime()
        state['stop_reason']=reason
        state['estimated_total_sec']=estimate
        atomic_json(state_path,state)
        entries = list(state['entries'].values())
        rows = [e['row'] for e in entries if 'row' in e]
        bases = {}
        for e in entries:
            if 'row' in e: bases.setdefault(str(e['row']['n']),{})[str(e['row']['seed'])]=e['baseline']
        result = {'exp':args.exp,'status':'partial','seeds':args.seeds,'metrics':rows,'baselines':bases,'representative_seed':None,'runtime_sec':state['runtime_sec'],'stop_reason':reason,'estimated_total_sec':estimate,'parameters':config,'budgets':{'topic_minutes':args.topic_budget_minutes,'experiment_minutes':exp_budget,'prior_runtime_sec':args.prior_runtime_sec},'environment':env}
        atomic_json(result_path,result)
        return result
    def check_budget():
        if limits() <= 0: raise RuntimeError('Experiment or topic cumulative wall-clock budget exceeded')
    print(f'Experiment {args.exp}: {len(conds)*len(args.seeds)} trainings, {args.steps} steps, {args.threads} CPU threads',flush=True)
    try:
        check_budget()
        for ci,c in enumerate(conds):
            for seed in args.seeds:
                key=f'{ci}_{seed}'
                entry=state['entries'].get(key, {})
                if 'row' in entry:
                    print(f'Resume: completed {key}',flush=True)
                    continue
                n=c['n']; data=toy.trajectories(n,seed); held=toy.trajectories(2000,seed+300000)
                checkpoint=output/'runs'/f'{args.exp}_{key}_ema.pt'
                if entry.get('checkpoint'):
                    model=toy.Model()
                    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True))
                    model.eval().requires_grad_(False)
                    elapsed=entry['training_sec']
                    print(f'Resume: EMA checkpoint {key}',flush=True)
                else:
                    print(f'Training {c} seed={seed}',flush=True)
                    model,elapsed=toy.train(data,seed,c['schedule'],c['attention'],c['alpha'],c['rho'],args.steps,limits())
                    checkpoint.parent.mkdir(parents=True,exist_ok=True)
                    tmp=checkpoint.with_suffix('.tmp'); torch.save(model.state_dict(),tmp); os.replace(tmp,checkpoint)
                    entry={'checkpoint':checkpoint.name,'training_sec':elapsed}
                    state['entries'][key]=entry
                    save()
                estimate=elapsed/args.steps*1500*66
                print(f'Training: {elapsed:.2f}s; estimated 66 x 1500-step training: {estimate/60:.1f} min, excluding evaluation',flush=True)
                if not args.resume and args.exp=='E1' and ci==0 and seed==args.seeds[0] and args.prior_runtime_sec+estimate>args.topic_budget_minutes*60:
                    raise RuntimeError('First-condition training estimate exceeds topic budget')
                check_budget()
                evaluation_start=time.monotonic()
                generated=toy.sample(model,seed)
                baseline=toy.baselines(data,held,seed,dirs)
                row=dict(c,seed=seed,swd=toy.swd(generated,held,dirs),mem=toy.memory_ratio(generated,held,data),training_sec=elapsed)
                eval_sec=time.monotonic()-evaluation_start
                sample_path=output/'runs'/f'{args.exp}_{key}_samples.pt'
                tmp=sample_path.with_suffix('.tmp'); torch.save(generated,tmp); os.replace(tmp,sample_path)
                entry.update(row=row,baseline=baseline,samples=sample_path.name,evaluation_sec=eval_sec)
                save()
                total_estimate=args.prior_runtime_sec+(elapsed/args.steps*1500+eval_sec)*66
                print(f'Evaluation: {eval_sec:.2f}s; estimated 66 trainings and evaluations: {total_estimate/60:.1f} min; SWD={row["swd"]:.5f} mem={row["mem"]:.5f}',flush=True)
                if not args.resume and args.exp=='E1' and ci==0 and seed==args.seeds[0] and total_estimate>args.topic_budget_minutes*60:
                    raise RuntimeError('First-condition training and evaluation estimate exceeds topic budget')
                check_budget()
        result=save()
        rows=result['metrics']; bases=result['baselines']
        if args.exp=='E2':
            for row in rows:
                reference=next(r for r in rows if r['n']==row['n'] and r['seed']==row['seed'] and label(r)=='single+full')
                row['gain']=reference['swd']-row['swd']
        samples={}
        if args.exp=='E1':
            for entry in state['entries'].values():
                row=entry['row']; samples[(row['seed'],label(row))]=torch.load(output/'runs'/entry['samples'],map_location='cpu',weights_only=True)
        target=[r for r in rows if label(r)=='dual+sep' and (args.exp!='E2' or r['n']==256) and (args.exp!='E3' or (r['alpha']==.25 and r['rho']==0))]
        result['representative_seed']=style.representative_seed([next(r['swd'] for r in target if r['seed']==s) for s in args.seeds],args.seeds)
        summaries=[]
        for c in conds:
            group=[r for r in rows if all(r[k]==v for k,v in c.items())]
            stats={}
            for metric in ['swd','mem'] + (['gain'] if args.exp=='E2' else []):
                values=[r[metric] for r in group]
                stats[metric]={'mean':float(np.mean(values)),'min':min(values),'max':max(values)}
            summaries.append({'condition':c,'metrics':stats})
        result['summary']=summaries
        plot(args.exp,rows,args.seeds,bases,samples,style,output/'figures')
        check_budget()
        result['runtime_sec']=runtime()
        result['status']='done' if args.steps==1500 and set(args.seeds)=={0,1,2} else 'partial'
        result['stop_reason']=None
        atomic_json(result_path,result)
        state['status']=result['status']; state['runtime_sec']=runtime(); atomic_json(state_path,state)
        print(f'Output: {output}; cumulative runtime {runtime():.1f}s',flush=True)
    except Exception as exc:
        save(str(exc))
        raise


if __name__=='__main__':
    try: main()
    except Exception as exc:
        print(f'STOP: {exc}',file=sys.stderr,flush=True)
        sys.exit(1)
