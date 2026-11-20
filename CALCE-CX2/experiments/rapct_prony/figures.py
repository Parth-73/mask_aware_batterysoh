"""Publication-resolution figures derived only from completed real/synthetic artifacts."""
import json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .audit import OUT
from .report import save
from .memory import NeuralProny


def generate():
    m=pd.read_csv(OUT/'metrics/matched_stage_comparison.csv')
    names=sorted(m.run.unique())
    fig,ax=plt.subplots(figsize=(12,5));x=np.arange(len(names))
    for split,shift in [('validation',-.18),('outer',.18)]:
        values=m[m.split.eq(split)].set_index('run').reindex(names).rmse_pp
        ax.bar(x+shift,values,width=.36,label=split)
    ax.set_xticks(x,[s.rsplit('_',1)[0].replace('_',' ') for s in names],rotation=30,ha='right');ax.set_ylabel('RMSE (SOH percentage points)');ax.legend()
    ax.set_title('Matched target cycles; single-seed pilots with different stage budgets');save(fig,'matched_staged_ablation')
    from matplotlib.lines import Line2D
    fig,(ax,key)=plt.subplots(1,2,figsize=(11,5),gridspec_kw={'width_ratios':[3,2]})
    handles=[]
    outer=m[m.split.eq('outer')].set_index('run')
    for index,name in enumerate(names):
        metric=pd.read_csv(OUT/'checkpoints'/name/'metrics.csv');parameters=int(metric.parameter_count.iloc[0])
        color=plt.get_cmap('tab10')(index)
        error=float(outer.loc[name,'rmse_pp']);ax.scatter(parameters,error,color=color,s=45)
        ax.annotate(str(index+1),(parameters,error),xytext=(5,12 if 'guided' in name else -13 if 'zero' in name else 6),textcoords='offset points')
        label=name.rsplit('_',1)[0].replace('_',' ')
        handles.append(Line2D([0],[0],marker='o',linestyle='',color=color,label=f'{index+1}. {label}'))
    ax.margins(x=.15,y=.12);ax.set(xlabel='Stored parameters (including frozen)',ylabel='Matched outer RMSE (SOH pp)',title='Matched primary pilot targets')
    key.axis('off');key.legend(handles=handles,loc='center',frameon=False);save(fig,'error_complexity')
    fig,ax=plt.subplots(figsize=(7,4))
    for path in (OUT/'checkpoints').glob('physics_*/physical_audit.csv'):
        run=path.parent;table=pd.read_csv(run/'predictions_outer.csv')
        if 'guided' not in run.name:continue
        for _,g in table.groupby('segment_id'):ax.plot(g.lifetime_cycle,g.degradation,color='tab:blue',alpha=.6,lw=.8)
    ax.set(xlabel='Observed cycle ordinal',ylabel='Estimated irreversible loss (fraction)',title='Latent aging: lines stop at unknown-history boundaries');save(fig,'irreversible_latent_segments')
    a=pd.read_csv(OUT/'metrics/prony_ablation.csv')
    fig,ax=plt.subplots(figsize=(7,4))
    for learnable,offset in [(False,-.16),(True,.16)]:
        g=a[a.learnable_timescales.eq(learnable)].groupby('modes').heldout_penalty_rmse_fraction.agg(['mean','std'])
        ax.bar(np.arange(4)+offset,g['mean'],yerr=g['std'],width=.32,label='Learnable times' if learnable else 'Fixed times')
    ax.set_xticks(np.arange(4),[1,2,4,8]);ax.set(xlabel='Memory modes',ylabel='Synthetic penalty RMSE (fraction)',title='Synthetic biexponential identification; three seeds');ax.legend();save(fig,'synthetic_prony_mode_ablation')
    directory=OUT/'checkpoints/synthetic_memory/M4_learn1_seed42'
    p=pd.read_csv(directory/'predictions.csv');layer=NeuralProny(4)
    checkpoint=torch.load(directory/'best.pt',weights_only=True);layer.load_state_dict(checkpoint['model'])
    t=torch.tensor(p.time_s.to_numpy(),dtype=torch.float32)
    with torch.no_grad():states=(.1*torch.sigmoid(checkpoint['initial_raw']))[None]*torch.exp(-t[:,None]/layer.timescales)
    fig,axs=plt.subplots(2,1,figsize=(8,7),sharex=True)
    axs[0].plot(p.time_s,p.true_synthetic_penalty,label='Known synthetic response');axs[0].plot(p.time_s,p.predicted_penalty,'--',label='Fitted Prony');axs[0].legend();axs[0].set_ylabel('Penalty (fraction)')
    for mode in range(4):axs[1].plot(p.time_s,states[:,mode],label=f'Mode {mode+1}')
    axs[1].set(xlabel='Synthetic verified rest duration (s)',ylabel='Synthetic memory state');axs[1].legend();save(fig,'synthetic_memory_trajectories')
    fig,ax=plt.subplots(figsize=(7,4))
    for modes in [1,2,4,8]:
        for path in (OUT/'checkpoints/synthetic_memory').glob(f'M{modes}_learn1_seed*/parameters.json'):
            tau=json.loads(path.read_text())['timescales_s'];ax.scatter([modes]*len(tau),tau,alpha=.6)
    ax.set_yscale('log');ax.set_xticks([1,2,4,8]);ax.set(xlabel='Memory modes',ylabel='Fitted synthetic timescales (s)',title='Synthetic estimates; no unique electrochemical interpretation');save(fig,'synthetic_timescales')
    for path in (OUT/'optuna_studies').glob('*_trials.csv'):
        trials=pd.read_csv(path);completed=trials[trials.state.eq('COMPLETE')].sort_values('number')
        fig,ax=plt.subplots(figsize=(7,4));ax.plot(completed.number,completed.value,'o-',label='Completed inner battery-macro RMSE')
        ax.plot(completed.number,completed.value.cummin(),'--',label='Best completed score')
        ax.set(xlabel='Trial number',ylabel='Inner validation RMSE (fraction)',title=f"{path.stem}: {int(trials.state.eq('PRUNED').sum())} pruned")
        ax.legend(fontsize=8);save(fig,path.stem+'_history')
        fig,ax=plt.subplots(figsize=(7,4));ax.axis('off');ax.text(.5,.5,f'Hyperparameter importance not estimated\nOnly {len(completed)} completed trials\nA larger study is required',ha='center',va='center')
        save(fig,path.stem+'_importance_unavailable')
    # Stage-3 ablation contains the same encoder and state recurrence in both settings.
    subset=m[m.stage.eq('A4')]
    fig,ax=plt.subplots(figsize=(7,4))
    for label,g in subset.groupby('split'):
        ax.plot(g.run.str.contains('guided').astype(int),g.rmse_pp,'o-',label=label)
    ax.set_xticks([0,1],['Zero regularization','Warm-up regularization']);ax.set_ylabel('RMSE (SOH percentage points)');ax.legend();save(fig,'physics_regularization_ablation')

if __name__=='__main__':generate()
