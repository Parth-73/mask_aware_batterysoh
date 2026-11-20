"""Artifact-derived reports only; no missing-stage scores are invented."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .audit import OUT
from .data import FEATURES, GRID


def save(fig,name):
    fig.tight_layout()
    fig.savefig(OUT/'plots'/f'{name}.png',dpi=300,bbox_inches='tight')
    fig.savefig(OUT/'plots'/f'{name}.pdf',bbox_inches='tight')
    plt.close(fig)


def report():
    f=pd.read_csv(OUT/'processed/samples.csv')
    c=np.load(OUT/'processed/curves.npy')
    fig,axs=plt.subplots(3,1,figsize=(7,8),sharex=True)
    for index in np.linspace(0,len(c)-1,min(20,len(c)),dtype=int):
        for j,ax in enumerate(axs):ax.plot(GRID,c[index,j],alpha=.3,lw=.8)
    for ax,label in zip(axs,['Q (Ah)','dQ/dV (Ah/V)','I (A)']):ax.set_ylabel(label)
    axs[-1].set_xlabel('Measured charging voltage (V)');save(fig,'raw_curves')
    normals=list((OUT/'checkpoints').glob('*/normalization.json'))
    if normals:
        norm=json.loads(normals[0].read_text())
        normalized=(c-np.array(norm['curve_mean'])[None,:,None])/np.array(norm['curve_scale'])[None,:,None]
        fig,axs=plt.subplots(3,1,figsize=(7,8),sharex=True)
        for index in np.linspace(0,len(c)-1,min(20,len(c)),dtype=int):
            for j,ax in enumerate(axs):ax.plot(GRID,normalized[index,j],alpha=.3,lw=.8)
        for ax,label in zip(axs,['Scaled Q','Scaled dQ/dV','Scaled I']):ax.set_ylabel(label)
        axs[-1].set_xlabel('Charging voltage (V)');save(fig,'normalized_curves_train_scaler')
    fig,ax=plt.subplots(figsize=(8,4))
    for b,g in f.groupby('battery'):ax.plot(g.lifetime_cycle,g.soh,label=b,lw=.7)
    ax.set(xlabel='Observed reconstructed cycle ordinal',ylabel='Validated fractional SOH');ax.legend(ncol=3);save(fig,'soh_trajectories')
    fig,axs=plt.subplots(5,1,figsize=(8,11),sharex=True)
    for b,g in f.groupby('battery'):
        for ax,name in zip(axs,FEATURES):ax.plot(g.lifetime_cycle,g[name],label=b,lw=.7);ax.set_ylabel(name)
    axs[-1].set_xlabel('Observed reconstructed cycle ordinal');save(fig,'engineered_features')
    q=pd.read_csv(OUT/'data_audit/curve_quality.csv')
    q.groupby(['battery','reason'],dropna=False).size().rename('count').to_csv(OUT/'data_audit/curve_rejection_counts.csv')
    categories=q.reason.fillna('accepted').str.split(':').str[0].str.replace('_',' ')
    fig,ax=plt.subplots(figsize=(10,4));categories.value_counts().plot.barh(ax=ax);ax.set(xlabel='Candidates',title='Exclusive cohort quality categories; detailed reasons retained in CSV');save(fig,'curve_quality')
    e=pd.read_csv(OUT/'rest_events/rest_event_inventory.csv')
    v=e[e.category.eq('verified_measured_rest')]
    if len(v):
        fig,ax=plt.subplots(figsize=(7,4));ax.hist(v.duration_s,bins=40);ax.set(xlabel='Verified sampled rest duration (s)',ylabel='Events');save(fig,'verified_rest_duration')
    paths=list((OUT/'checkpoints').glob('*/metrics.csv'))
    comparisons='No completed training runs.'
    if paths:
        m=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True)
        comparisons=m[['model','split','seed','rmse_pp','mae_pp','epochs_completed','status']].to_csv(index=False)
        for p in paths:
            if json.loads((p.parent/'config.json').read_text())['label'].startswith('optuna'):continue
            run=p.parent
            for split in ['validation','outer']:
                pp=run/f'predictions_{split}.csv'
                if not pp.exists():continue
                table=pd.read_csv(pp)
                fig,ax=plt.subplots(figsize=(8,4));ax.plot(table.soh.to_numpy(),label='Validated');ax.plot(table.predicted_soh.to_numpy(),label='Predicted')
                ax.set(xlabel='Matched retained target-cycle order',ylabel='Fractional SOH',title=f'{run.name}: {split}');ax.legend();save(fig,f'{run.name}_{split}')
    from .full_report import write_report
    write_report()

if __name__=='__main__':report()
