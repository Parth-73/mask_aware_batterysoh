"""Nested battery-fold Optuna search. Pruning compares the same completed inner folds.

The executable pilot searches the CNN, then refits its frozen selected configuration
on all non-outer cells. Later component searches remain separate research work.
"""
import argparse
import hashlib
import json
from dataclasses import asdict,replace
from pathlib import Path
import joblib
import numpy as np
import optuna
import pandas as pd
from src.common import save_json
from .audit import OUT
from .metrics import read_metrics
from .data import BATTERIES
from .train import Config,train,aggregate


def execute(config):
    identifier=hashlib.sha256(json.dumps(asdict(config),sort_keys=True).encode()).hexdigest()[:12]
    name=config.pathway+('_refit' if config.refit_all_inner else '')+'_'+identifier
    run=OUT/'checkpoints'/name
    if (run/'metrics.csv').exists():
        if json.loads((run/'dataset_manifest.json').read_text())!=json.loads((OUT/'processed/manifest.json').read_text()):raise ValueError('cached_trial_dataset_changed')
        scores=read_metrics(run/'metrics.csv')
        v=scores[scores.split.eq('validation')]
        return float(v.rmse.mean()) if len(v) else float('nan'),run
    if run.exists():
        if (run/'last.pt').exists():return train(config,resume_run=run)
        raise ValueError(f'incomplete_trial_without_recovery_checkpoint: {run}')
    # Reuse exact completed numeric configurations even when their study/trial labels differ.
    core={k:v for k,v in asdict(config).items() if k!='label'}
    for saved in (OUT/'checkpoints').glob('*/config.json'):
        prior=json.loads(saved.read_text())
        if {k:v for k,v in prior.items() if k!='label'}==core and (saved.parent/'metrics.csv').exists():
            if json.loads((saved.parent/'dataset_manifest.json').read_text())!=json.loads((OUT/'processed/manifest.json').read_text()):continue
            m=read_metrics(saved.parent/'metrics.csv');v=m[m.split.eq('validation')]
            return float(v.rmse.mean()) if len(v) else float('nan'),saved.parent
    # Suppress outer evaluation for inner search, without using trial-level per-epoch reporting.
    class FoldOnly:
        def report(self,*args):pass
        def should_prune(self):return False
    return train(config,trial=None if config.refit_all_inner else FoldOnly())


def nested(outer,inner,trials=10,epochs=30,seeds=(42,123,2026),study_label='cnn_nested'):
    if outer in inner or len(set(inner))!=len(inner) or set(inner)-set(BATTERIES):raise ValueError('invalid_inner_battery_folds')
    data=json.loads((OUT/'processed/manifest.json').read_text())
    scope=dict(outer=outer,inner_batteries=inner,epochs=epochs,search_stage='raw_curve_cnn',dataset=data['samples_sha256'])
    scope_id=hashlib.sha256(json.dumps(scope,sort_keys=True).encode()).hexdigest()[:10]
    name=f'{study_label}_{outer}_{scope_id}';db=OUT/'optuna_studies'/f'{name}.sqlite3'
    storage=optuna.storages.RDBStorage(url='sqlite:///'+db.as_posix(),heartbeat_interval=60,grace_period=120)
    sampler_path=OUT/'optuna_studies'/f'{name}_sampler.joblib'
    sampler=joblib.load(sampler_path) if sampler_path.exists() else optuna.samplers.TPESampler(seed=42)
    study=optuna.create_study(study_name=name,storage=storage,load_if_exists=True,direction='minimize',
        sampler=sampler,pruner=optuna.pruners.MedianPruner(n_startup_trials=2,n_warmup_steps=0))
    save_json(OUT/'configs'/f'{name}.json',dict(**scope,sampler='TPESampler(seed=42)',pruner='MedianPruner_at_equivalent_completed_inner_fold_count',
        requested_trials=trials,final_seeds=list(seeds),status='bounded_nested_CNN_pilot_not_full_all_model_study'))
    def objective(trial):
        channels=trial.suggest_categorical('channels',[16,32,64])
        lr=trial.suggest_float('lr',1e-4,1e-3,log=True)
        scores=[];paths=[];best_epochs=[];training_times=[];peaks=[]
        for index,validation in enumerate(inner):
            config=Config(pathway='cnn',outer=outer,validation=validation,epochs=epochs,channels=channels,lr=lr,
                label=f'optuna_{name}_trial{trial.number}_inner{validation}')
            score,run=execute(config);scores.append(score);paths.append(str(run))
            checkpoint=__import__('torch').load(run/'best.pt',map_location='cpu',weights_only=True)
            best_epochs.append(checkpoint['epoch'])
            m=read_metrics(run/'metrics.csv');training_times.append(float(m.training_time_s.iloc[0]));peaks.append(int(m.peak_gpu_memory_bytes.iloc[0]))
            trial.set_user_attr('inner_runs',paths);trial.set_user_attr('inner_scores',scores);trial.set_user_attr('best_epochs',best_epochs)
            trial.set_user_attr('training_time_s',sum(training_times));trial.set_user_attr('peak_gpu_memory_bytes',max(peaks))
            trial.report(float(np.mean(scores)),step=index)
            if trial.should_prune():raise optuna.TrialPruned()
        return float(np.mean(scores))
    completed=len([t for t in study.trials if t.state in [optuna.trial.TrialState.COMPLETE,optuna.trial.TrialState.PRUNED]])
    if completed<trials:study.optimize(objective,n_trials=trials-completed,callbacks=[lambda s,t:joblib.dump(s.sampler,sampler_path)])
    study.trials_dataframe().to_csv(OUT/'optuna_studies'/f'{name}_trials.csv',index=False)
    best=study.best_trial
    refit_epochs=max(1,int(round(np.median(best.user_attrs['best_epochs']))))
    selection=dict(study=name,best_trial=best.number,inner_battery_macro_rmse=best.value,parameters=best.params,
        refit_epochs=refit_epochs,epoch_policy='median_best_epoch_across_inner_batteries',scope=scope,
        outer_selection_used=False,universal_optimum_claim=False)
    save_json(OUT/'optuna_studies'/f'{name}_best.json',selection)
    refit_runs=[]
    for seed in seeds:
        final=Config(pathway='cnn',outer=outer,validation=inner[0],epochs=refit_epochs,channels=best.params['channels'],lr=best.params['lr'],seed=seed,
            refit_all_inner=True,label=f'nested_CNN_refit_{name}')
        _,run=execute(final);refit_runs.append(str(run))
    selection['refit_runs']=refit_runs
    save_json(OUT/'optuna_studies'/f'{name}_best.json',selection)
    aggregate()
    return selection


def main():
    p=argparse.ArgumentParser();p.add_argument('--outer',nargs='+',default=BATTERIES);p.add_argument('--inner',nargs='+')
    p.add_argument('--trials',type=int,default=10);p.add_argument('--epochs',type=int,default=30);p.add_argument('--seeds',nargs='+',type=int,default=[42,123,2026]);p.add_argument('--study-label',default='cnn_nested')
    a=p.parse_args()
    for outer in a.outer:
        inner=a.inner or [b for b in BATTERIES if b!=outer]
        print(json.dumps(nested(outer,inner,a.trials,a.epochs,a.seeds,a.study_label)),flush=True)

if __name__=='__main__':main()
