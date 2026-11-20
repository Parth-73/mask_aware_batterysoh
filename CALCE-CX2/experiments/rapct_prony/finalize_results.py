"""Consolidate real artifacts, protocols and stage gates without manufacturing unavailable scores."""
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from src.common import ROOT,digest,save_json
from .audit import OUT
from .compare import compare
from .metrics import metrics,read_metrics
from .train import aggregate


def finalize():
    aggregate();compare()
    matched=pd.read_csv(OUT/'metrics/matched_stage_comparison.csv')
    matched[matched.stage.eq('A4')].to_csv(OUT/'metrics/physics_ablation.csv',index=False)
    matched[matched.stage.isin(['A0','A1','A2'])].to_csv(OUT/'metrics/feature_ablation.csv',index=False)
    matched.assign(evidence='single_seed_fixed_protocol_pilot_not_final_nested_result').to_csv(OUT/'metrics/final_comparison.csv',index=False)
    all_runs=read_metrics(OUT/'metrics/all_runs.csv')
    studies=[];nested=[];trials=[]
    for path in sorted((OUT/'optuna_studies').glob('*_best.json')):
        study=json.loads(path.read_text());studies.append(study)
        table=OUT/'optuna_studies'/f"{study['study']}_trials.csv"
        if table.exists():
            t=pd.read_csv(table);t['study']=study['study'];trials.append(t)
        runs=[]
        for cfgfile in (OUT/'checkpoints').glob('cnn_refit_*/config.json'):
            cfg=json.loads(cfgfile.read_text())
            if cfg['outer']==study['scope']['outer'] and cfg['channels']==study['parameters']['channels'] and cfg['lr']==study['parameters']['lr'] and cfg['epochs']==study['refit_epochs']:
                mfile=cfgfile.parent/'metrics.csv'
                if not mfile.exists():continue
                m=read_metrics(mfile);m['study']=study['study'];m['run']=cfgfile.parent.name;nested.append(m);runs.append(str(cfgfile.parent))
        study['refit_runs']=runs;save_json(path,study)
    if trials:pd.concat(trials,ignore_index=True).to_csv(OUT/'metrics/hyperparameter_trials.csv',index=False)
    nested_table=pd.concat(nested,ignore_index=True) if nested else pd.DataFrame()
    if len(nested_table):nested_table.to_csv(OUT/'metrics/nested_CNN_outer_metrics.csv',index=False)
    stability=[]
    if len(nested_table):
        for (study,battery),g in nested_table.groupby(['study','battery']):
            g=g.drop_duplicates('seed')
            stability.append(dict(protocol='nested_CNN_pilot',study=study,battery=battery,seed_count=len(g),rmse_fraction_mean=g.rmse.mean(),rmse_fraction_sample_std=g.rmse.std(ddof=1)))
    for run,g in matched[matched.split.eq('outer')].groupby('run'):
        stability.append(dict(protocol='primary_single_seed_pilot',study=run,battery=g.battery.iloc[0],seed_count=g.seed.nunique(),rmse_fraction_mean=g.rmse.mean(),rmse_fraction_sample_std=np.nan))
    pd.DataFrame(stability).to_csv(OUT/'metrics/seed_stability.csv',index=False)
    regions=[];prediction_index=[]
    for metricfile in (OUT/'checkpoints').glob('*/metrics.csv'):
        run=metricfile.parent;cfg=json.loads((run/'config.json').read_text())
        for split in ['validation','outer']:
            path=run/f'predictions_{split}.csv'
            if not path.exists():continue
            copy=OUT/'predictions'/f'{run.name}_{split}.csv'
            if copy.exists() and digest(copy)!=digest(path):raise ValueError(f'prediction_archive_changed: {copy}')
            if not copy.exists():shutil.copyfile(path,copy)
            prediction_index.append(dict(run=run.name,split=split,path=str(copy.relative_to(OUT)),sha256=digest(copy),protocol=cfg['label']))
            if cfg['label'].startswith(('optuna','nested_CNN_refit')):continue
            table=pd.read_csv(path);keep=matched[matched.run.eq(run.name)&matched.split.eq(split)]
            if keep.empty:continue
            # Temporal comparison targets are exactly the audited length-5 target IDs.
            windows=np.load(OUT/'processed/windows_L5.npy');samples=pd.read_csv(OUT/'processed/samples.csv')
            ids=set(samples.iloc[windows[:,-1]].cycle_id)
            table=table[table.cycle_id.isin(ids)].copy()
            table['region']=np.select([table.soh<.6,table.soh<.8],['below_0.6','0.6_to_0.8'],default='at_least_0.8')
            for region,g in table.groupby('region'):
                r=metrics(g.soh,g.predicted_soh);r.update(run=run.name,split=split,region=region,retrospective_evaluation_only=True);regions.append(r)
    pd.DataFrame(prediction_index).to_csv(OUT/'predictions/manifest.csv',index=False)
    pd.DataFrame(regions).to_csv(OUT/'metrics/soh_region_metrics.csv',index=False)
    support=pd.read_csv(OUT/'rest_events/causal_state_support.csv')
    support['prediction_error_available']=False;support['metric_type']='feasibility_counts_not_recovery_prediction_error'
    support.to_csv(OUT/'metrics/rest_event_metrics.csv',index=False)
    protected=json.loads((OUT/'data_audit/historical_snapshot.json').read_text())
    changes={p:dict(before=sha,after=digest(ROOT/p) if (ROOT/p).exists() else None) for p,sha in protected.items() if not (ROOT/p).exists() or digest(ROOT/p)!=sha}
    save_json(OUT/'data_audit/historical_preservation_check.json',dict(protected_file_count=len(protected),changed=changes,all_unchanged=not bool(changes)))
    source=ROOT/'experiments/rapct_prony'
    save_json(OUT/'configs/research_source_hashes.json',{str(p.relative_to(source)):digest(p) for p in source.rglob('*') if p.is_file() and 'vendor' not in p.parts and '__pycache__' not in p.parts and p.suffix in ['.py','.md','.txt']})
    status={
        'checkpoint_0':'executed_audit_and_rest_inventory; literature_full_text_verification_partial',
        'checkpoint_1':'MLP_CNN_fusion_pilots_executed; CNN_interrupted_then_weights_continued',
        'checkpoint_2':'frozen_and_joint_Transformer_pilots_executed; selected_on_validation_only',
        'checkpoint_3':'zero_and_nonzero_regularization_aging_pilots_executed',
        'checkpoint_4':'experimental_recovery_blocked_under_conservative_gap_policy',
        'checkpoint_5':'single_mode_synthetic_unit_tests_passed',
        'checkpoint_6':'multi_mode_synthetic_tests_and_24_seeded_identification_runs_executed',
        'checkpoint_7':'full_composition_synthetic_forward_tests_only; no_experimental_joint_training',
        'checkpoint_8':'bounded_nested_CNN_pilot_only; full_outer_study_and_other_component_searches_incomplete',
        'checkpoint_9':'interim_artifact_derived_research_report_not_final_scientific_confirmation'}
    save_json(OUT/'reports/stage_status.json',status)
    print(json.dumps(dict(completed_experimental_run_artifacts=len(list((OUT/'checkpoints').glob('*/metrics.csv'))),historical_files_unchanged=not bool(changes),studies=len(studies))),flush=True)

if __name__=='__main__':finalize()
