"""Export the historical compact five-feature cohort without changing original artifacts."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import shutil
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parent
FEATURES=['cc_duration_s','cv_duration_s','dva_peak','ic_peak_voltage','ic_peak_width']


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def write_json(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def main():
    if (HERE/'all_data.csv').exists():raise FileExistsError('Existing export preserved. Use a new folder for a new export.')
    source=PROJECT/'outputs/cx2_features_all.csv'
    quality_source=PROJECT/'outputs/cycle_quality.csv'
    config_source=PROJECT/'outputs/lstm_baselines/experiment_config.json'
    selection_source=PROJECT/'outputs/lstm_baselines/feature_selection.json'
    definitions_source=PROJECT/'outputs/feature_definitions.csv'
    sources=[source,quality_source,config_source,selection_source,definitions_source]
    before={str(p.relative_to(PROJECT)):sha(p) for p in sources}
    config=json.loads(config_source.read_text(encoding='utf-8-sig'))
    selection=json.loads(selection_source.read_text(encoding='utf-8-sig'))
    assert config['features']['F3']==FEATURES
    splits=config['splits']
    batteries=sum(splits.values(),[])
    assert len(batteries)==len(set(batteries))
    frame=pd.read_csv(source,low_memory=False,float_precision='round_trip')
    quality=pd.read_csv(quality_source,low_memory=False,float_precision='round_trip').set_index('cycle_id')
    assert not frame.cycle_id.duplicated().any() and not quality.index.duplicated().any()
    audit=[];accepted=[]
    for index,row in frame.iterrows():
        reasons=[]
        if row.battery not in batteries:reasons.append('outside_conventional_benchmark_batteries')
        if not bool(row.validated_complete):reasons.append('original_cycle_quality_rejected')
        if row.protocol!='conventional':reasons.append('nonconventional_discharge_protocol')
        if not np.isfinite(row.soh):reasons.append('missing_validated_soh')
        missing=[name for name in FEATURES if not np.isfinite(row[name])]
        if missing:reasons.append('missing_features:'+','.join(missing))
        audit.append(dict(battery=row.battery,cycle_id=row.cycle_id,included=not reasons,
            exclusion_reasons=';'.join(reasons),original_rejection_reasons=row.rejection_reasons,
            original_quality_flags=row.quality_flags,original_feature_quality_flags=row.feature_quality_flags))
        if not reasons:accepted.append(index)
    selected=frame.loc[accepted].copy().sort_values(['battery','lifetime_cycle']).reset_index(drop=True)
    assert np.isfinite(selected[FEATURES+['soh']]).all().all()
    original=quality.loc[selected.cycle_id]
    assert original.validated_complete.all()
    assert np.allclose(selected.soh,original.validated_discharge_capacity_ah.to_numpy()/1.35,rtol=0,atol=1e-10)
    assert np.allclose(selected.soh,original.soh.to_numpy(),rtol=0,atol=1e-10)
    selected=selected.rename(columns={'lifetime_cycle':'cycle_ordinal','start':'cycle_start','sources':'source_identifiers'})
    selected['split']=selected.battery.map({b:split for split,bs in splits.items() for b in bs})
    selected.insert(0,'row_id',np.arange(len(selected)))
    metadata=['row_id','battery','cycle_id','cycle_ordinal','cycle_start','source_identifiers','split']
    data=selected[metadata+FEATURES+['soh']]
    data.to_csv(HERE/'all_data.csv',index=False)
    pd.DataFrame(audit).to_csv(HERE/'row_audit.csv',index=False)
    for directory in ['model_inputs','standardized','source_evidence']:(HERE/directory).mkdir(exist_ok=True)
    train=data[data.split.eq('train')]
    scaler=StandardScaler().fit(train[FEATURES])
    for split,bs in splits.items():
        subset=data[data.split.eq(split)]
        assert len(subset) and set(subset.battery)==set(bs)
        subset.to_csv(HERE/f'{split}.csv',index=False)
        subset[FEATURES].to_csv(HERE/'model_inputs'/f'X_{split}.csv',index=False)
        subset[['soh']].to_csv(HERE/'model_inputs'/f'y_{split}.csv',index=False)
        pd.DataFrame(scaler.transform(subset[FEATURES]),columns=FEATURES).to_csv(HERE/'standardized'/f'X_{split}.csv',index=False)
        restored=pd.read_csv(HERE/f'{split}.csv',float_precision='round_trip')
        assert np.array_equal(restored.soh.to_numpy(),subset.soh.to_numpy())
        assert np.array_equal(restored[FEATURES].to_numpy(),subset[FEATURES].to_numpy())
    normalized=pd.read_csv(HERE/'standardized/X_train.csv').to_numpy()
    assert np.allclose(normalized.mean(axis=0),0,atol=1e-10)
    assert np.allclose(normalized.std(axis=0,ddof=0),1,atol=1e-10)
    definitions=pd.read_csv(definitions_source).set_index('feature').loc[FEATURES].reset_index()
    definitions.to_csv(HERE/'feature_definitions.csv',index=False)
    units=dict(zip(definitions.feature,definitions.units))
    write_json(HERE/'feature_order.json',dict(features=FEATURES,target='soh',target_units='fraction',raw_feature_units=units,
        forbidden_predictors=metadata+['soh','any_same_cycle_discharge_measurement'],prediction_cutoff='end_of_available_current_charge',
        ordering_is_not_an_importance_ranking=True))
    write_json(HERE/'split_definition.json',dict(splits=splits,source=str(config_source.relative_to(PROJECT)),
        note='Frozen historical LSTM benchmark split; independent of the currently edited outputs/battery_splits.json. Not untouched external validation.'))
    write_json(HERE/'standardized/scaler.json',dict(method='StandardScaler',formula='(x-training_mean)/training_scale',feature_order=FEATURES,
        training_batteries=splits['train'],training_rows=len(train),mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),
        variance=scaler.var_.tolist(),target_scaled=False,new_cross_validation_requires_refitting=True))
    statistics=[]
    eligible_train=frame[frame.battery.isin(splits['train'])&frame.validated_complete&frame.protocol.eq('conventional')]
    for name in FEATURES:
        within=[float(spearmanr(group[name],group.soh).statistic) for _,group in train.groupby('battery')]
        statistics.append(dict(feature=name,units=units[name],training_rows=len(train),training_mean=float(train[name].mean()),
            training_std_population=float(train[name].std(ddof=0)),training_min=float(train[name].min()),training_max=float(train[name].max()),
            available_fraction_before_complete_case_filter=float(np.isfinite(eligible_train[name]).mean()),
            pooled_training_spearman=float(spearmanr(train[name],train.soh).statistic),
            mean_within_battery_training_spearman=float(np.mean(within)),
            association_is_not_causation_or_individual_feature_importance=True))
    pd.DataFrame(statistics).to_csv(HERE/'training_feature_statistics.csv',index=False)
    counts=data.groupby(['split','battery']).size().rename('rows').reset_index()
    counts.to_csv(HERE/'sample_counts.csv',index=False)
    for path in [config_source,selection_source,definitions_source]:shutil.copyfile(path,HERE/'source_evidence'/path.name)
    after={str(p.relative_to(PROJECT)):sha(p) for p in sources}
    assert before==after,'Original source files changed during export.'
    artifacts={str(p.relative_to(HERE)):sha(p) for p in HERE.rglob('*') if p.is_file() and p.suffix in ['.csv','.json']}
    write_json(HERE/'manifest.json',dict(source_hashes=before,output_hashes=artifacts,total_rows=len(data),
        split_rows=data.split.value_counts().to_dict(),predictors=FEATURES,target='soh',nominal_capacity_ah=1.35,
        target_clipped=False,target_smoothed=False,missing_features_imputed=False,feature_values_modified_in_original_unit_csvs=False,
        normalization_fitted_on_training_only=True,source_files_unchanged=True,source_curve_cohort_requirement=False,
        packages={name:importlib.metadata.version(name) for name in ['numpy','pandas','scipy','scikit-learn']},
        historical_selection=selection,seed=42))
    write_json(HERE/'verification.json',dict(finite_predictors_and_targets=True,unique_cycle_ids=True,
        target_matches_original_validated_capacity=True,raw_csv_roundtrip_exact=True,train_only_standardization_verified=True,
        battery_splits_disjoint=True,original_sources_unchanged=True,total_rows=len(data),split_rows=data.split.value_counts().to_dict()))
    manifest=json.loads((HERE/'manifest.json').read_text())
    manifest['output_hashes']={str(p.relative_to(HERE)):sha(p) for p in HERE.rglob('*') if p.is_file() and p.name!='manifest.json' and '__pycache__' not in p.parts}
    write_json(HERE/'manifest.json',manifest)
    print(counts.to_string(index=False));print('Total ready rows:',len(data))

if __name__=='__main__':main()
