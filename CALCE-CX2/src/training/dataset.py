"""Audited, causal within-battery windows with train-only standardization."""
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from .config import ROOT,SPLITS,FEATURES,SEQUENCES,json_write,file_sha

@dataclass
class WindowSet:
    x:np.ndarray
    y:np.ndarray
    metadata:pd.DataFrame

    def subset(self,columns):
        positions=[FEATURES['F2'].index(f) for f in columns]
        return WindowSet(self.x[:,:,positions].copy(),self.y.copy(),self.metadata.copy())


def audit_source(feature_path,quality_path,rejected_path):
    f=pd.read_csv(feature_path,low_memory=False)
    q=pd.read_csv(quality_path,low_memory=False)
    rejected=pd.read_csv(rejected_path,low_memory=False)
    required=set(FEATURES['F2']+['cycle_id','battery','lifetime_cycle','start','end','sources','validated_complete','soh','validated_discharge_capacity_ah','protocol','rejection_reasons','local_counter_resets'])
    if required-set(f):raise ValueError(f'Missing required features/provenance: {sorted(required-set(f))}')
    if f.cycle_id.duplicated().any() or q.cycle_id.duplicated().any():raise ValueError('Duplicate cycle IDs prevent trustworthy alignment')
    wanted=sum(SPLITS.values(),[])
    if len(wanted)!=len(set(wanted)):raise ValueError('Battery splits overlap')
    f=f[f.battery.isin(wanted)].copy()
    if set(wanted)-set(f.battery):raise ValueError('Requested batteries missing')
    aligned=q.set_index('cycle_id').reindex(f.cycle_id)
    if aligned.battery.isna().any():raise ValueError('Feature record missing from quality table')
    if not np.array_equal(aligned.battery.to_numpy(),f.battery.to_numpy()):raise ValueError('Battery/quality alignment mismatch')
    if not np.array_equal(aligned.validated_complete.to_numpy(),f.validated_complete.to_numpy()):raise ValueError('Quality acceptance mismatch')
    for name in ['soh','validated_discharge_capacity_ah']:
        if not np.allclose(aligned[name],f[name],equal_nan=True,rtol=0,atol=1e-10):raise ValueError(f'Quality target mismatch: {name}')
    valid=f.validated_complete
    if set(f.loc[valid,'cycle_id'])&set(rejected.cycle_id):raise ValueError('A rejected cycle has a validated label')
    y=f.loc[valid,'soh'].to_numpy(float);cap=f.loc[valid,'validated_discharge_capacity_ah'].to_numpy(float)
    if not np.isfinite(y).all() or not (y>0).all() or not np.allclose(y,cap/1.35,atol=1e-10,rtol=0):raise ValueError('SOH is not a verified fractional capacity target')
    if f.loc[valid,'rejection_reasons'].astype('string').fillna('').str.len().gt(0).any():raise ValueError('Accepted label contains rejection reasons')
    for phase in ['charge','discharge']:
        a=f.loc[valid,phase+'_integral_ah'].to_numpy(float);b=f.loc[valid,phase+'_counter_ah'].to_numpy(float)
        if not (np.isfinite(a)&np.isfinite(b)).all() or (np.abs(a-b)>np.maximum(.01,.05*np.maximum(a,b))+1e-10).any():raise ValueError(f'{phase} capacity validation failed')
    f['start']=pd.to_datetime(f.start,errors='raise');f['end']=pd.to_datetime(f.end,errors='raise')
    if (f.end<f.start).any():raise ValueError('Negative experimental cycle duration')
    f['benchmark_split']=f.battery.map({b:s for s,bs in SPLITS.items() for b in bs})
    return f.sort_values(['battery','lifetime_cycle']).reset_index(drop=True)


def cycle_reasons(row,common_features):
    reasons=[]
    if not row.validated_complete:reasons.append('quality_rejected')
    if row.protocol!='conventional':reasons.append('nonconventional_protocol')
    if not np.isfinite(row.soh) or row.soh<=0:reasons.append('unverified_soh')
    missing=[f for f in common_features if not np.isfinite(float(row[f]))]
    if missing:reasons.append('missing_or_nonfinite_features:'+','.join(missing))
    if ';' in str(row.sources):reasons.append('multiple_source_cycle')
    if row.local_counter_resets>0:reasons.append('local_counter_reset')
    return ';'.join(reasons)


def build_windows(frame,length,feature_order,mean,scale,max_gap_s=3600.):
    """Evaluate every target candidate on the ORIGINAL ordinal sequence, before filtering."""
    xs=[];ys=[];metadata=[];audit=[]
    for battery,g in frame.groupby('battery',sort=True):
        g=g.sort_values('lifetime_cycle').reset_index(drop=True)
        ordinals=g.lifetime_cycle.to_numpy(int)
        for j,row in g.iterrows():
            reasons=[]
            if j<length-1:reasons.append('insufficient_history')
            else:
                w=g.iloc[j-length+1:j+1]
                if w.cycle_exclusion.str.len().gt(0).any():reasons.append('rejected_or_missing_cycle_in_window')
                if not np.all(np.diff(w.lifetime_cycle)==1):reasons.append('ordinal_gap')
                if w.sources.nunique()!=1 or w.sources.str.contains(';',regex=False).any():reasons.append('acquisition_session_boundary')
                gap=(w.start.iloc[1:].reset_index(drop=True)-w.end.iloc[:-1].reset_index(drop=True)).dt.total_seconds()
                if not gap.between(0,max_gap_s).all():reasons.append('acquisition_gap_or_overlap')
                if (w.local_counter_resets>0).any():reasons.append('counter_reset_in_window')
            audit.append(dict(battery=battery,split=row.benchmark_split,cycle_id=row.cycle_id,lifetime_cycle=int(row.lifetime_cycle),sequence_length=length,accepted=not reasons,rejection_reasons=';'.join(reasons)))
            if reasons:continue
            x=(w[feature_order].to_numpy(float)-mean)/scale
            if not np.isfinite(x).all():raise ValueError('Nonfinite value escaped the complete-case gate')
            xs.append(x.astype(np.float32));ys.append(float(row.soh))
            metadata.append(dict(battery=battery,split=row.benchmark_split,cycle_id=row.cycle_id,lifetime_cycle=int(row.lifetime_cycle),start=str(row.start),end=str(row.end),source=row.sources,
                                 input_cycle_ids=json.dumps(w.cycle_id.tolist()),input_ordinals=json.dumps(w.lifetime_cycle.astype(int).tolist()),
                                 input_start=str(w.start.iloc[0]),current_charge_start=str(row.start)))
    x=np.stack(xs) if xs else np.empty((0,length,len(feature_order)),np.float32)
    return WindowSet(x,np.asarray(ys,np.float32),pd.DataFrame(metadata)),pd.DataFrame(audit)


def prepare(output,max_gap_s=3600.,feature_path=None,quality_path=None,rejected_path=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    feature_path=Path(feature_path or ROOT/'outputs/cx2_features_all.csv');quality_path=Path(quality_path or ROOT/'outputs/cycle_quality.csv');rejected_path=Path(rejected_path or ROOT/'outputs/rejected_cycles.csv')
    frame=audit_source(feature_path,quality_path,rejected_path)
    order=FEATURES['F2']
    frame['cycle_exclusion']=frame.apply(lambda r:cycle_reasons(r,order),axis=1)
    train=frame[(frame.benchmark_split=='train')&frame.cycle_exclusion.eq('')]
    if train.empty:raise ValueError('No complete training rows')
    scaler=StandardScaler().fit(train[order])
    json_write(output/'feature_preprocessing.json',dict(method='StandardScaler',feature_order=order,mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),fit_split='train',fit_batteries=sorted(train.battery.unique()),fit_cycle_count=len(train),fit_cycle_ids_sha256=__import__('hashlib').sha256('\n'.join(train.cycle_id).encode()).hexdigest(),imputation='none; common complete cases',target_scaling='none; fraction'))
    frame[['battery','cycle_id','lifetime_cycle','benchmark_split','validated_complete','protocol','cycle_exclusion']].to_csv(output/'cycle_audit.csv',index=False)
    counts=[]
    for battery,g in frame.groupby('battery'):
        for name,cols in FEATURES.items():
            own=g.validated_complete&(g.protocol=='conventional')&np.isfinite(g[cols]).all(axis=1)
            counts.append(dict(battery=battery,split=g.benchmark_split.iloc[0],feature_set=name,candidates=len(g),validated_complete=int(g.validated_complete.sum()),individually_usable_feature_cycles=int(own.sum()),common_eligible_cycles=int(g.cycle_exclusion.eq('').sum())))
    pd.DataFrame(counts).to_csv(output/'cycle_counts.csv',index=False)
    prepared={};audits=[];samples=[];payload={}
    for length in SEQUENCES:
        whole,audit=build_windows(frame,length,order,scaler.mean_,scaler.scale_,max_gap_s)
        audits.append(audit)
        for split in SPLITS:
            mask=whole.metadata['split'].eq(split).to_numpy()
            subset=WindowSet(whole.x[mask],whole.y[mask],whole.metadata.loc[mask].reset_index(drop=True))
            if not len(subset.y):raise ValueError(f'No {split} windows of length {length}')
            prepared[(split,length)]=subset
            payload[f'{split}_{length}_x']=subset.x;payload[f'{split}_{length}_y']=subset.y
            subset.metadata.to_csv(output/f'windows_{split}_{length}.csv',index=False)
            for battery,g in subset.metadata.groupby('battery'):
                samples.append(dict(split=split,battery=battery,sequence_length=length,samples=len(g)))
    np.savez_compressed(output/'prepared_windows.npz',**payload)
    pd.concat(audits,ignore_index=True).to_csv(output/'window_audit.csv',index=False)
    pd.DataFrame(samples).to_csv(output/'sample_counts.csv',index=False)
    json_write(output/'source_audit.json',dict(source_files={str(p):file_sha(p) for p in [feature_path,quality_path,rejected_path]},soh_units='fraction',nominal_capacity_ah=1.35,
       selected_splits=SPLITS,features_exist=True,capacity_and_quality_agreement=True,no_discharge_inputs=True,
       smoothing_audit='Per-charge Savitzky-Golay smoothing; no across-cycle smoothing. No historical feature columns in any benchmark feature set.',
       input_features=order,common_cohort=True,split_target_range_audit_training_only={'train_min':float(train.soh.min()),'train_max':float(train.soh.max())}))
    return prepared
