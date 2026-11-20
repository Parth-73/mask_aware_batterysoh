"""Within-battery halves, training-only trend corrections and [0,1] normalization."""
from dataclasses import dataclass
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from .config import FEATURES,json_write

@dataclass
class Windows:
    x:np.ndarray
    y:np.ndarray
    metadata:pd.DataFrame


def split_roles(frame,config):
    f=frame.sort_values('lifetime_cycle').reset_index(drop=True).copy()
    good=f.feature_exclusion.fillna('').eq('') & np.isfinite(f[['fixed_peak','full_cc_peak','partial_charge_ah','partial_dv_dq']]).all(axis=1)
    positions=np.flatnonzero(good)
    n_train=int(len(positions)*config.train_fraction)
    n_fit=int(n_train*(1-config.validation_fraction_of_train))
    if n_fit<20 or n_train-n_fit<10 or len(positions)-n_train<20:raise ValueError('Too few chronological records for the fixed paper split')
    fit_end=int(f.lifetime_cycle.iloc[positions[n_fit-1]]);train_end=int(f.lifetime_cycle.iloc[positions[n_train-1]])
    f['common_valid']=good
    f['role']=np.select([f.lifetime_cycle<=fit_end,f.lifetime_cycle<=train_end],['fit','validation'],default='test')
    return f,dict(common_eligible_cycles=len(positions),training_half_cycles=n_train,internal_fit_cycles=n_fit,internal_validation_cycles=n_train-n_fit,test_half_cycles=len(positions)-n_train,fit_end_ordinal=fit_end,train_end_ordinal=train_end,split_before_windowing=True)


def robust_trend(ordinal,values,adjacent,config):
    x=np.asarray(ordinal,float);y=np.asarray(values,float)
    origin=float(x.min());span=max(float(np.ptp(x)),1.);z=(x-origin)/span;good=np.isfinite(y)
    for _ in range(6):
        coefficients=np.polyfit(z[good],y[good],deg=config.trend_degree)
        residual=y-np.polyval(coefficients,z);center=np.median(residual[good]);mad=np.median(abs(residual[good]-center))
        sigma=max(1.4826*float(mad),.005*abs(float(np.median(y[good]))),1e-6)
        updated=np.isfinite(y)&(abs(residual-center)<=config.trend_sigma_threshold*sigma)
        if updated.sum()<config.trend_degree+3 or np.array_equal(updated,good):break
        good=updated
    diff=abs(np.diff(y))[np.asarray(adjacent,bool)]
    median_diff=float(np.median(diff)) if len(diff) else 0.
    diff_sigma=1.4826*float(np.median(abs(diff-median_diff))) if len(diff) else 0.
    jump=max(median_diff+config.trend_sigma_threshold*diff_sigma,config.jump_level_floor*abs(float(np.median(y[good]))),1e-6)
    return dict(coefficients=coefficients.tolist(),origin=origin,span=span,residual_center=float(center),residual_limit=config.trend_sigma_threshold*sigma,jump_limit=jump,fit_records=int(len(y)),robust_inliers=int(good.sum()))


def process(frame,variant,fit_mask,config):
    """Fit on fit_mask only; transform current observations with frozen parameters.

    Correction requires BOTH a trend residual and an abrupt adjacent raw-input jump.
    Gradual test aging is not overwritten merely for extrapolating outside train range.
    """
    f=frame.copy();fit=f[fit_mask & f.common_valid]
    if len(fit)<20:raise ValueError('Too few fitting features')
    parameters={};corrections=[];values={};processing_valid=f.common_valid.to_numpy().copy()
    for source,name,bounds in [(variant,'ic_peak_height',(0,100)),('partial_charge_ah','partial_charge_ah',(1e-5,1.5))]:
        adjacent=(np.diff(fit.lifetime_cycle)==1)&fit.sources.iloc[1:].reset_index(drop=True).eq(fit.sources.iloc[:-1].reset_index(drop=True)).to_numpy()
        trend=robust_trend(fit.lifetime_cycle,fit[source],adjacent,config);parameters[name]=trend
        raw=f[source].to_numpy(float);corrected=raw.copy();x=(f.lifetime_cycle.to_numpy(float)-trend['origin'])/trend['span'];prediction=np.polyval(trend['coefficients'],x)
        residual=abs(raw-prediction-trend['residual_center'])
        same=(f.lifetime_cycle.diff()==1)&f.sources.eq(f.sources.shift())&f.common_valid&f.common_valid.shift(fill_value=False)
        spikes=same.to_numpy()&(residual>trend['residual_limit'])&(abs(np.r_[0,np.diff(raw)])>trend['jump_limit'])
        for idx in np.flatnonzero(spikes):
            usable=np.isfinite(prediction[idx]) and bounds[0]<prediction[idx]<bounds[1]
            corrected[idx]=prediction[idx] if usable else np.nan
            if not usable:processing_valid[idx]=False
            corrections.append(dict(battery=f.battery.iloc[idx],cycle_id=f.cycle_id.iloc[idx],lifetime_cycle=int(f.lifetime_cycle.iloc[idx]),role=f.role.iloc[idx],feature=name,raw_value=float(raw[idx]),corrected_value=float(corrected[idx]),reason='large_adjacent_jump_and_training_trend_residual',action='corrected' if usable else 'rejected_unphysical_trend_value'))
        values[name]=corrected
    values['partial_dv_dq']=.2/values['partial_charge_ah']
    matrix=np.column_stack([values[k] for k in FEATURES]);processing_valid &= np.isfinite(matrix).all(axis=1)
    scaler=MinMaxScaler(feature_range=(0,1),clip=False).fit(matrix[fit_mask.to_numpy() & processing_valid])
    scaled=scaler.transform(matrix)
    for j,name in enumerate(FEATURES):f[name+'_corrected']=matrix[:,j]
    f['processing_valid']=processing_valid
    metadata=dict(trends=parameters,scaler=dict(method='MinMaxScaler',range=[0,1],clip=False,feature_order=FEATURES,data_min=scaler.data_min_.tolist(),data_max=scaler.data_max_.tolist(),scale=scaler.scale_.tolist(),min=scaler.min_.tolist(),fit_cycle_ids=f.loc[fit_mask & f.processing_valid,'cycle_id'].tolist()),target_transform='none; validated fractional SOH',fit_roles=sorted(f.loc[fit_mask & f.common_valid,'role'].unique()))
    return f,scaled,metadata,pd.DataFrame(corrections,columns=['battery','cycle_id','lifetime_cycle','role','feature','raw_value','corrected_value','reason','action'])


def make_windows(frame,scaled,config):
    result={};audit=[];buckets={role:([],[],[]) for role in ['fit','validation','test']};length=config.sequence_length
    for j,row in frame.iterrows():
        reasons=[]
        if j<length-1:reasons.append('insufficient_history')
        else:
            w=frame.iloc[j-length+1:j+1]
            if not w.processing_valid.all():reasons.append('rejected_or_missing_input')
            if not np.all(np.diff(w.lifetime_cycle)==1):reasons.append('ordinal_gap')
            if w.sources.nunique()!=1 or w.sources.str.contains(';',regex=False).any():reasons.append('session_boundary')
            gaps=(w.start.iloc[1:].reset_index(drop=True)-w.end.iloc[:-1].reset_index(drop=True)).dt.total_seconds()
            if not gaps.between(0,config.max_acquisition_gap_s).all():reasons.append('acquisition_gap_or_overlap')
            if (w.local_counter_resets>0).any():reasons.append('counter_reset')
        audit.append(dict(battery=row.battery,cycle_id=row.cycle_id,lifetime_cycle=int(row.lifetime_cycle),role=row.role,accepted=not reasons,rejection_reasons=';'.join(reasons)))
        if reasons:continue
        xs,ys,meta=buckets[row.role];xs.append(scaled[j-length+1:j+1].astype(np.float32));ys.append(float(row.soh))
        meta.append(dict(battery=row.battery,cycle_id=row.cycle_id,lifetime_cycle=int(row.lifetime_cycle),role=row.role,source=row.sources,input_cycle_ids=json.dumps(w.cycle_id.tolist()),input_ordinals=json.dumps(w.lifetime_cycle.astype(int).tolist()),start=str(row.start),end=str(row.end)))
    for role,(xs,ys,meta) in buckets.items():
        result[role]=Windows(np.stack(xs) if xs else np.empty((0,length,3),np.float32),np.asarray(ys,np.float32),pd.DataFrame(meta))
    return result,pd.DataFrame(audit)


def prepare(frame,battery,variant,config,output):
    directory=output/'datasets'/f'{battery}_{variant}';directory.mkdir(parents=True,exist_ok=True)
    f,split=split_roles(frame[frame.battery==battery],config)
    inner,scaled,inner_parameters,inner_corrections=process(f,variant,f.role.eq('fit'),config)
    iw,ia=make_windows(inner,scaled,config)
    final,scaled,final_parameters,final_corrections=process(f,variant,f.role.isin(['fit','validation']),config)
    fw,fa=make_windows(final,scaled,config)
    train=Windows(np.concatenate([fw['fit'].x,fw['validation'].x]),np.concatenate([fw['fit'].y,fw['validation'].y]),pd.concat([fw['fit'].metadata,fw['validation'].metadata],ignore_index=True))
    windows={'inner_train':iw['fit'],'inner_validation':iw['validation'],'final_train':train,'test':fw['test']}
    for role,w in windows.items():
        if len(w.y)<10:raise ValueError(f'{battery}/{variant}/{role}: fewer than 10 continuous windows')
        w.metadata.to_csv(directory/f'{role}_windows.csv',index=False)
    np.savez_compressed(directory/'windows.npz',**{role+'_'+kind:getattr(w,kind) for role,w in windows.items() for kind in ['x','y']})
    inner.to_csv(directory/'inner_processed_features.csv',index=False);final.to_csv(directory/'final_processed_features.csv',index=False)
    inner_corrections.to_csv(directory/'inner_corrections.csv',index=False);final_corrections.to_csv(directory/'final_corrections.csv',index=False)
    ia.to_csv(directory/'inner_window_audit.csv',index=False);fa.to_csv(directory/'final_window_audit.csv',index=False)
    json_write(directory/'preprocessing.json',dict(split=split,inner=inner_parameters,final=final_parameters,variant=variant,window_counts={role:len(w.y) for role,w in windows.items()},no_test_fitting=True))
    return windows
