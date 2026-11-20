"""Recompute Gaussian IC from cached charging rows without rebuilding SOH labels."""
import gc,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
from ..reconstruct_cycles import prepare_battery
from ..extract_features import charge_curve
from .config import ROOT,BATTERIES,FEATURES,file_sha,json_write


def gaussian_features(parts,config):
    result={k:np.nan for k in ['fixed_peak','full_cc_peak','partial_charge_ah','partial_dv_dq']}
    cc,cv,ccdur,cvdur,cvq,bad=charge_curve(parts)
    if bad or len(cc)<20:return result,'unreliable_or_short_cc_curve',None
    v,q=cc[:,0],cc[:,1]
    if v[0]>3.8 or v[-1]<4.0:return result,'partial_voltage_interval_not_covered',None
    partial=float(np.interp(4.,v,q)-np.interp(3.8,v,q))
    if not 1e-5<partial<1.5:return result,'partial_capacity_outside_bounds',None
    result['partial_charge_ah']=partial;result['partial_dv_dq']=.2/partial
    low=np.ceil(v[0]/config.voltage_grid_v)*config.voltage_grid_v
    high=np.floor(v[-1]/config.voltage_grid_v)*config.voltage_grid_v
    grid=np.arange(low,high+config.voltage_grid_v*.1,config.voltage_grid_v)
    if len(grid)<10:return result,'insufficient_voltage_grid',None
    qg=np.interp(grid,v,q)
    ic=gaussian_filter1d(np.gradient(qg,grid),sigma=config.gaussian_sigma_samples,mode='reflect')
    result['full_cc_peak']=float(np.max(ic))
    if grid[0]<=config.fixed_peak_low_v and grid[-1]>=config.fixed_peak_high_v:
        mask=(grid>=config.fixed_peak_low_v-1e-9)&(grid<=config.fixed_peak_high_v+1e-9)
        result['fixed_peak']=float(np.max(ic[mask]))
    for key in ['full_cc_peak','fixed_peak']:
        if not np.isfinite(result[key]) or not 0<result[key]<100:result[key]=np.nan
    missing=[k for k,val in result.items() if not np.isfinite(val)]
    return result,('missing_gaussian_features:'+','.join(missing)) if missing else '',(grid,ic)


def source_frames(config):
    frame=pd.read_csv(ROOT/'outputs/cx2_features_all.csv',low_memory=False)
    quality=pd.read_csv(ROOT/'outputs/cycle_quality.csv',low_memory=False).set_index('cycle_id')
    frame=frame[frame.battery.isin(BATTERIES)].copy()
    assert frame.cycle_id.is_unique
    aligned=quality.reindex(frame.cycle_id)
    if not np.array_equal(frame.validated_complete,aligned.validated_complete):raise ValueError('Feature/quality acceptance mismatch')
    for name in ['soh','validated_discharge_capacity_ah']:
        if not np.allclose(frame[name],aligned[name],equal_nan=True,atol=1e-10,rtol=0):raise ValueError(f'Quality mismatch: {name}')
    good=frame.validated_complete
    if not np.allclose(frame.loc[good,'soh'],frame.loc[good,'validated_discharge_capacity_ah']/1.35,atol=1e-10,rtol=0):raise ValueError('Invalid fractional SOH labels')
    for phase in ['charge','discharge']:
        a=frame.loc[good,phase+'_integral_ah'];b=frame.loc[good,phase+'_counter_ah']
        if not (np.isfinite(a)&np.isfinite(b)).all() or (abs(a-b)>np.maximum(.01,.05*np.maximum(a,b))+1e-10).any():raise ValueError('Capacity validation failed')
    frame['start']=pd.to_datetime(frame.start);frame['end']=pd.to_datetime(frame.end)
    return frame.sort_values(['battery','lifetime_cycle']).reset_index(drop=True)


def extract(config,output):
    output=Path(output);cache=output/'gaussian_features.csv';meta=output/'feature_cache_metadata.json'
    fingerprint=hashlib.sha256((file_sha(Path(__file__))+file_sha(ROOT/'src/extract_features.py')+file_sha(ROOT/'src/reconstruct_cycles.py')+file_sha(ROOT/'outputs/cx2_features_all.csv')+json.dumps(config.__dict__,sort_keys=True)).encode()).hexdigest()
    if cache.exists() and meta.exists() and json.loads(meta.read_text()).get('fingerprint')==fingerprint:
        data=pd.read_csv(cache,low_memory=False);data['start']=pd.to_datetime(data.start);data['end']=pd.to_datetime(data.end);return data
    frame=source_frames(config);manifest=pd.read_csv(ROOT/'outputs/inspection_manifest.csv');rows=[];domains=[];curves={}
    for battery in tqdm(BATTERIES,desc='Gaussian charging features'):
        all_b=frame[frame.battery==battery].copy();valid=all_b.validated_complete&all_b.protocol.eq('conventional')
        below=all_b[valid & (all_b.soh<config.eol_soh_fraction)]
        cutoff=int(below.lifetime_cycle.iloc[0]) if len(below) else int(all_b.lifetime_cycle.max())+1
        selected=all_b[all_b.lifetime_cycle<cutoff].copy()
        domains.append(dict(battery=battery,first_below_80_ordinal=cutoff,first_below_80_soh=float(below.soh.iloc[0]) if len(below) else None,
                            candidate_prefix_cycles=len(selected),validated_routine_prefix_cycles=int((selected.validated_complete&selected.protocol.eq('conventional')).sum())))
        # Reuse normalized raw-row caches and existing cycle timestamps/labels. No Excel reread,
        # cycle reconstruction call, label recomputation, or raw-file mutation is performed.
        raw,audit=prepare_battery(manifest,battery)
        samples=set(np.linspace(0,max(0,len(selected)-1),3,dtype=int));examples=[]
        for number,r in enumerate(tqdm(selected.to_dict('records'),desc=battery,leave=False)):
            reasons=[];f={k:np.nan for k in ['fixed_peak','full_cc_peak','partial_charge_ah','partial_dv_dq']}
            if not r['validated_complete']:reasons.append('quality_rejected')
            if r['protocol']!='conventional':reasons.append('nonconventional_protocol')
            if ';' in r['sources']:reasons.append('multiple_source_cycle')
            if r['local_counter_resets']>0:reasons.append('local_counter_reset')
            if not reasons:
                left=raw.timestamp.searchsorted(r['start'],side='left');right=raw.timestamp.searchsorted(r['end'],side='right')
                piece=raw.iloc[left:right]
                if len(piece)!=r['rows'] or set(piece.source)!=set(r['sources'].split(';')):
                    reasons.append('raw_interval_provenance_mismatch')
                else:
                    parts=[g for _,g in piece.groupby('segment',sort=False)]
                    f,reason,curve=gaussian_features(parts,config)
                    if reason:reasons.append(reason)
                    if number in samples and curve is not None:examples.append((int(r['lifetime_cycle']),curve))
            rows.append(dict(**{k:r[k] for k in ['battery','cycle_id','lifetime_cycle','start','end','sources','local_counter_resets','validated_complete','protocol','soh']},**f,
                             feature_exclusion=';'.join(reasons),eol_cutoff_ordinal=cutoff))
        curves[battery]=examples;del raw;gc.collect()
    data=pd.DataFrame(rows);data.to_csv(cache,index=False);json_write(output/'aging_domains.json',domains)
    import pickle
    with open(output/'ic_examples.pkl','wb') as f:pickle.dump(curves,f)
    json_write(meta,dict(fingerprint=fingerprint,inputs='Existing validated target table and normalized Channel row caches',gaussian_sigma_samples=1,voltage_grid_v=config.voltage_grid_v))
    return data
