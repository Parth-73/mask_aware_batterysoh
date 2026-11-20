"""Validate frozen curves, audit actual rest observations, and freeze matched cohorts."""
import argparse
import importlib.metadata as md
import json
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from src.common import ROOT, digest
from src.training.config import json_write
from ..audit import OUT as PARENT
from ..data import FEATURES, BATTERIES, GRID, curve_tensor
from src.reconstruct_cycles import prepare_battery, reconstruct

OUT = PARENT/'staged'
SOURCE = ROOT/'experiments/rapct_prony'
FOLDERS = ['data_audit','rest_events','processed','features','splits','configs','optuna_studies',
           'checkpoints','scalers','predictions','latent_states','metrics','plots','reports','research']


def windows(samples, length):
    frame = samples.sort_values(['battery','lifetime_cycle']).reset_index(drop=True)
    assert frame.cycle_id.tolist() == samples.cycle_id.tolist(), 'Cached arrays must stay aligned'
    indices, rows = [], []
    for battery, g in frame.groupby('battery', sort=True):
        ids = g.index.to_numpy()
        if len(g)<length:
            continue
        w = np.lib.stride_tricks.sliding_window_view(ids, length)
        ordinal = frame.lifetime_cycle.to_numpy()[w]
        source = frame.sources.to_numpy()[w]
        starts = pd.to_datetime(frame.start).astype('int64').to_numpy()[w]
        ends = pd.to_datetime(frame.end).astype('int64').to_numpy()[w]
        # pandas 3 may store datetime microseconds; convert explicitly to ns.
        starts = pd.to_datetime(frame.start).to_numpy(dtype='datetime64[ns]')[w]
        ends = pd.to_datetime(frame.end).to_numpy(dtype='datetime64[ns]')[w]
        gaps = (starts[:,1:]-ends[:,:-1])/np.timedelta64(1,'s')
        accepted = ((np.diff(ordinal,axis=1)==1).all(1) & (source==source[:,:1]).all(1) &
                    ~np.char.find(source.astype(str), ';').__ge__(0).any(1) &
                    ((gaps>=0)&(gaps<=3600)).all(1) &
                    (frame.local_counter_resets.to_numpy()[w]==0).all(1))
        for window, ok in zip(w, accepted):
            target = int(window[-1])
            rows.append({'battery': battery, 'cycle_id': frame.cycle_id.iloc[target], 'length': length,
                         'accepted': bool(ok), 'input_cycle_ids': json.dumps(frame.cycle_id.iloc[window].tolist()),
                         'reason': '' if ok else 'ordinal_source_clock_gap_or_reset'})
            if ok:
                indices.append(window)
    return np.asarray(indices, dtype=np.int64).reshape(-1,length), pd.DataFrame(rows)


def validate_cached_data():
    manifest = json.loads((PARENT/'processed/manifest.json').read_text())
    for name, key in [('samples.csv','samples_sha256'), ('curves.npy','curves_sha256')]:
        if digest(PARENT/'processed'/name)!=manifest[key]:
            raise ValueError(f'Cached {name} hash differs from its manifest')
    f = pd.read_csv(PARENT/'processed/samples.csv')
    curves = np.load(PARENT/'processed/curves.npy', mmap_mode='r')
    old = pd.read_csv(ROOT/'outputs/cx2_features_all.csv', low_memory=False).set_index('cycle_id')
    aligned = old.loc[f.cycle_id]
    assert not f.cycle_id.duplicated().any()
    assert set(f.battery)==set(BATTERIES)
    assert curves.shape==(len(f),3,len(GRID))
    assert np.isfinite(curves).all() and np.isfinite(f[FEATURES]).all().all()
    assert np.allclose(f[FEATURES], aligned[FEATURES], rtol=0, atol=1e-9)
    assert np.allclose(f.soh, aligned.validated_discharge_capacity_ah/1.35, rtol=0, atol=1e-10)
    assert aligned.validated_complete.all() and aligned.protocol.eq('conventional').all()
    assert (pd.to_datetime(f.start)<=pd.to_datetime(f.prediction_cutoff)).all()
    assert (pd.to_datetime(f.prediction_cutoff)<pd.to_datetime(f.end)).all()
    return f, curves


def rest_audit(samples):
    """Verify cached raw observations for each rest fragment; merge only shared endpoints."""
    prior = pd.read_csv(PARENT/'rest_events/rest_event_inventory.csv')
    m = pd.read_csv(PARENT/'data_audit/inspection_manifest.csv')
    m = m[m.status.eq('ready')].copy()
    source_manifest = {str(r.source_file)+'::'+str(r.sheet):r for r in m.itertuples()}
    prior['event_start'] = pd.to_datetime(prior.event_start)
    prior['event_end'] = pd.to_datetime(prior.event_end)
    records = []
    for source, group in prior.groupby('source_identifiers', sort=True):
        raw = None
        record = source_manifest.get(source)
        if record is not None:
            raw = pd.read_pickle(record.cache)
            raw = raw.sort_values(['timestamp','source_row'], kind='stable')
            times = raw.timestamp.to_numpy(dtype='datetime64[ns]')
        for row in group.sort_values(['event_start','event_end']).to_dict('records'):
            row['original_fragment_count'] = 1
            if row['category']=='verified_measured_rest':
                reason = ''
                if raw is None:
                    reason='source_not_uniquely_available'
                elif float(record.source_timestamp_reversals)>0 or float(record.test_time_resets)>0:
                    reason='source_clock_reset'
                else:
                    lo = np.searchsorted(times, np.datetime64(row['event_start']), side='left')
                    hi = np.searchsorted(times, np.datetime64(row['event_end']), side='right')
                    observed = raw.iloc[lo:hi]
                    dt = observed.timestamp.diff().dt.total_seconds().to_numpy()[1:]
                    test_dt = observed.test_s.diff().to_numpy()[1:]
                    if len(observed)<2 or not np.isfinite(observed.current).all():
                        reason='insufficient_finite_current_observations'
                    elif observed.current.abs().max()>.005:
                        reason='active_current_at_event_boundary_or_interior'
                    elif ((dt<0)|(dt>180)|(test_dt<0)|(np.abs(dt-test_dt)>2)).any():
                        reason='unobserved_or_inconsistent_clock_interval'
                    row['observed_points'] = len(observed)
                    row['current_abs_max_a'] = float(observed.current.abs().max()) if len(observed) else np.nan
                if reason:
                    row.update(category='uncertain_recording_gap', verification_status=False, uncertainty_reason=reason)
            if (records and row['category']=='verified_measured_rest' and
                records[-1]['category']=='verified_measured_rest' and
                records[-1]['source_identifiers']==source and records[-1]['event_end']==row['event_start'] and
                records[-1]['preceding_cycle_id']==row['preceding_cycle_id'] and
                records[-1]['following_cycle_id']==row['following_cycle_id']):
                records[-1]['event_end']=row['event_end']
                records[-1]['duration_s']=(row['event_end']-records[-1]['event_start']).total_seconds()
                records[-1]['original_fragment_count']+=1
                records[-1]['observed_points']=records[-1].get('observed_points',0)+row.get('observed_points',0)-1
                records[-1]['current_abs_max_a']=max(records[-1]['current_abs_max_a'],row['current_abs_max_a'])
            else:
                records.append(row)
    events = pd.DataFrame(records).sort_values(['battery','event_start','source_identifiers']).reset_index(drop=True)
    events.insert(0,'event_id',[f'rest_audit_{i:06d}' for i in range(len(events))])
    events['prediction_cutoff'] = events.following_cycle_id.map(samples.set_index('cycle_id').prediction_cutoff)
    events['available_for_cached_target'] = events.prediction_cutoff.notna() & (events.event_end<=pd.to_datetime(events.prediction_cutoff))
    events.to_csv(OUT/'rest_events/rest_event_inventory.csv',index=False)
    verified = events[events.category.eq('verified_measured_rest')]
    usable = verified[verified.validated_pair & verified.consistent_discharge_conditions & verified.available_for_cached_target]
    stats = {'raw_verified_fragments_before_reaudit': int(prior.category.eq('verified_measured_rest').sum()),
             'verified_observed_intervals': len(verified), 'uncertain_gaps': int(events.category.eq('uncertain_recording_gap').sum()),
             'events_per_battery': verified.groupby('battery').size().to_dict(),
             'duration_quantiles_s': {str(k):float(v) for k,v in verified.duration_s.quantile([0,.25,.5,.75,.95,1]).items()},
             'validated_pre_post_pairs': int(verified.validated_pair.sum()),
             'consistent_condition_pairs': int(verified.consistent_discharge_conditions.sum()),
             'usable_pre_cutoff_event_pairs': len(usable), 'unique_eligible_targets': usable.following_cycle_id.nunique(),
             'eligible_batteries': sorted(usable.battery.unique()),
             'verified_rests_over_one_hour': int(verified.duration_s.gt(3600).sum()),
             'independent_long_rest_recovery_evaluation': False,
             'recovery_gate': 'synthetic_memory_only; ordinary short protocol waits lack irregular long-rest excitation and independent recovery-state identification',
             'interpretation': 'No claim that short rests have zero capacity effect. Labels never trigger events. Discharge-condition comparability is retrospective audit metadata only.'}
    json_write(OUT/'rest_events/feasibility.json',stats)
    (OUT/'reports/REST_DATA_FEASIBILITY.md').write_text('# Raw-observation rest feasibility\n\n```json\n'+json.dumps(stats,indent=2)+'\n```\n\nVerified means finite measured |I|<=0.005 A, same source without recorder resets, adjacent observed/test clock intervals <=180 seconds and agreement within two seconds. Zero-duration timestamp ties may connect observed intervals; only exact shared endpoints with the same mapped cycles are merged. Gaps, active boundary observations and overlapping/uncertain sources are never interpreted as rest. Only events ending by the target charge cutoff are available as inputs. Event count is not a count of independent recovery interventions. No hour-scale verified rests are assumed from filenames or sessions. For the irregular long-rest recovery question, experimental A6-A8 remain unavailable; synthetic memory results do not validate capacity recovery.\n',encoding='utf-8')
    return stats


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--raw',type=Path,default=ROOT.parents[1]/'data')
    args=parser.parse_args()
    for name in FOLDERS:
        (OUT/name).mkdir(parents=True,exist_ok=True)
    # Snapshot every pre-existing report/configuration and source without rewriting it.
    protected = [*SOURCE.glob('*.py'),SOURCE/'research/RELATED_WORK.md',
                 *sorted((PARENT/'configs').glob('*.json')), *sorted((PARENT/'reports').glob('*.md')),
                 ROOT/'outputs/battery_splits.json', ROOT/'outputs/feature_definitions.csv',
                 ROOT/'outputs/feature_sets.json', ROOT/'outputs/soh_four_model_benchmarks/experiment_manifest.json']
    snapshot = {str(p.relative_to(ROOT)):digest(p) for p in protected if p.is_file()}
    dest=OUT/'data_audit/preexisting_snapshot.json'
    if not dest.exists():
        json_write(dest,snapshot)
        for p in (PARENT/'reports').glob('*.md'):
            shutil.copy2(p,OUT/'reports'/('preserved_initial_'+p.name))
    elif json.loads(dest.read_text())!=snapshot:
        raise ValueError('Pre-existing artifacts changed since this continuation audit')
    f,c=validate_cached_data()
    # Verify actual raw contents, not modification times.
    prior_raw=pd.read_csv(PARENT/'data_audit/raw_hashes.csv')
    actual=[]
    for row in prior_raw.itertuples():
        p=args.raw/Path(row.path)
        if digest(p)!=row.sha256:
            raise ValueError(f'Raw hash changed: {row.path}')
        actual.append({'path':row.path,'bytes':p.stat().st_size,'sha256':row.sha256})
    pd.DataFrame(actual).to_csv(OUT/'data_audit/raw_hashes.csv',index=False)
    f.to_csv(OUT/'processed/samples.csv',index=False)
    for length in [5,10,20]:
        idx,audit=windows(f,length)
        np.save(OUT/'processed'/f'indices_L{length}.npy',idx)
        audit.to_csv(OUT/'data_audit'/f'window_audit_L{length}.csv',index=False)
        target=f.iloc[idx[:,-1]].copy()
        target['input_cycle_ids']=[json.dumps(f.cycle_id.iloc[w].tolist()) for w in idx]
        target.to_csv(OUT/'splits'/f'targets_L{length}.csv',index=False)
    metadata={'samples':len(f),'batteries':BATTERIES,'curve_shape':list(c.shape),
              'samples_sha256':digest(PARENT/'processed/samples.csv'),'curves_sha256':digest(PARENT/'processed/curves.npy'),
              'raw_hashes_sha256':digest(OUT/'data_audit/raw_hashes.csv'),
              'preprocessing':'Reuse validated caches; no repeated full raw reconstruction claimed',
              'input_contract':'Current end-of-charge; no target discharge or post-cutoff rest inputs',
              'engineered_features':FEATURES,'raw_channels':['Q_Ah','dQdV_Ah_per_V','I_A'],
              'voltage_grid_v':GRID.tolist(),'mask_policy':'Full measured coverage only; unavailable cycles reject windows',
              'window_counts':{str(n):len(np.load(OUT/'processed'/f'indices_L{n}.npy')) for n in [5,10,20]}}
    json_write(OUT/'processed/manifest.json',metadata)
    shutil.copy2(PARENT/'configs/feature_availability.json',OUT/'configs/feature_availability.json')
    shutil.copy2(ROOT/'outputs/feature_definitions.csv',OUT/'features/feature_definitions.csv')
    versions={}
    for p in ['torch','numpy','pandas','scipy','scikit-learn','optuna','python-calamine','matplotlib','seaborn','tqdm','openpyxl','pytest','joblib','sqlalchemy','alembic']:
        try:versions[p]=md.version(p)
        except md.PackageNotFoundError:versions[p]=None
    hardware={'cuda_available':torch.cuda.is_available(),'torch_cuda':torch.version.cuda,
              'nvidia_smi':subprocess.check_output(['nvidia-smi'],text=True)}
    if torch.cuda.is_available():
        prop=torch.cuda.get_device_properties(0)
        hardware.update(gpu=prop.name,vram_bytes=prop.total_memory,bf16_supported=torch.cuda.is_bf16_supported())
    json_write(OUT/'configs/environment.json',{'python':sys.version,'executable':sys.executable,'packages':versions,'hardware':hardware,
               'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()})
    (OUT/'configs/requirements.txt').write_text('\n'.join(f'{k}=={v}' for k,v in versions.items() if v)+'\n')
    stats=rest_audit(f)
    print(json.dumps({'data':metadata,'rest':stats},indent=2),flush=True)


if __name__=='__main__':main()
