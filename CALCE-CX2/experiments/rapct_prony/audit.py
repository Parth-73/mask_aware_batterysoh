"""Checkpoint 0 and isolated curve preparation. Run from CALCE-CX2."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
import torch
from src.common import ROOT, digest, save_json
from src.inspect_dataset import inspect
from src.reconstruct_cycles import prepare_battery, reconstruct
from .data import BATTERIES, FEATURES, GRID, curve_tensor, rest_inventory

OUT = ROOT/'outputs/rapct_prony'
DIRECTORIES = ['data_audit','rest_events','processed','features','splits','configs','optuna_studies','checkpoints','scalers','predictions','latent_states','metrics','plots','reports']


def audit(raw):
    for name in DIRECTORIES:
        (OUT/name).mkdir(parents=True, exist_ok=True)
    packages = {}
    for p in ['torch','numpy','pandas','scipy','scikit-learn','optuna','matplotlib','seaborn','tqdm','openpyxl','pytest','joblib']:
        try: packages[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError: packages[p] = None
    environment = dict(python=platform.python_version(), packages=packages, cuda_available=torch.cuda.is_available(), torch_cuda=torch.version.cuda)
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        environment.update(gpu=p.name, vram_bytes=p.total_memory, capability=list(torch.cuda.get_device_capability()), bf16=torch.cuda.is_bf16_supported())
    save_json(OUT/'configs/environment.json', environment)
    paths = sorted(p for p in raw.rglob('*') if p.is_file())
    pd.DataFrame([dict(path=str(p.relative_to(raw)),bytes=p.stat().st_size,sha256=digest(p)) for p in paths]).to_csv(OUT/'data_audit/raw_hashes.csv',index=False)
    protected = [ROOT/'README.md', *sorted((ROOT/'src').rglob('*.py')),
        *[ROOT/'outputs'/p for p in ['feature_definitions.csv','feature_sets.json','battery_splits.json','cx2_features_all.csv','cycle_quality.csv']],
        *sorted((ROOT/'outputs/lstm_baselines').glob('*.json'))]
    save_json(OUT/'data_audit/historical_snapshot.json', {str(p.relative_to(ROOT)):digest(p) for p in protected})
    contract = dict(cutoff='end_of_current_available_charge', target='validated_discharge_capacity_ah/1.35', target_fraction=True,
        target_smoothing=False, allowed_features=FEATURES, raw_channels={'Q':'Ah','dQdV':'Ah/V','I':'A'}, voltage_grid=GRID.tolist(),
        forbidden=['same_cycle_discharge','future_soh','post_cutoff_rest','unverified_temperature','unverified_resistance'],
        rest_current_threshold_a=.005, rest_max_observation_interval_s=180, seeds=[42,123,2026])
    save_json(OUT/'configs/feature_availability.json',contract)
    f = pd.read_csv(ROOT/'outputs/cx2_features_all.csv',low_memory=False)
    f['five_feature_complete']=np.isfinite(f[FEATURES]).all(axis=1)
    f.groupby(['battery','protocol']).agg(candidates=('cycle_id','size'),validated=('soh','count'),five_complete=('five_feature_complete','sum')).to_csv(OUT/'data_audit/cohort_summary.csv')
    historical=[]
    for base in [ROOT/'outputs/lstm_baselines',ROOT/'outputs/soh_four_model_benchmarks']:
        for p in sorted(base.rglob('*')):
            if p.is_file() and p.suffix in {'.json','.csv','.md'} and ('metric' in str(p).lower() or p.name in {'experiment_config.json','experiment_manifest.json','BASELINE_REPORT.md'}):
                historical.append(dict(path=str(p.relative_to(ROOT)),sha256=digest(p),comparison_status='historical_separate_cohort_protocol_requires_matching'))
    pd.DataFrame(historical).to_csv(OUT/'data_audit/historical_results_inventory.csv',index=False)
    (OUT/'reports/REPOSITORY_AUDIT.md').write_text('''# Repository audit
The unchanged reconstruction integrates within physical steps, validates counter agreement, rejects ambiguous boundaries and exports all rejection reasons. SOH is raw validated discharge Ah / 1.35. IC uses the measured 3.80-4.15 V domain, 2 mV grid and SG(11,3), with smoothing checks at 7/11/15. DVA follows the existing 3.8-4.0 V partial-capacity rules. The five engineered features are independently MLP encoded.

Original preprocessing holds out CX2_37 for validation; historical LSTM training includes CX2_37 and validates on CX2_36. CX2_38 is historical test. Existing four-model CNN convolves across successive cycles with five features as channels; it is not a raw voltage-curve CNN. Historical metrics cannot be pooled with the new matched raw-coverage cohort. All historical source/config hashes and artifact locations are preserved in data_audit. Existing untracked benchmark files predate this task and are left unchanged.

Raw channel caches reference D:/battery_soh on another machine; rebuild under the isolated output. Acquisition timestamps establish chronology; filenames never do. Temperature logger/current session synchronization remains unverified. Rest after end-of-charge, including the exported relaxation_60s_v feature, is excluded from this prediction contract.
''',encoding='utf-8')
    print(json.dumps(environment),flush=True)


def prepare(raw, selected):
    # Faster Excel parser; all reconstructed labels are independently checked below.
    original_excel_file, original_read_excel = pd.ExcelFile, pd.read_excel
    def excel_file(*args, **kwargs):
        kwargs.setdefault('engine','calamine')
        return original_excel_file(*args, **kwargs)
    def read_excel(io, *args, **kwargs):
        if isinstance(io, (str, Path)):kwargs.setdefault('engine','calamine')
        return original_read_excel(io, *args, **kwargs)
    pd.ExcelFile, pd.read_excel = excel_file, read_excel
    try:manifest = inspect(raw,OUT/'data_audit',selected)
    finally:pd.ExcelFile, pd.read_excel = original_excel_file, original_read_excel
    prior = pd.read_csv(ROOT/'outputs/cx2_features_all.csv',low_memory=False).set_index('cycle_id')
    records=[];arrays=[];quality=[];events=[];session_audit=[]
    for battery in selected:
        d,a = prepare_battery(manifest,battery)
        cycles,q = reconstruct(d,battery)
        events.extend(rest_inventory(d,cycles))
        if not a.empty:session_audit.extend(a.to_dict('records'))
        for r,parts in cycles:
            reason='';x=None
            if r['cycle_id'] not in prior.index:
                reason='no_historical_cycle_match'
            else:
                old=prior.loc[r['cycle_id']]
                if str(pd.Timestamp(old.start)) != str(pd.Timestamp(r['start'])) or bool(old.validated_complete) != bool(r['validated_complete']):
                    reason='reconstruction_mismatch'
                elif r['validated_complete'] and not np.isclose(old.soh,r['soh'],atol=1e-9,rtol=0):
                    reason='target_mismatch'
                elif not r['validated_complete']:reason='quality_rejected:'+r['rejection_reasons']
                elif r['protocol']!='conventional':reason='nonconventional'
                elif not np.isfinite(old[FEATURES].to_numpy(float)).all():reason='missing_engineered_feature'
                else:
                    try:x=curve_tensor(parts)
                    except ValueError as e:reason=str(e)
            quality.append(dict(battery=battery,cycle_id=r['cycle_id'],accepted=not reason,reason=reason))
            if reason:continue
            charge=pd.concat([g for g in parts if g.phase.iloc[0]=='charge'])
            cutoff=charge.timestamp.max()
            if cutoff>=pd.concat([g for g in parts if g.phase.iloc[0]=='discharge']).timestamp.min():
                raise ValueError('prediction_cutoff_overlaps_discharge')
            record={k:old[k] for k in FEATURES}
            record.update(battery=battery,cycle_id=r['cycle_id'],lifetime_cycle=r['lifetime_cycle'],start=str(r['start']),end=str(r['end']),
                prediction_cutoff=str(cutoff),soh=r['soh'],sources=r['sources'],local_counter_resets=r['local_counter_resets'])
            records.append(record);arrays.append(x)
        print(f'{battery}: reconstructed {len(cycles)}, cumulative curve samples {len(records)}, events {len(events)}',flush=True)
    pd.DataFrame(records).to_csv(OUT/'processed/samples.csv',index=False)
    np.save(OUT/'processed/curves.npy',np.stack(arrays) if arrays else np.empty((0,3,len(GRID)),dtype='float32'))
    pd.DataFrame(quality).to_csv(OUT/'data_audit/curve_quality.csv',index=False)
    pd.DataFrame(session_audit).to_csv(OUT/'data_audit/session_audit.csv',index=False)
    cols=['battery','preceding_cycle_id','following_cycle_id','event_start','event_end','duration_s','current_abs_max_a','category','source_identifiers','verification_status','uncertainty_reason','validated_pair','consistent_discharge_conditions','available_by_following_charge']
    e=pd.DataFrame(events,columns=cols)
    e.to_csv(OUT/'rest_events/rest_event_inventory.csv',index=False)
    verified=e[e.category.eq('verified_measured_rest')]
    usable=verified[verified.validated_pair & verified.consistent_discharge_conditions & verified.available_by_following_charge]
    stats=dict(verified_events=len(verified),uncertain_gaps=int(e.category.eq('uncertain_recording_gap').sum()),
        per_battery=verified.groupby('battery').size().to_dict(), duration_quantiles_s=verified.duration_s.quantile([0,.25,.5,.75,1]).to_dict(),
        validated_pairs=int(verified.validated_pair.sum()),consistent_conditions=int(verified.consistent_discharge_conditions.sum()),
        usable_pair_events=len(usable), usable_batteries=sorted(usable.battery.unique()),
        recovery_training_status='NOT_AUTHORIZED_BY_FEASIBILITY_YET: event mapping, independent batteries, initial states and sufficient duration diversity require review')
    save_json(OUT/'rest_events/feasibility.json',stats)
    (OUT/'reports/REST_DATA_FEASIBILITY.md').write_text('# Measured-rest feasibility\n\n```json\n'+json.dumps(stats,indent=2)+'\n```\n\nEvents are runs of measured |I| <= 0.005 A with clock-consistent adjacent samples <=180 s apart, confined to one source and recorder epoch. This is operational sampling evidence, not proof of continuous open circuit between samples. Longer gaps remain uncertain. Capacity increases never establish events. Events wholly inside a cycle are inventoried but the conservative before/after mapping does not establish recovery eligibility. No recovery training claim is supported by this inventory alone.\n',encoding='utf-8')
    print(json.dumps(stats),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['audit','prepare']);p.add_argument('--raw',type=Path,default=ROOT.parents[1]/'data');p.add_argument('--batteries',nargs='+',default=BATTERIES)
    args=p.parse_args()
    if args.command=='audit':audit(args.raw)
    else:prepare(args.raw,args.batteries)

if __name__=='__main__':main()
