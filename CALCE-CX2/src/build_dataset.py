"""CLI orchestration. Example: python -m src.build_dataset --batteries CX2_16 --output outputs/pilot"""
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import platform
import importlib.metadata
import numpy as np
import pandas as pd
from tqdm import tqdm
from .common import ROOT, CONFIG, save_json
from .inspect_dataset import inspect
from .reconstruct_cycles import prepare_battery,reconstruct
from .extract_features import extract,add_history,definitions_frame,FEATURES
from .validate_features import validate
from .feature_analysis import analyze,plot_evolution,plot_quality,split_name


def curve_plots(examples,output,battery):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for key,xlabel,ylabel in [('vq','CC charge capacity (Ah)','Voltage (V)'),('ic','Voltage (V)','dQ/dV (Ah/V)'),('dva','Partial CC charge capacity (Ah)','dV/dQ (V/Ah)')]:
        fig,ax=plt.subplots(figsize=(8,5))
        for number,curves in examples:
            if key in curves:ax.plot(*curves[key],label=f'cycle {number}')
        if ax.lines:ax.legend(fontsize=7)
        ax.set(xlabel=xlabel,ylabel=ylabel,title=battery);fig.tight_layout();fig.savefig(output/'figures'/f'{battery}_{key}.png',dpi=140);plt.close(fig)


def build(raw,output,batteries=None):
    np.random.seed(42);output=Path(output);output.mkdir(parents=True,exist_ok=True);(output/'figures').mkdir(exist_ok=True)
    manifest=inspect(raw,output,batteries)
    # Freeze split manifest before statistics or feature selection.
    from .feature_analysis import SPLITS
    save_json(output/'battery_splits.json',dict(seed=42,splits=SPLITS,policy='Frozen a priori by identifiers'))
    definitions_frame().to_csv(output/'feature_definitions.csv',index=False)
    collected=[];audits=[]
    # Fingerprint code, raw file contents and configuration. Cache never depends on modification time.
    code=hashlib.sha256(b''.join(p.read_bytes() for p in sorted((ROOT/'src').glob('*.py')))+json.dumps(CONFIG,sort_keys=True).encode()).hexdigest()
    for battery in tqdm(sorted(manifest.battery.unique()),desc='Reconstruct batteries'):
        bm=manifest[manifest.battery==battery]
        if not (bm.status=='ready').any():continue
        fingerprint=hashlib.sha256((code+bm.to_csv(index=False)).encode()).hexdigest()
        cache=output/'cache'/f'{battery}_{fingerprint}.pkl'
        if cache.exists():
            with open(cache,'rb') as f:rows,audit,examples=pickle.load(f)
        else:
            d,audit=prepare_battery(manifest,battery);cycles,quality=reconstruct(d,battery)
            rows=[];examples=[]
            # Representative curves across lifetime, independent of labels.
            chosen=set(np.linspace(0,max(0,len(cycles)-1),6,dtype=int))
            for j,(r,parts) in enumerate(tqdm(cycles,desc=battery,leave=False)):
                f,metadata,curves=extract(parts);rows.append(dict(**r,**f,**metadata))
                if j in chosen:examples.append((r['lifetime_cycle'],curves))
            del d,cycles
            with open(cache,'wb') as f:pickle.dump((rows,audit,examples),f,protocol=5)
        collected.extend(rows)
        if len(audit):audits.append(audit)
        curve_plots(examples,output,battery)
        pd.DataFrame(collected).to_csv(output/'features_checkpoint.csv',index=False)
    if not collected:raise RuntimeError('No reconstructable timestamped Channel sheets. See inspection_manifest.csv.')
    pd.concat(audits,ignore_index=True).to_csv(output/'session_events.csv',index=False) if audits else pd.DataFrame(columns=['battery','source','event','count']).to_csv(output/'session_events.csv',index=False)
    return finalize(pd.DataFrame(collected),output,manifest,code)


def finalize(frame,output,manifest,code,resumed=False):
    frame=add_history(frame)
    for feature in FEATURES:
        if feature not in frame:frame[feature]=np.nan
    frame['split']=frame.battery.map(split_name)
    frame['eligible_analysis']=frame.validated_complete&(frame.protocol=='conventional')
    frame=validate(frame,output)
    frame.to_csv(output/'cx2_features_all.csv',index=False)
    # Quality table carries provenance and labels, but no charging feature columns.
    qc=[c for c in frame if c not in FEATURES and c not in ['feature_quality_flags'] and not c.startswith('ic_peak_height_sg')]
    frame[qc].to_csv(output/'cycle_quality.csv',index=False)
    frame.loc[~frame.validated_complete,qc].to_csv(output/'rejected_cycles.csv',index=False)
    rank,selected=analyze(frame,output);plot_evolution(frame,output);plot_quality(frame,output)
    summary=frame.groupby('battery').agg(total=('cycle_id','size'),validated=('validated_complete','sum'),conventional=('eligible_analysis','sum'))
    save_json(output/'run_metadata.json',dict(config=CONFIG,python=platform.python_version(),packages={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','matplotlib','openpyxl','tqdm']},code_sha256=code,
        raw_files=int(manifest.source_file.nunique()),batteries_inventoried=int(manifest.battery.nunique()),batteries_reconstructed=int(frame.battery.nunique()),
        candidates=len(frame),validated_complete=int(frame.validated_complete.sum()),conventional_validated=int(frame.eligible_analysis.sum()),
        f4=selected,no_models_trained=True,resumed_from_checkpoint=resumed,checkpoint_sha256=__import__('src.common',fromlist=['digest']).digest(output/'features_checkpoint.csv')))
    text=['# Feature pipeline execution report','',summary.to_markdown() if __import__('importlib').util.find_spec('tabulate') else summary.to_string(),'',
          'Rejected/incomplete candidates: '+str(int((~frame.validated_complete).sum())),
          'Provisional compact features: '+(', '.join(selected) or 'Insufficient reliable training data'),'',
          'F4 uses training statistics only. Validation-performance confirmation is deferred; no predictive model has been fitted.',
          'Same-cycle measured discharge C-rate is audit metadata, not a charging-time predictor.',
          'Text sessions without absolute timestamps/verified capacity units are inventoried but not reconstructed.',
          'Lifetime cycle numbers are observed reconstruction ordinals, not a claim of continuous lifetime coverage.']
    def table(data):
        def value(x):
            if isinstance(x,float):return f'{x:.5g}' if np.isfinite(x) else 'NA'
            return str(x)
        return '\n'.join(['| '+' | '.join(data.columns)+' |','| '+' | '.join(['---']*len(data.columns))+' |']+['| '+' | '.join(value(v) for v in row)+' |' for row in data.itertuples(index=False,name=None)])
    # Complete, readable execution report; detailed tables remain available as CSV.
    text[2]=table(summary.reset_index())
    text.extend(['','## Inventory','',table(manifest.status.value_counts().rename_axis('status').reset_index(name='sheets_or_files')),
                 '','## Training-only associations','',table(rank[['feature','n','pearson','spearman','mutual_information','battery_mean_spearman','sign_consistency']]),
                 '','## Feature missingness among validated complete cycles','',table(pd.DataFrame({'feature':FEATURES,'missing_percent':[100*frame.loc[frame.validated_complete,f].isna().mean() for f in FEATURES]})),
                 '','Unresolved current-text sessions are not counted as zero-cycle batteries. Separate temperature logs cannot yet be synchronized to these current exports.',
                 '','Unit/regression tests: run `python -B -m unittest discover -s tests -v`. See `tests/test_pipeline.py` for numerical, reconstruction, causal-history, protocol-family, and held-out-selection regression checks.'])
    (output/'REPORT.md').write_text('\n'.join(text),encoding='utf-8')
    print(summary.to_string());print('Provisional F4:',selected);print('Outputs:',output.resolve())
    return frame


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',type=Path,default=ROOT/'CX2_data');p.add_argument('--output',type=Path,default=ROOT/'outputs');p.add_argument('--batteries',nargs='+');p.add_argument('--reports-only',action='store_true',help='Resume validation, statistics and plots from features_checkpoint.csv; does not reconstruct raw data')
    a=p.parse_args()
    if a.reports_only:
        if a.batteries:p.error('--reports-only cannot be combined with --batteries')
        manifest=pd.read_csv(a.output/'inspection_manifest.csv');frame=pd.read_csv(a.output/'features_checkpoint.csv',low_memory=False)
        code=hashlib.sha256(b''.join(p.read_bytes() for p in sorted((ROOT/'src').glob('*.py')))+json.dumps(CONFIG,sort_keys=True).encode()).hexdigest()
        finalize(frame,a.output,manifest,code,resumed=True)
    else:build(a.raw,a.output,a.batteries)
if __name__=='__main__':main()
