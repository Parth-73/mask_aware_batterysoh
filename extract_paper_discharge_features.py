"""Extract explicit approximations to Ma et al. H1-H3 from local raw caches.

Run with the base Anaconda Python (pandas 3 + pyarrow available there).
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'cnn_lstm_paper_informed_results'
OUT.mkdir(exist_ok=True)
data = pd.read_csv(ROOT / 'pre processed ready to use data/all_data.csv')
cycles = pd.read_csv(ROOT / 'benchmark-repository/CALCE-CX2/outputs/cx2_features_all.csv', low_memory=False)
cycles = cycles[cycles.cycle_id.isin(data.cycle_id)].copy()
manifest = pd.read_csv(ROOT / 'benchmark-repository/CALCE-CX2/outputs/rapct_prony/data_audit/inspection_manifest.csv')
records = []
for source, part in cycles.groupby('sources'):
    keys = manifest.battery.astype(str) + '/' + manifest.source_file.map(lambda x: str(x).replace('\\', '/').split('/')[-1]) + '::' + manifest.sheet.astype(str)
    matches = manifest[keys.eq(source)]
    if len(matches) != 1:
        continue  # Multi-source cycles cannot be safely resolved by this extractor.
    cache = ROOT / 'benchmark-repository/CALCE-CX2/outputs/rapct_prony/data_audit/cache/channels' / Path(matches.iloc[0]['cache']).name
    raw = pd.read_pickle(cache).sort_values('timestamp')
    raw = raw[raw.current < -0.01].copy()
    stamps = pd.to_datetime(raw.timestamp).to_numpy()
    for r in part.itertuples():
        a = np.searchsorted(stamps, np.datetime64(r.start), side='left')
        b = np.searchsorted(stamps, np.datetime64(r.end), side='right')
        d = raw.iloc[a:b]
        if len(d) < 5 or d.step.nunique() != 1:
            continue
        t = d.test_s.to_numpy(dtype=float)
        elapsed = t - t[0] + float(d.step_s.iloc[0])
        voltage = d.voltage.to_numpy(dtype=float)
        if not np.isfinite(elapsed).all() or np.any(np.diff(elapsed) <= 0):
            continue
        records.append(dict(cycle_id=r.cycle_id,
                            h1_min_voltage_time_s=float(elapsed[np.argmin(voltage)]),
                            h2_mean_discharge_time_s=float(elapsed.mean()),
                            h3_mean_discharge_voltage_v=float(voltage.mean())))
    print(source, 'features extracted:', len(records), flush=True)
features = pd.DataFrame(records)
assert not features.cycle_id.duplicated().any()
merged = data.merge(features, on='cycle_id', how='left', validate='one_to_one')
merged.to_csv(OUT / 'discharge_features.csv', index=False)
print(merged.groupby('split')[features.columns[1:].tolist()].count())
