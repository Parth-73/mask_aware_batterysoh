"""Reuse audited reconstruction exports and conservative original-ordinal windows."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from ..training.config import FEATURES as OLD_FEATURES, ROOT, SPLITS as HISTORICAL, json_write
from ..training.dataset import audit_source, cycle_reasons, WindowSet

BATTERIES = ['CX2_16', 'CX2_33', 'CX2_34', 'CX2_35', 'CX2_36', 'CX2_37', 'CX2_38']
FEATURES = {'five': OLD_FEATURES['F3'], 'three': OLD_FEATURES['F1'], 'nine': OLD_FEATURES['F2']}
UNITS = {'cc_duration_s': 's', 'cv_duration_s': 's', 'dva_peak': 'V/Ah',
         'ic_peak_voltage': 'V', 'ic_peak_width': 'V', 'ic_peak_height': 'Ah/V',
         'partial_charge_ah': 'Ah', 'partial_dv_dq': 'V/Ah', 'ic_peak_area': 'Ah'}


def partitions(name):
    if name == 'historical':
        return {k: list(v) for k, v in HISTORICAL.items()}
    i = BATTERIES.index(name)
    validation = BATTERIES[(i + 1) % len(BATTERIES)]
    return {'train': [b for b in BATTERIES if b not in (name, validation)],
            'validation': [validation], 'test': [name]}


def load_frame(output):
    output = Path(output)
    f = audit_source(ROOT/'outputs/cx2_features_all.csv', ROOT/'outputs/cycle_quality.csv',
                     ROOT/'outputs/rejected_cycles.csv')
    f['cycle_exclusion'] = f.apply(lambda r: cycle_reasons(r, FEATURES['nine']), axis=1)
    f['rest_duration_s'] = np.nan
    f['rest_verified'] = False
    f['rest_status'] = 'unavailable_not_reconstructed_from_gaps'
    output.mkdir(parents=True, exist_ok=True)
    f.to_csv(output/'cycle_audit.csv', index=False)
    counts = f.groupby('battery').agg(candidates=('cycle_id', 'size'), validated=('validated_complete', 'sum'))
    counts['complete_case_eligible'] = f[f.cycle_exclusion.eq('')].groupby('battery').size()
    counts.to_csv(output/'cycle_counts.csv')
    json_write(output/'rest_assessment.json', {'verified': False, 'duration_available': False,
        'reason': 'Aggregate exports lack preceding observed rest event boundaries and state coverage. Recording gaps and relaxation_60s_v do not establish preceding rest duration.',
        'missing_policy': 'NaN; never zero imputed', 'optional_six_feature_experiment': 'not supported'})
    definitions = pd.read_csv(ROOT/'outputs/feature_definitions.csv').set_index('feature')
    for feature, unit in UNITS.items():
        if definitions.loc[feature, 'units'] != unit:
            raise ValueError(f'Feature unit mismatch: {feature}')
    return f


def audited_windows(frame, length):
    """Vectorized equivalent of training.dataset.build_windows, with identical gates.

    The reference iterates pandas frames per target and costs minutes per length.
    Differential tests compare arrays, IDs, and every rejection reason exactly.
    """
    all_x, all_y, metadata, audits = [], [], [], []
    order = FEATURES['nine']
    for battery, original in frame.groupby('battery', sort=True):
        g = original.sort_values('lifetime_cycle').reset_index(drop=True)
        n = len(g)
        def any_window(values, width):
            result = np.zeros(n, bool)
            if n >= width:
                cumulative = np.r_[0, np.cumsum(np.asarray(values, int))]
                result[width-1:] = cumulative[width:]-cumulative[:-width] > 0
            return result
        sufficient = np.arange(n) >= length-1
        gap = (g.start-g.end.shift()).dt.total_seconds().to_numpy()
        ordinal = g.lifetime_cycle.to_numpy(int)
        source = g.sources.astype(str).to_numpy()
        checks = {
            'rejected_or_missing_cycle_in_window': any_window(g.cycle_exclusion.str.len().gt(0), length),
            'ordinal_gap': any_window(np.r_[False, np.diff(ordinal)!=1], length-1),
            'acquisition_session_boundary': (any_window(np.r_[False, source[1:]!=source[:-1]], length-1) |
                                             any_window(g.sources.str.contains(';', regex=False), length)),
            'acquisition_gap_or_overlap': any_window(~((gap>=0)&(gap<=3600)), length-1),
            'counter_reset_in_window': any_window(g.local_counter_resets>0, length)}
        reasons = [[] if ok else ['insufficient_history'] for ok in sufficient]
        for reason, mask in checks.items():
            for j in np.flatnonzero(mask & sufficient):
                reasons[j].append(reason)
        accepted = np.array([not r for r in reasons])
        audit = pd.DataFrame({'battery': battery, 'split': g.benchmark_split, 'cycle_id': g.cycle_id,
                              'lifetime_cycle': ordinal, 'sequence_length': length, 'accepted': accepted,
                              'rejection_reasons': [';'.join(r) for r in reasons]})
        audits.append(audit)
        indices = np.flatnonzero(accepted)
        if not len(indices):
            continue
        arr = g[order].to_numpy(np.float32)
        sliced = np.lib.stride_tricks.sliding_window_view(arr, length, axis=0).transpose(0, 2, 1)
        all_x.append(sliced[indices-length+1].copy())
        all_y.append(g.soh.to_numpy(np.float32)[indices])
        ids = g.cycle_id.tolist()
        starts, ends = g.start.tolist(), g.end.tolist()
        for j in indices:
            metadata.append(dict(battery=battery, split=g.benchmark_split.iloc[j], cycle_id=ids[j],
                lifetime_cycle=int(ordinal[j]), start=str(starts[j]), end=str(ends[j]), source=source[j],
                input_cycle_ids=json.dumps(ids[j-length+1:j+1]),
                input_ordinals=json.dumps(ordinal[j-length+1:j+1].tolist()),
                input_start=str(starts[j-length+1]), current_charge_start=str(starts[j])))
    x = np.concatenate(all_x) if all_x else np.empty((0, length, len(order)), np.float32)
    y = np.concatenate(all_y) if all_y else np.empty(0, np.float32)
    return WindowSet(x, y, pd.DataFrame(metadata)), pd.concat(audits, ignore_index=True)


def windows(frame, length, matched=False):
    # Unscaled arrays; fit scaling only after assigning inner partitions.
    order = FEATURES['nine']
    w, audit = audited_windows(frame, length)
    if matched:
        longest, _ = audited_windows(frame, 20)
        mask = w.metadata.cycle_id.isin(longest.metadata.cycle_id).to_numpy()
        w.x, w.y, w.metadata = w.x[mask], w.y[mask], w.metadata.loc[mask].reset_index(drop=True)
    if not len(w.y):
        raise ValueError('No eligible windows')
    return w, audit


def fold_data(frame, whole, fold, feature_set):
    split = partitions(fold)
    features = FEATURES[feature_set]
    fit = frame[frame.battery.isin(split['train']) & frame.cycle_exclusion.eq('')]
    scaler = StandardScaler().fit(fit[features])
    positions = [FEATURES['nine'].index(f) for f in features]
    result = {}
    for role, batteries in split.items():
        mask = whole.metadata.battery.isin(batteries).to_numpy()
        meta = whole.metadata.loc[mask].reset_index(drop=True).copy()
        x = whole.x[mask][:, :, positions]
        x = ((x - scaler.mean_) / scaler.scale_).astype(np.float32)
        if not len(x):
            raise ValueError(f'Empty {fold}/{role}')
        # Pair only directly adjacent target windows in one verified source.
        previous = np.arange(len(x))
        valid = np.zeros(len(x), bool)
        battery_values = meta.battery.to_numpy()
        source_values = meta.source.to_numpy()
        ordinal_values = meta.lifetime_cycle.to_numpy()
        valid[1:] = ((battery_values[1:]==battery_values[:-1]) &
                     (source_values[1:]==source_values[:-1]) & (np.diff(ordinal_values)==1))
        previous[valid] -= 1
        result[role] = (x, whole.y[mask], meta, previous, valid)
    info = {'feature_order': features, 'fit_batteries': split['train'], 'fit_cycles': fit.cycle_id.tolist(),
            'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(), 'partitions': split,
            'target_scaling': 'none', 'cohort': 'common complete-case nine-feature union'}
    return result, scaler, info
