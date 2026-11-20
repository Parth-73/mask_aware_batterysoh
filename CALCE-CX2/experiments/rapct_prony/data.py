"""Charging-only inputs and conservative measured-rest inventory."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.extract_features import charge_curve, ic_curves

FEATURES = ['cc_duration_s', 'cv_duration_s', 'dva_peak', 'ic_peak_voltage', 'ic_peak_width']
GRID = np.round(np.arange(3.8, 4.15001, .002), 6)
BATTERIES = ['CX2_16', 'CX2_33', 'CX2_34', 'CX2_35', 'CX2_36', 'CX2_37', 'CX2_38']


def curve_tensor(parts):
    cc, _, _, _, _, bad = charge_curve(parts)
    if bad or len(cc) < 5:
        raise ValueError('unreliable_charge_curve')
    if cc[0, 0] > GRID[0] or cc[-1, 0] < GRID[-1]:
        raise ValueError('incomplete_measured_voltage_coverage')
    result = ic_curves(cc)
    if result is None:
        raise ValueError('ic_curve_unavailable')
    vg, q, variants = result
    if len(vg) != len(GRID) or not np.allclose(vg, GRID):
        raise ValueError('incompatible_voltage_grid')
    curve = variants[11]
    spread = np.ptp([v.max() for v in variants.values()]) / max(np.median([v.max() for v in variants.values()]), 1e-9)
    if spread > .2 or np.mean(curve < 0) > .01:
        raise ValueError('unstable_ic')
    x = np.stack([q, curve, np.interp(GRID, cc[:, 0], cc[:, 3])]).astype('float32')
    if not np.isfinite(x).all():
        raise ValueError('nonfinite_curve')
    return x


def fit_scalers(features, curves, batteries, training):
    mask = np.isin(batteries, training)
    if not mask.any() or set(training) - set(np.asarray(batteries)[mask]):
        raise ValueError('missing_training_battery')
    fs = StandardScaler().fit(features[mask])
    cs = StandardScaler().fit(curves[mask].transpose(0, 2, 1).reshape(-1, curves.shape[1]))
    return fs, cs


def rest_inventory(d, cycles):
    """No cross-source rest inference; gaps >180s remain uncertain, even at zero endpoints."""
    if d.empty:return []
    rows = []
    endpoints=[]
    for r,parts in cycles:
        charge=[g.timestamp.max() for g in parts if g.phase.iloc[0]=='charge']
        discharge=[g.timestamp.max() for g in parts if g.phase.iloc[0]=='discharge']
        endpoints.append((r, max(charge) if charge else pd.NaT, max(discharge) if discharge else pd.NaT))
    charges=sorted([(t,r) for r,t,_ in endpoints if pd.notna(t)],key=lambda x:x[0])
    discharges=sorted([(t,r) for r,_,t in endpoints if pd.notna(t)],key=lambda x:x[0])
    ct=np.array([t.value for t,_ in charges],dtype=np.int64)
    dtimes=np.array([t.value for t,_ in discharges],dtype=np.int64)
    for (source, epoch), group in d.groupby(['source', 'epoch'], sort=False):
        g = group.sort_values('source_order')
        dt = g.timestamp.diff().dt.total_seconds()
        test_dt = g.test_s.diff()
        compatible = dt.gt(0) & dt.le(180) & test_dt.gt(0) & (test_dt-dt).abs().le(2)
        zero = g.current.abs().le(.005)
        verified = compatible & zero & zero.shift(fill_value=False)
        uncertain = dt.gt(180) | (dt.gt(0) & ~compatible)
        category = np.where(verified, 'verified_measured_rest', np.where(uncertain, 'uncertain_recording_gap', 'no_verified_rest'))
        block = pd.Series(category, index=g.index).ne(pd.Series(category, index=g.index).shift()).cumsum()
        for _, h in g.groupby(block):
            cat = category[g.index.get_indexer(h.index)][0]
            if cat == 'no_verified_rest':
                continue
            first_pos = g.index.get_loc(h.index[0])
            if first_pos == 0:
                continue
            start = g.timestamp.iloc[first_pos-1]
            end = h.timestamp.iloc[-1]
            previous_index=np.searchsorted(dtimes,start.value,side='right')-1
            next_index=np.searchsorted(ct,end.value,side='left')
            before=discharges[previous_index][1] if previous_index>=0 else None
            after=charges[next_index][1] if next_index<len(charges) else None
            current = g.iloc[first_pos-1:first_pos+len(h)].current
            pair = bool(before and after and before['cycle_id'] != after['cycle_id'] and before['validated_complete'] and after['validated_complete'] and after['lifetime_cycle']==before['lifetime_cycle']+1)
            consistent = bool(pair and before['protocol'] == after['protocol'] == 'conventional' and abs(before.get('discharge_crate_median', 99)-after.get('discharge_crate_median', -99)) < .02)
            rows.append(dict(battery=g.battery.iloc[0], preceding_cycle_id=before['cycle_id'] if before else '',
                following_cycle_id=after['cycle_id'] if after else '', event_start=start, event_end=end,
                duration_s=(end-start).total_seconds(), current_abs_max_a=current.abs().max(),
                category=cat, source_identifiers=source, verification_status=cat == 'verified_measured_rest',
                uncertainty_reason='' if cat == 'verified_measured_rest' else 'unobserved_state_or_inconsistent_clock',
                validated_pair=pair, consistent_discharge_conditions=consistent,
                available_by_following_charge=bool(after and end <= charges[next_index][0])))
    return rows

def chronological_windows(frame, length=5, max_gap_s=3600.):
    """Indexes into original retained rows; ordinal, source and acquisition gaps are gates."""
    windows=[];audit=[]
    for battery,group in frame.groupby('battery',sort=True):
        group=group.sort_values('lifetime_cycle')
        indices=group.index.to_numpy()
        for j,row in enumerate(group.itertuples()):
            reason=''
            if j<length-1:reason='insufficient_history'
            else:
                w=group.iloc[j-length+1:j+1]
                if not np.all(np.diff(w.lifetime_cycle)==1):reason='missing_cycle_history'
                elif w.sources.nunique()!=1 or w.sources.str.contains(';',regex=False).any():reason='source_discontinuity'
                elif (w.local_counter_resets>0).any():reason='counter_reset'
                else:
                    gap=(pd.to_datetime(w.start.iloc[1:]).reset_index(drop=True)-pd.to_datetime(w.end.iloc[:-1]).reset_index(drop=True)).dt.total_seconds()
                    if not gap.between(0,max_gap_s).all():reason='acquisition_gap_or_overlap'
            audit.append(dict(battery=battery,cycle_id=row.cycle_id,length=length,accepted=not reason,reason=reason))
            if not reason:windows.append(indices[j-length+1:j+1])
    return np.asarray(windows,dtype=int).reshape(-1,length),pd.DataFrame(audit)

def chronological_segments(frame,max_gap_s=3600.):
    """Whole observed segments for cycle-count state recurrence, not elapsed-time rest inference."""
    segments=[]
    for _,g in frame.groupby('battery',sort=True):
        g=g.sort_values('lifetime_cycle')
        current=[];previous=None
        for index,row in g.iterrows():
            gap=(pd.Timestamp(row.start)-pd.Timestamp(previous.end)).total_seconds() if previous is not None else None
            connected=bool(previous is not None and row.lifetime_cycle==previous.lifetime_cycle+1 and row.sources==previous.sources and row.local_counter_resets==0 and 0<=gap<=max_gap_s)
            if current and not connected:segments.append(np.asarray(current,dtype=int));current=[]
            current.append(index);previous=row
        if current:segments.append(np.asarray(current,dtype=int))
    return segments
