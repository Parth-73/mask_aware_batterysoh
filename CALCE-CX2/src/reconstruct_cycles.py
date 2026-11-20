"""Measured-phase cycle reconstruction independent of repeated local cycle indices."""
import numpy as np
import pandas as pd
from .common import CONFIG, integrate_current


def classify_protocol(median_crate,peak_crate,discharge_segments):
    """Routine single continuous discharges at the observed 0.4C/0.5C/1C rates.

    Multiple segments and 3C/high-rate records remain separately audited, even when
    a particular multisegment sequence might later be verified as a routine step.
    """
    if not discharge_segments:return 'unknown'
    routine=(discharge_segments==1 and CONFIG['conventional_crate_min']<=median_crate<=CONFIG['conventional_crate_max']
             and peak_crate<=CONFIG['conventional_peak_crate_max'] and peak_crate-median_crate<.1)
    return 'conventional' if routine else 'unusual_discharge'


def phase_integral(g, sign):
    """Integrate within a single physical step, including its bounded initial interval.

    Step_Time describes time since the step switch. The unobserved initial interval
    is bounded by the preceding sample and at most max_sample_gap_s; use the first
    measured current as a rectangular approximation and expose the interval.
    """
    cv_event_logging=(sign==1 and g.voltage.min()>=4.18 and g.voltage.max()-g.voltage.min()<=.035 and g.current.iloc[-1]<=g.current.iloc[0])
    limit=CONFIG['max_cv_sample_gap_s'] if cv_event_logging else CONFIG['max_sample_gap_s']
    q,bad=integrate_current(g.test_s,g.current*sign,limit)
    initial=float(g.get('integration_initial_s',g.step_s).iloc[0])
    if not np.isfinite(initial) or initial<0 or initial>CONFIG['max_sample_gap_s']:
        initial=0.
        bad+=1
    q+=max(0,float(g.current.iloc[0])*sign)*initial/3600
    return q,bad,initial


def counter_delta(g,column):
    """Step-local counter endpoint minus preceding row (or file's known zero baseline).

    At source boundaries a first-row counter is usable only when the step begins
    near zero time and the counter is near zero. Never treat a lifetime counter as
    a cycle capacity. Resets within a step invalidate counter validation.
    """
    x=g[column].to_numpy(float)
    prior=g['prev_'+column].iloc[0]
    resets=int(np.sum(np.diff(x)<-1e-4))
    if not np.isfinite(prior):
        # A first file row can be mid-step: capacity baseline is then unknown.
        if g.step_s.iloc[0]<=1 and x[0]<=0.001:
            prior=0.
        else:
            return np.nan,resets,'unknown_initial_counter'
    if x[0]<prior-1e-4:
        # A reset at a step switch is safe only when it starts near zero.
        if x[0]<=0.001 and g.step_s.iloc[0]<=1:
            prior=0.
        else:
            return np.nan,resets+1,'counter_reset_unresolved'
    if resets:
        return np.nan,resets,'counter_reset_within_step'
    return float(x[-1]-prior),resets,''


def prepare_battery(manifest,battery):
    chunks=[]; audit=[]
    rows=manifest[(manifest.battery==battery)&(manifest.status=='ready')].sort_values(['start','source_file','sheet'])
    for r in rows.itertuples():
        d=pd.read_pickle(r.cache)
        d['source_order']=np.arange(len(d))
        for c in ['charge_counter','discharge_counter']:
            d['prev_'+c]=d[c].shift()
        d['raw_bad']=~np.isfinite(d[['test_s','step_s','current','voltage','charge_counter','discharge_counter']]).all(axis=1)
        d['source_reversal']=(d.timestamp.diff().dt.total_seconds()<0)|(d.test_s.diff()<0)
        d['local_reset']=d.local_cycle.diff()<0
        # Separate recorder epochs BEFORE chronological sorting; never integrate across them.
        d['epoch']=d.source_reversal.cumsum().astype(int)
        invalid=d.timestamp.isna()
        if invalid.any():
            audit.append(dict(battery=battery,source=d.source.iloc[0],event='missing_timestamp_rows',count=int(invalid.sum())))
        chunks.append(d[~invalid])
    if not chunks:return pd.DataFrame(),pd.DataFrame(audit)
    d=pd.concat(chunks,ignore_index=True)
    d=d.sort_values(['timestamp','source','source_order'],kind='stable').reset_index(drop=True)
    # Exact repeated exports are deduplicated and counted. Distinct rows sharing second-resolution
    # timestamps remain; test_s resolves subsecond order inside each source.
    key=['timestamp','test_s','step_s','step','current','voltage','charge_counter','discharge_counter']
    duplicate=d.duplicated(key)
    if duplicate.any():audit.append(dict(battery=battery,source='multiple',event='exact_duplicate_rows',count=int(duplicate.sum())))
    d=d[~duplicate].copy().reset_index(drop=True)
    d['phase']=np.select([d.current>CONFIG['active_current_a'],d.current < -CONFIG['active_current_a']],['charge','discharge'],default='rest')
    change=(d.source!=d.source.shift())|(d.epoch!=d.epoch.shift())|(d.step!=d.step.shift())|(d.step_s.diff()<-0.1)|(d.phase!=d.phase.shift())
    d['segment']=change.cumsum()
    d['gap_s']=d.timestamp.diff().dt.total_seconds().fillna(0)
    d['integration_initial_s']=d.step_s
    d['boundary_continuation']=False
    source_switch=d.source.ne(d.source.shift()) & (d.index>0)
    dt_test=d.test_s.diff()
    continuation=(source_switch & d.step.eq(d.step.shift()) & d.phase.eq(d.phase.shift()) &
                  d.step_s.ge(d.step_s.shift()) & dt_test.gt(0) & dt_test.le(CONFIG['max_sample_gap_s']) &
                  d.gap_s.ge(0) & d.gap_s.le(CONFIG['max_sample_gap_s']))
    d.loc[continuation,'integration_initial_s']=dt_test[continuation]
    d.loc[continuation,'boundary_continuation']=True
    for column in ['charge_counter','discharge_counter']:
        # Verify common counter and clock continuity, otherwise preserve unknown baseline.
        prior=d[column].shift()
        compatible=continuation & d[column].ge(prior-1e-4)
        d.loc[compatible,'prev_'+column]=prior[compatible]
    # Interleaved sources from nonidentical overlapping exports are ambiguous, not silently chosen.
    starts=rows.copy(); intervals=[]
    for r in starts.itertuples():
        start,end=pd.Timestamp(r.start),pd.Timestamp(r.end)
        if any(start<e and end>s for s,e in intervals):
            audit.append(dict(battery=battery,source=r.source_file,event='overlapping_session_interval',count=1))
        intervals.append((start,end))
    return d,pd.DataFrame(audit)


def reconstruct(d,battery):
    """A cycle is charge(s), optional rests, then all discharges up to the next charge.

    Long rests/recording gaps terminate candidates. Source changes split step integrals,
    but not candidates; a proven continuous boundary may continue the same cycle.
    Rest samples never contribute discharge capacity. Every orphan is retained.
    """
    if d.empty:return [],pd.DataFrame()
    segments=[g.copy() for _,g in d.groupby('segment',sort=False)]
    candidates=[]; current=[]; has_discharge=False
    for g in segments:
        phase=g.phase.iloc[0]
        gap=float(g.gap_s.iloc[0])
        new_charge=phase=='charge' and has_discharge
        long_gap=gap>CONFIG['max_rest_gap_s']
        previous_rest_long=bool(current and current[-1].phase.iloc[0]=='rest' and
                                (current[-1].timestamp.iloc[-1]-current[-1].timestamp.iloc[0]).total_seconds()>CONFIG['max_rest_gap_s'])
        if current and (new_charge or long_gap or previous_rest_long):
            candidates.append(current);current=[];has_discharge=False
        if phase=='rest' and not current:
            continue
        current.append(g)
        has_discharge=has_discharge or phase=='discharge'
    if current:candidates.append(current)
    quality=[];cycles=[]
    for number,parts in enumerate(candidates,1):
        allrows=pd.concat(parts,ignore_index=True)
        charge=[g for g in parts if g.phase.iloc[0]=='charge']
        discharge=[g for g in parts if g.phase.iloc[0]=='discharge']
        reasons=[];flags=[]
        r=dict(battery=battery,cycle_id=f'{battery}:{number:05d}',lifetime_cycle=number,
               start=allrows.timestamp.min(),end=allrows.timestamp.max(),
               sources=';'.join(sorted(allrows.source.unique())),
               local_cycles=';'.join(map(str,allrows.local_cycle.unique())),
               rows=len(allrows),charge_steps=len(charge),discharge_steps=len(discharge),
               file_boundary=allrows.source.nunique()>1,
               local_counter_resets=int(allrows.local_reset.sum()),
               source_timestamp_reversals=int(allrows.source_reversal.sum()))
        if not charge:reasons.append('missing_charge')
        if not discharge:reasons.append('missing_discharge')
        if allrows.raw_bad.any():reasons.append('nonfinite_raw_measurements')
        if r['source_timestamp_reversals']:reasons.append('source_clock_reversal')
        if r['file_boundary']:
            flags.append('cross_file_candidate')
            # Discontinuous file transitions cannot be integrated as a complete physical step.
            switches=allrows.source.ne(allrows.source.shift()) & (allrows.index>0)
            if (allrows.loc[switches,'gap_s']>CONFIG['max_sample_gap_s']).any():reasons.append('discontinuous_file_boundary')
            # Interleaving overlapping sources causes very short fragmentary segments.
            if int(switches.sum())>len(allrows.source.unique()):reasons.append('overlapping_sources')
        if r['local_counter_resets']:flags.append('local_cycle_counter_reset')
        for label,groups,sign,column in [('charge',charge,1,'charge_counter'),('discharge',discharge,-1,'discharge_counter')]:
            q=0.;counter=0.;bad=0;resets=0;initial=0.;unknown=False
            for g in groups:
                z,b,ini=phase_integral(g,sign);q+=z;bad+=b;initial+=ini
                if sign==1 and (g.test_s.diff()>CONFIG['max_sample_gap_s']).any() and not b:flags.append('sparse_cv_event_logging_counter_checked')
                c,rs,reason=counter_delta(g,column);resets+=rs
                if np.isfinite(c):counter+=c
                else:unknown=True;flags.append(label+'_'+reason)
                # A source beginning mid-step cannot prove initial coverage.
                if g.source_row.iloc[0]==2 and g.step_s.iloc[0]>CONFIG['max_sample_gap_s'] and not g.get('boundary_continuation',pd.Series([False])).iloc[0]:
                    reasons.append(label+'_starts_mid_step')
            r[label+'_integral_ah']=q if groups else np.nan
            r[label+'_counter_ah']=counter if groups and not unknown else np.nan
            r[label+'_invalid_intervals']=bad
            r[label+'_initial_approximation_s']=initial
            r[label+'_capacity_counter_resets']=resets
            disagreement=abs(counter-q) if groups and not unknown else np.nan
            r[label+'_capacity_disagreement_ah']=disagreement
            if bad:reasons.append(label+'_sampling_discontinuity')
            if groups and unknown:reasons.append(label+'_counter_unvalidated')
            if resets:flags.append(label+'_capacity_counter_reset')
            if np.isfinite(disagreement) and disagreement>max(CONFIG['capacity_agreement_absolute_ah'],CONFIG['capacity_agreement_relative']*max(q,counter)):
                reasons.append(label+'_counter_integral_disagreement')
        if charge:
            c=pd.concat(charge)
            r['charge_start_v']=float(c.voltage.iloc[0]);r['charge_end_v']=float(c.voltage.iloc[-1]);r['charge_end_a']=float(c.current.iloc[-1])
            r['charge_complete']=bool((c.voltage<4.18).sum()>=5 and c.voltage.max()>=CONFIG['charge_voltage_v'] and c.voltage.iloc[-1]>=CONFIG['charge_voltage_v'] and c.current.iloc[-1]<=CONFIG['termination_current_a'])
            if not r['charge_complete']:reasons.append('charge_endpoint_incomplete')
        else:r['charge_complete']=False
        if discharge:
            z=pd.concat(discharge)
            r['discharge_start_v']=float(z.voltage.iloc[0]);r['discharge_end_v']=float(z.voltage.iloc[-1])
            r['discharge_crate_median']=float(abs(z.current).median()/CONFIG['nominal_capacity_ah'])
            r['discharge_crate_max']=float(abs(z.current).max()/CONFIG['nominal_capacity_ah'])
            r['discharge_complete']=bool(z.voltage.iloc[-1]<=CONFIG['discharge_cutoff_v'] and z.voltage.iloc[-1]>=2.6)
            if not r['discharge_complete']:reasons.append('discharge_endpoint_nonstandard_or_incomplete')
            if not (0.02<=r['discharge_integral_ah']<=1.8):reasons.append('implausible_discharge_capacity')
            r['protocol']=classify_protocol(r['discharge_crate_median'],r['discharge_crate_max'],len(discharge))
            conventional=r['protocol']=='conventional'
            if not conventional:flags.append('protocol_dependent_soh')
            if len(discharge)>1:flags.append('multiple_discharge_segments')
        else:r['discharge_complete']=False;r['protocol']='unknown'
        # Label is the observed validated current integral, independent of feature availability.
        r['rejection_reasons']=';'.join(sorted(set(reasons)))
        r['quality_flags']=';'.join(sorted(set(flags)))
        r['validated_complete']=not reasons
        r['validated_discharge_capacity_ah']=r['discharge_integral_ah'] if not reasons else np.nan
        r['soh']=r['validated_discharge_capacity_ah']/CONFIG['nominal_capacity_ah']
        r['soh_candidate']=r['discharge_integral_ah']/CONFIG['nominal_capacity_ah']
        quality.append(r);cycles.append((r,parts))
    return cycles,pd.DataFrame(quality)
