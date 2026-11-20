"""Add conservative cross-source acquisition-gap and overlap checks to event audit."""
import json
import pandas as pd
from src.common import save_json, digest
from .audit import OUT


def finalize():
    m=pd.read_csv(OUT/'data_audit/inspection_manifest.csv')
    m=m[m.status.eq('ready')].copy()
    m['start']=pd.to_datetime(m.start);m['end']=pd.to_datetime(m.end)
    e=pd.read_csv(OUT/'rest_events/rest_event_inventory.csv')
    e['event_start']=pd.to_datetime(e.event_start);e['event_end']=pd.to_datetime(e.event_end)
    additions=[]
    for battery,g in m.groupby('battery'):
        g=g.sort_values('start');previous=None
        for row in g.itertuples():
            if previous is not None:
                gap=(row.start-previous.end).total_seconds()
                if gap>0:
                    additions.append(dict(battery=battery,preceding_cycle_id='',following_cycle_id='',event_start=previous.end,event_end=row.start,
                        duration_s=gap,current_abs_max_a=float('nan'),category='uncertain_recording_gap',source_identifiers=previous.source_file+'::'+str(previous.sheet)+';'+row.source_file+'::'+str(row.sheet),
                        verification_status=False,uncertainty_reason='cross_source_gap_no_observed_current',validated_pair=False,consistent_discharge_conditions=False,available_by_following_charge=False))
            if previous is None or row.end>previous.end:previous=row
        # A time-overlapping second source blocks verification without proving equivalence.
        for row in g.itertuples():
            source=row.source_file+'::'+str(row.sheet)
            overlap=e.battery.eq(battery)&e.source_identifiers.ne(source)&e.event_start.lt(row.end)&e.event_end.gt(row.start)
            # Only simultaneous exports, not neighboring non-overlapping sessions.
            if overlap.any():
                e.loc[overlap,'category']='uncertain_recording_gap';e.loc[overlap,'verification_status']=False
                e.loc[overlap,'uncertainty_reason']='overlapping_acquisition_sources_not_independently_resolved'
    e=pd.concat([e,pd.DataFrame(additions)],ignore_index=True).drop_duplicates(['battery','event_start','event_end','source_identifiers'])
    e.to_csv(OUT/'rest_events/rest_event_inventory.csv',index=False)
    v=e[e.category.eq('verified_measured_rest')]
    usable=v[v.validated_pair & v.consistent_discharge_conditions & v.available_by_following_charge]
    summary=dict(verified_events=len(v),uncertain_gaps=int(e.category.eq('uncertain_recording_gap').sum()),
        per_battery=v.groupby('battery').size().to_dict(),duration_quantiles_s={str(k):float(x) for k,x in v.duration_s.quantile([0,.25,.5,.75,.95,1]).items()},
        validated_pairs=int(v.validated_pair.sum()),consistent_conditions=int(v.consistent_discharge_conditions.sum()),usable_pair_events=len(usable),
        unique_target_cycles=int(usable.following_cycle_id.nunique()),usable_batteries=sorted(usable.battery.unique()),
        long_verified_events_gt_1h=int((v.duration_s>3600).sum()),
        recovery_training_status='requires_duration_diversity_and_state_history_review; ordinary_short_rest_is_not_irregular_long_rest_validation')
    save_json(OUT/'rest_events/feasibility.json',summary)
    (OUT/'reports/REST_DATA_FEASIBILITY.md').write_text('# Rest data feasibility\n\n```json\n'+json.dumps(summary,indent=2)+'\n```\n\n|I| <=0.005 A at clock-consistent consecutive observations <=180 s apart defines sampled measured rest. No cross-source gap is open-circuit evidence. Overlapping acquisition sources downgrade verification conservatively. Events are joined to the preceding discharge endpoint and following charge cutoff, never from SOH changes; pair eligibility requires consecutive validated cycles and discharge C-rate agreement within 0.02C. Counts are events, with unique target cycles reported separately. Label-derived discharge conditions are retrospective evaluation metadata only. A cutoff-valid event is not independent evidence of capacity recovery. Long-rest data support and unknown-history state initialization must be reviewed before any recovery training.\n',encoding='utf-8')
    save_json(OUT/'processed/manifest.json',dict(samples_sha256=digest(OUT/'processed/samples.csv'),curves_sha256=digest(OUT/'processed/curves.npy'),
        raw_hash_inventory_sha256=digest(OUT/'data_audit/raw_hashes.csv'),channel_order=['Q_Ah','dQdV_Ah_per_V','I_A'],
        voltage_interval_v=[3.8,4.15],spacing_v=.002,missing_curve_policy='reject_no_extrapolation',
        feature_source='unchanged_historical_charging_features',parser='mixed_original_openpyxl_cache_and_calamine_with_historical_target_checks'))
    print(json.dumps(summary),flush=True)

if __name__=='__main__':finalize()
