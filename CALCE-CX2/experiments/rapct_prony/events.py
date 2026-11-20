"""Causal event availability for each retained charging cutoff; no capacity inputs."""
import json
import numpy as np
import pandas as pd
from src.common import save_json
from .audit import OUT


def build():
    f=pd.read_csv(OUT/'processed/samples.csv')
    f['prediction_cutoff']=pd.to_datetime(f.prediction_cutoff)
    f['start']=pd.to_datetime(f.start)
    e=pd.read_csv(OUT/'rest_events/rest_event_inventory.csv')
    e['event_start']=pd.to_datetime(e.event_start);e['event_end']=pd.to_datetime(e.event_end)
    rows=[]
    for battery,g in f.groupby('battery'):
        events=e[e.battery.eq(battery)]
        previous=None
        for r in g.sort_values('lifetime_cycle').itertuples():
            consecutive=bool(previous is not None and r.lifetime_cycle==previous.lifetime_cycle+1 and r.sources==previous.sources and r.local_counter_resets==0)
            beginning=previous.prediction_cutoff if consecutive else r.start
            available=events[events.event_start.ge(beginning)&events.event_end.le(r.prediction_cutoff)]
            v=available[available.category.eq('verified_measured_rest')]
            unknown=bool(available.category.eq('uncertain_recording_gap').any())
            rows.append(dict(battery=battery,cycle_id=r.cycle_id,prediction_cutoff=r.prediction_cutoff,
                verified_rest_indicator=bool(len(v)),verified_rest_duration_s=float(v.duration_s.sum()),
                verified_event_count=len(v),unknown_recording_gap=unknown,consecutive_observed_history=consecutive,
                state_history_usable=bool(consecutive and not unknown)))
            previous=r
    table=pd.DataFrame(rows)
    table.to_csv(OUT/'processed/causal_events.csv',index=False)
    summary=table.groupby('battery').agg(samples=('cycle_id','size'),with_verified_rest=('verified_rest_indicator','sum'),
        unknown_gaps=('unknown_recording_gap','sum'),usable_state_transitions=('state_history_usable','sum'))
    summary.to_csv(OUT/'rest_events/causal_state_support.csv')
    details=json.loads((OUT/'rest_events/feasibility.json').read_text())
    details['matched_curve_state_support']=summary.reset_index().to_dict('records')
    count=int(table.state_history_usable.sum())
    details['recovery_training_status']='unavailable_under_conservative_gap_policy_zero_usable_persistent_state_transitions' if count==0 else 'additional_independent_battery_and_duration_support_review_required'
    save_json(OUT/'rest_events/feasibility.json',details)
    save_json(OUT/'rest_events/recovery_gate.json',dict(experimental_recovery_enabled=False,usable_state_transitions=count,
        reason='Unknown acquisition gaps prevent documented memory evolution.' if count==0 else 'Independent event support and initialization review is pending.',
        permitted='synthetic_single_and_multi_mode_tests',future_resolution='document sparse-CV gap dynamics or acquire continuous rest records'))
    report=OUT/'reports/REST_DATA_FEASIBILITY.md'
    before=report.read_text(encoding='utf-8').split('## Persistent-state feasibility gate')[0]
    report.write_text(before+'## Persistent-state feasibility gate\n\n```csv\n'+summary.to_csv()+'\n```\n\n'+details['recovery_training_status']+'\n\nSparse CV event logging can pass capacity counter checks without independently documenting memory dynamics inside the unobserved interval. No gap is relabeled as open-circuit rest. This conservative policy is a limitation of this execution, not proof that CALCE lacks recovery behavior.\n',encoding='utf-8')
    print(summary.to_string(),flush=True)

if __name__=='__main__':build()
