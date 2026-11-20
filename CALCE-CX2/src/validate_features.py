"""Quality reports without silently removing candidates or filling missing physics."""
import numpy as np
import pandas as pd
from .extract_features import FEATURES

BOUNDS={'partial_charge_ah':(0,1.5),'partial_dv_dq':(0,200), 'ic_peak_height':(0,100),
        'ic_peak_voltage':(3.8,4.15),'ic_peak_area':(0,1.5),'ic_peak_width':(0,.35),
        'cc_duration_s':(0,30000),'cv_duration_s':(0,30000),'cv_capacity_ah':(0,1.5),
        'cv_decay_rate_s_inv':(0,1),'charge_crate':(0,5),'temperature_c':(-20,80),
        'resistance_ohm':(0,1),'relaxation_60s_v':(-.5,.5)}

def validate(frame,output):
    f=frame.copy();events=[]
    for col,(lo,hi) in BOUNDS.items():
        bad=f[col].notna()&(~np.isfinite(f[col])|~f[col].between(lo,hi))
        for r in f.loc[bad,['battery','cycle_id',col]].itertuples(index=False,name=None):
            events.append(dict(battery=r[0],cycle_id=r[1],feature=col,value=r[2],reason='outside_documented_plausibility_bounds'))
        # Preserve original numeric evidence in a separate audit column.
        if bad.any():f.loc[bad,col+'_raw_implausible']=f.loc[bad,col];f.loc[bad,col]=np.nan
    pd.DataFrame(events,columns=['battery','cycle_id','feature','value','reason']).to_csv(output/'implausible_features.csv',index=False)
    missing=[]
    for scope,data in [('all_candidates',f),('validated_complete',f[f.validated_complete])]:
        for battery,g in [('ALL',data),*list(data.groupby('battery'))]:
            for feature in FEATURES:
                missing.append(dict(scope=scope,battery=battery,feature=feature,records=len(g),missing=int(g[feature].isna().sum()),missing_percent=100*g[feature].isna().mean() if len(g) else np.nan))
    pd.DataFrame(missing).to_csv(output/'feature_missingness.csv',index=False)
    summary=f.groupby('battery').agg(candidate_cycles=('cycle_id','size'),validated_complete_cycles=('validated_complete','sum'),
        conventional_validated=('eligible_analysis','sum'),first_experiment=('start','min'),last_experiment=('end','max'))
    summary['rejected_or_incomplete']=summary.candidate_cycles-summary.validated_complete_cycles
    summary.to_csv(output/'battery_summary.csv')
    counts=f.loc[~f.validated_complete,'rejection_reasons'].str.split(';').explode().value_counts()
    counts.rename_axis('reason').rename('count').to_csv(output/'rejection_reason_counts.csv')
    return f
