"""Missing historical observations: remove complete lag vectors in verified windows."""
import json
import numpy as np
import pandas as pd
from cs2_research_pipeline import OUT
from cs2_research_analysis import model,metrics

def run_history_robustness():
 manifest=json.loads((OUT/'dataset_manifest.json').read_text());selected=manifest['selected_features'];data=pd.read_csv(OUT/'cs2_cycle_features_all.csv',low_memory=False)
 data=data.sort_values(['battery','source_file','episode_in_file']);keys=['battery','source_file','history_segment'];groups=data.groupby(keys)
 xx=data[['cycle_id']+selected].copy();cols=list(selected)
 for lag in range(1,5):
  lagged=groups[selected].shift(lag);lagged.columns=[f'lag{lag}__{c}' for c in selected];xx=pd.concat([xx,lagged],axis=1);cols+=list(lagged.columns)
 xx['four_prior_observations_exist']=groups.cycle_id.shift(4).notna()
 rows=[]
 for h in [1,3,5]:
  pairs=pd.read_csv(OUT/f'forecast_pairs_h{h}.csv')
  if pairs.empty:continue
  base=pairs.drop(columns=selected,errors='ignore').merge(xx,on='cycle_id',how='left',validate='one_to_one');base=base[base.four_prior_observations_exist]
  tr=base[base.battery.isin(manifest['train_batteries'])]
  if len(tr)<30:continue
  trainmask=tr[cols].copy();drop=[f'lag{lag}__{c}' for lag in [1,2] for c in selected];trainmask[drop]=np.nan
  for kind in ['ridge','hgb']:
   m=model(kind);m.fit(pd.concat([tr[cols],trainmask]),pd.concat([tr.target_soh,tr.target_soh]))
   for b,g in base[base.battery.isin(manifest['test_batteries'])].groupby('battery'):
    if len(g)<5:continue
    for scenario in ['four_observed_history_vectors','two_consecutive_history_observations_missing']:
     tx=g[cols].copy()
     if scenario.startswith('two_'):tx[drop]=np.nan
     rows.append(dict(horizon=h,battery=b,model=kind,scenario=scenario,**metrics(g.target_soh,m.predict(tx))))
 result=pd.DataFrame(rows,columns=['horizon','battery','model','scenario','rmse','mae','r2','n']);result.to_csv(OUT/'historical_observation_robustness_results.csv',index=False)
 if len(result):result.groupby(['horizon','model','scenario'])[['rmse','mae','r2']].mean().reset_index().to_csv(OUT/'historical_observation_robustness_battery_macro.csv',index=False)
 (OUT/'historical_observation_robustness_protocol.json').write_text(json.dumps(dict(status='executed',lags=4,masked_lags=[1,2],input_features=selected,training='Complete and masked-history augmentation; identical windows and masks across Ridge/HGB',window_rule='Within source, no missing raw episode indices or unsupported continuity segment; all four preceding observation identities must exist',interpretation='Acquisition dropout of historical feature vectors; not physical partial-charge invariance'),indent=2))
 print('Historical-observation robustness rows:',len(result),flush=True)
 return result
if __name__=='__main__':run_history_robustness()
