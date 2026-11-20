"""Missing CC/CV information masks feature dependencies, including aggregate features."""
import json
import numpy as np
import pandas as pd
from cs2_research_pipeline import OUT
from cs2_research_analysis import model,metrics,phase_dependent_columns

def run_acquisition_robustness():
 manifest=json.loads((OUT/'dataset_manifest.json').read_text());selected=manifest['selected_features'];cc,cv=phase_dependent_columns(selected);rows=[]
 for h in [1,3,5]:
  pairs=pd.read_csv(OUT/f'forecast_pairs_h{h}.csv');tr=pairs[pairs.split=='train'];test=pairs[pairs.split=='test']
  if len(tr)<30 or test.empty:continue
  rng=np.random.default_rng(42);xx=tr[selected].copy();random=xx.copy();random[rng.random(xx.shape)<.25]=np.nan;cc_aug=xx.copy();cc_aug[cc]=np.nan;cv_aug=xx.copy();cv_aug[cv]=np.nan;history_aug=xx.copy();history_aug.iloc[len(xx)//3:len(xx)//3+max(1,len(xx)//5),:]=np.nan
  for kind in ['ridge','hgb']:
   m=model(kind);m.fit(pd.concat([xx,random,cc_aug,cv_aug,history_aug]),pd.concat([tr.target_soh]*5))
   for b,g in test.groupby('battery'):
    rng=np.random.default_rng(42+h);mask=rng.random((len(g),len(selected)))<.25
    for scenario in ['complete','missing_cc','missing_cv','random_missing','consecutive_missing_origins']:
     tx=g[selected].copy()
     if scenario=='missing_cc':tx[cc]=np.nan
     if scenario=='missing_cv':tx[cv]=np.nan
     if scenario=='random_missing':tx[mask]=np.nan
     if scenario=='consecutive_missing_origins':tx.iloc[len(tx)//3:len(tx)//3+max(1,len(tx)//5),:]=np.nan
     rows.append(dict(horizon=h,battery=b,model=kind,scenario=scenario,**metrics(g.target_soh,m.predict(tx))))
 result=pd.DataFrame(rows,columns=['horizon','battery','model','scenario','rmse','mae','r2','n']);result.to_csv(OUT/'robustness_results.csv',index=False)
 if len(result):result.groupby(['horizon','model','scenario'])[['rmse','mae','r2']].mean().reset_index().to_csv(OUT/'robustness_battery_macro.csv',index=False)
 (OUT/'robustness_mask_definition.json').write_text(json.dumps(dict(missing_cc=cc,missing_cv=cv,random_feature_probability=.25,training='Complete/random/CC/CV/contiguous missing augmentation; identical across Ridge/HGB',interpretation='CC/CV masks remove all selected features that require the missing phase, including aggregate charging features. This is feature masking; raw partial charging is a separate experiment.'),indent=2))
 print('Acquisition masking rows:',len(result),flush=True)
 return result
if __name__=='__main__':run_acquisition_robustness()
