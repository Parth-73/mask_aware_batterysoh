"""Bounded raw-sample partial-charge diagnostic; distinct from artificial feature masking."""
from pathlib import Path
import json
import pandas as pd
import numpy as np
from cs2_research_pipeline import OUT,read_workbook,reconstruct,charge_features
from cs2_research_analysis import model,metrics

def run_partial_views():
 manifest=json.loads((OUT/'dataset_manifest.json').read_text());selected=manifest['selected_features'];data=pd.read_csv(OUT/'cs2_cycle_features_all.csv',low_memory=False)
 trainb=manifest['train_batteries'];testb=manifest['test_batteries'];rows=[];sources=[]
 for battery in trainb+testb:
  counts=data[(data.battery==battery)&data.soh.notna()].groupby('source_file').size()
  possible=[Path(p) for p,n in counts.items() if n>=5]
  if not possible:continue
  p=min(possible,key=lambda p:p.stat().st_size);sources.append(str(p))
  def observe(c,sign,row):
   if not row['label_valid'] or len(c)<12:return
   for name,view in [('complete',c),('prefix_50_percent_samples',c.iloc[:max(1,len(c)//2)]),('prefix_80_percent_samples',c.iloc[:max(1,int(.8*len(c)))]),('suffix_50_percent_samples',c.iloc[len(c)//2:])]:
    features,quality=charge_features(view,sign)
    rows.append(dict(battery=battery,source_file=str(p),episode_in_file=row['episode_in_file'],soh=row['soh'],view=name,n_charge_samples=len(view),**features))
  frames,_=read_workbook(p);reconstruct(p,frames,observer=observe)
  print('Partial raw views',battery,p.name,flush=True)
 views=pd.DataFrame(rows);views.to_csv(OUT/'physical_partial_observation_features.csv',index=False);results=[]
 if len(views) and selected:
  tr=views[views.battery.isin(trainb)]
  for kind in ['ridge','hgb']:
   m=model(kind);m.fit(tr[selected],tr.soh)
   for (b,view),g in views[views.battery.isin(testb)].groupby(['battery','view']):results.append(dict(battery=b,view=view,model=kind,training='complete_and_raw_partial_charge_views',**metrics(g.soh,m.predict(g[selected]))))
 pd.DataFrame(results,columns=['battery','view','model','training','rmse','mae','r2','n']).to_csv(OUT/'physical_partial_observation_results.csv',index=False)
 (OUT/'physical_partial_observation_manifest.json').write_text(json.dumps(dict(status='executed',source_files=sources,n_rows=len(views),scope='One smallest eligible raw workbook per split battery; current-cycle estimation only; sample-prefix/suffix views re-extracted from measured samples; no claim of acquisition invariance'),indent=2))
 if results:pd.DataFrame(results).groupby(['view','model'])[['rmse','mae','r2']].mean().reset_index().to_csv(OUT/'physical_partial_observation_battery_macro.csv',index=False)
 return results
if __name__=='__main__':run_partial_views()
