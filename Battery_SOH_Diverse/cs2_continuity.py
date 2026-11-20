"""Validate episode adjacency from raw rest samples and operating-step clocks."""
import hashlib,json
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from cs2_research_pipeline import OUT,read_workbook,interval_evidence

def audit_continuity(data=None):
 if data is None:data=pd.read_csv(OUT/'cs2_cycle_features_all.csv',low_memory=False)
 cache=OUT/'continuity_checkpoints';cache.mkdir(exist_ok=True);allrows=[]
 for source,g in tqdm(data.dropna(subset=['episode_in_file']).groupby('source_file'),desc='Verify raw episode boundaries',mininterval=5):
  p=Path(source);g=g.sort_values('episode_in_file');key=hashlib.sha256((source+str(p.stat().st_size)+str(p.stat().st_mtime_ns)+'continuity-v1').encode()).hexdigest()[:24];cp=cache/(key+'.json')
  if cp.exists():allrows.extend(json.loads(cp.read_text()));continue
  rows=[]
  try:
   frames,_=read_workbook(p)
   if not frames:continue
   d=pd.concat(frames,ignore_index=True);d=d.drop_duplicates(subset=[c for c in ['t','i','v','cycle','step'] if c in d]).sort_values('t',kind='stable').reset_index(drop=True);times=d.t.to_numpy(float)
   prev=None
   for _,r in g.iterrows():
    row=dict(battery=r.battery,source_file=source,episode_in_file=float(r.episode_in_file),prior_episode_continuity_verified=False,continuity_reason='first_or_missing_charge',observed_rest_samples_between_episodes=0)
    if prev is not None and r.episode_in_file-prev.episode_in_file==1 and np.isfinite(r.charge_start_s):
     lo=int(np.searchsorted(times,prev.discharge_end_s));hi=int(np.searchsorted(times,r.charge_start_s));window=d.iloc[lo:hi+1]
     endpoint_match=len(window)>=2 and abs(window.t.iloc[0]-prev.discharge_end_s)<1e-4 and abs(window.t.iloc[-1]-r.charge_start_s)<1e-4
     middle=window.iloc[1:-1]
     rest=(middle.i.abs()<=.01).all()
     clocks=interval_evidence(window).all() if len(window) else False
     valid=bool(endpoint_match and rest and clocks)
     row.update(prior_episode_continuity_verified=valid,continuity_reason='verified_rest_samples_and_step_clock' if valid else 'unknown_interval_or_intervening_active_samples',observed_rest_samples_between_episodes=len(middle),gap_before_charge_s=float(r.charge_start_s-prev.discharge_end_s))
    rows.append(row);prev=r
  except Exception as e:rows=[dict(battery=g.battery.iloc[0],source_file=source,episode_in_file=float(r.episode_in_file),prior_episode_continuity_verified=False,continuity_reason='read_error:'+str(e)) for _,r in g.iterrows()]
  cp.write_text(json.dumps(rows,default=str));allrows.extend(rows)
 result=pd.DataFrame(allrows);result.to_csv(OUT/'cs2_forecast_continuity.csv',index=False)
 print('Verified episode boundaries:',int(result.prior_episode_continuity_verified.sum()),flush=True)
 return result
if __name__=='__main__':audit_continuity()
