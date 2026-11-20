"""Conservative, resumable CALCE raw-channel research pipeline."""
from pathlib import Path
import re, json, hashlib, logging, argparse, warnings
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
ROOT=Path(__file__).resolve().parent/'CS2'
OUT=ROOT/'cs2_feature_research'
CACHE=OUT/'checkpoints'
CONFIG=dict(version=4,seed=42,nominal_capacity_ah=1.1,rest_current_a=.01,active_current_a=.03,max_interval_s=120,full_charge_v=4.18,full_discharge_end_v=2.72,minimum_discharge_points=12,capacity_agreement_fraction=.03,metadata_url='https://calce.umd.edu/battery-data')
def battery_id(p):
 ids=set(re.findall(r'(?<![A-Za-z0-9])CS2[_ -](\d+)(?!\d)',str(p),re.I))
 if len(ids)!=1: raise ValueError(f'Ambiguous battery tokens: {ids}')
 return 'CS2_'+ids.pop()
def discover():
 return sorted(p for p in ROOT.rglob('*') if p.suffix.lower() in {'.xlsx','.xls','.txt'} and not any(x in p.parts for x in ['cs2_processed','cs2_feature_research']) and not p.name.startswith('~$'))
def source_snapshot_sha256():
 signatures=[(str(p),p.stat().st_size,p.stat().st_mtime_ns) for p in discover()]
 return hashlib.sha256(json.dumps(dict(sources=signatures,config=CONFIG),sort_keys=True).encode()).hexdigest()

def norm(s): return re.sub('[^a-z0-9]','',str(s).lower())
ALIASES={'t':['Test_Time(s)','Test Time','Time(s)'],'v':['Voltage(V)','Voltage'],'i':['Current(A)','Current'],'cycle':['Cycle_Index','Cycle'],'step':['Step_Index'],'step_t':['Step_Time(s)'],'stamp':['Date_Time'],'qd':['Discharge_Capacity(Ah)'],'qc':['Charge_Capacity(Ah)'],'temp':['Temperature(C)','Temperature','Aux_Temperature_1(C)']}
def mapping(cols):
 lookup={norm(c):c for c in cols}
 return {k:lookup[norm(a)] for k,aliases in ALIASES.items() for a in aliases if norm(a) in lookup}
def read_workbook(p):
 inventory=[]; frames=[]
 if p.suffix=='.txt':
  d=pd.read_csv(p,sep='\t')
  required={'Time','Duration','mA','mV','Pgm step'}
  if not required.issubset(d):return [],[dict(path=str(p),battery=battery_id(p),sheet='CADEX text',status='manual_review_unrecognized_cadex_schema')]
  # Independently verify clock units from the two logged clocks; Capacity is a percentage, not Ah.
  fit=np.polyfit(d.Time,d.Duration,1);residual=float(np.max(np.abs(d.Duration-np.polyval(fit,d.Time))))
  valid_clock=abs(fit[0]-60)<.1 and residual<5 and (d.Duration.diff().dropna()>0).all()
  rec=dict(path=str(p),battery=battery_id(p),sheet='CADEX text',status='raw_cadex_clock_verified' if valid_clock else 'manual_review_cadex_clock',time_to_duration_slope=float(fit[0]),clock_max_residual_s=residual,capacity_column_interpretation='percent_not_Ah_do_not_use_as_capacity_label')
  if not valid_clock:return [],[rec]
  f=pd.DataFrame(dict(t=d.Duration.astype(float),i=d.mA/1000,v=d.mV/1000,step=d['Pgm step'],cycle=d['Discharge count'],temp=d.Temperature if 'Temperature' in d else np.nan))
  f['stamp']=pd.NaT;f['source_sheet']='CADEX text'
  # No artificial absolute timestamp is assigned from a filename date.
  return [f],[rec]
 x=pd.ExcelFile(p,engine='calamine')
 for sh in x.sheet_names:
  rec=dict(path=str(p),battery=battery_id(p),sheet=sh)
  if not sh.lower().startswith('channel'):
   rec['status']='statistics_excluded' if sh.lower().startswith('statistics') else 'metadata'; inventory.append(rec);continue
  d=pd.read_excel(x,sheet_name=sh,header=None,nrows=20)
  best=max(range(len(d)),key=lambda j:len(mapping(d.iloc[j].tolist())))
  d=pd.read_excel(x,sheet_name=sh,header=best); m=mapping(d.columns)
  rec.update(header_row=best,mapping=json.dumps(m),status='raw' if {'t','i','v'}.issubset(m) else 'unrecognized_schema');inventory.append(rec)
  if rec['status']!='raw':continue
  f=pd.DataFrame({k:pd.to_numeric(d[c],errors='coerce') for k,c in m.items() if k!='stamp'})
  if 'stamp' in m:
   s=d[m['stamp']]
   if pd.api.types.is_numeric_dtype(s): f['stamp']=pd.to_datetime(s,unit='D',origin='1899-12-30',errors='coerce')
   else:f['stamp']=pd.to_datetime(s,errors='coerce')
  else:f['stamp']=pd.NaT
  f['source_sheet']=sh;frames.append(f)
 return frames,inventory

def integral(t,i):
 dt=np.diff(t); ok=np.isfinite(dt)&(dt>0)&(dt<=CONFIG['max_interval_s'])&np.isfinite(i[:-1])&np.isfinite(i[1:])
 return float(np.sum(dt[ok]*(i[:-1][ok]+i[1:][ok])/2))
def stats(a):
 return (float(np.nanmean(a)),float(np.nanstd(a))) if len(a) else (np.nan,np.nan)
def slope(t,v):
 return float(np.polyfit(t-t[0],v,1)[0]) if len(t)>3 and np.ptp(t)>0 else np.nan

def interval_evidence(g):
 dt=g.t.diff().to_numpy(float)
 valid=np.isfinite(dt)&(dt>0)&(dt<=120)
 if 'step' in g and 'step_t' in g:
  same=g.step.eq(g.step.shift()).to_numpy()
  clock=np.isclose(g.step_t.diff().to_numpy(float),dt,atol=.5,rtol=1e-4)
  # A recorded zero-current operating step and matching step clock establish rest.
  rest=(g.i.abs()<=.01)&(g.i.shift().abs()<=.01)
  supported=rest.to_numpy().copy()
  for counter,sgn in [('qc',1),('qd',-1)]:
   if counter in g:
    change=g[counter].diff().to_numpy(float)
    active=(g.i*sgn>.03)&(g.i.shift()*sgn>.03)
    supported |= active.to_numpy()&np.isfinite(change)&(change>0)
  valid |= same&clock&supported&(dt>0)
  # For a transition, Step_Time locates the new step start at the preceding recorded boundary.
  transition=~same&np.isfinite(dt)&(dt>0)
  start=g.t.to_numpy(float)-g.step_t.to_numpy(float)
  prev=g.t.shift().to_numpy(float)
  valid |= transition&np.isclose(start,prev,atol=1,rtol=0)
 valid[0]=True
 return valid

def measured_integral(g,kind='charge',energy=False):
 if len(g)<2:return 0.
 t=g.t.to_numpy(float);i=np.abs(g.i.to_numpy(float));dt=np.diff(t)
 valid=interval_evidence(g)[1:]
 same=g.step.to_numpy()[1:]==g.step.to_numpy()[:-1] if 'step' in g else np.ones(len(dt),bool)
 valid &= same
 trap=dt*(i[:-1]+i[1:])/7200
 counter='qc' if kind=='charge' else 'qd'
 if counter in g:
  delta=np.diff(g[counter].to_numpy(float));use=(dt>120)&valid&np.isfinite(delta)&(delta>=0)
  trap[use]=delta[use]
 if energy:trap*= (g.v.to_numpy(float)[:-1]+g.v.to_numpy(float)[1:])/2
 return float(np.sum(trap[valid]))

def charge_features(c,sign):
 names=['charge_duration_s','charge_throughput_ah','charge_energy_wh','charge_voltage_integral_vs','charge_voltage_mean_v','charge_voltage_std_v','cc_duration_s','cc_throughput_ah','cc_current_mean_a','cc_current_std_a','cc_initial_voltage_v','cc_final_voltage_v','cc_voltage_slope_v_s','cc_voltage_increase_v','cv_duration_s','cv_throughput_ah','cv_current_mean_a','cv_current_decay_a_s','cv_duration_ratio','cv_throughput_ratio','cv_decay_tau_s','time_3p8_4p0_s','time_3p9_4p1_s','throughput_bracket_3p8_4p0_ah','throughput_bracket_3p9_4p1_ah','ica_peak_height_ah_v','ica_peak_voltage_v','ica_peak_width_v','ica_peak_area_ah','ica_peak_count','dva_peak_height_v_ah','dva_peak_location_ah','dva_peak_width_ah','dva_curve_area_v','charge_temperature_mean_c','charge_temperature_max_c','charge_temperature_rise_c','charge_c_rate']
 f=dict.fromkeys(names,np.nan);q=dict(charge_quality='absent',ica_status='insufficient_coverage',dva_status='insufficient_coverage')
 if len(c)<12:return f,q
 t=c.t.to_numpy(float);v=c.v.to_numpy(float);i=c.i.to_numpy(float)*sign;dt=np.diff(t)
 q.update(charge_quality='valid' if interval_evidence(c).all() else 'discontinuous',sampling_median_s=float(np.median(dt)),sampling_max_s=float(np.max(dt)),charge_window_min_v=float(v.min()),charge_window_max_v=float(v.max()))
 if q['charge_quality']!='valid' and not c.attrs.get('linked_charge',False):return f,q
 if c.attrs.get('linked_charge',False):q['charge_quality']='validated_step_clock_counter_and_phase_linkage'
 total=measured_integral(c)
 f.update(charge_duration_s=float(sum(np.ptp(z.t) for _,z in c.groupby('step'))) if 'step' in c else float(np.ptp(t)),charge_throughput_ah=total,charge_energy_wh=measured_integral(c,energy=True),charge_voltage_integral_vs=integral(t,v),charge_voltage_mean_v=float(v.mean()),charge_voltage_std_v=float(v.std()))
 # Plateau plus sustained current decay is required; threshold alone cannot establish CV.
 plateau=np.flatnonzero(v>=4.18)
 cvstart=None
 if len(plateau)>=8:
  j=plateau[0];tail=i[j:];vv=v[j:]
  if len(tail)>=8 and np.ptp(vv)<=.045 and np.median(tail[-max(2,len(tail)//4):])<.8*np.median(i[:min(10,j+1)]):cvstart=j
 cc=c.iloc[:cvstart] if cvstart is not None else c
 if len(cc)>=12:
  tt=cc.t.to_numpy(float);vv=cc.v.to_numpy(float);ii=cc.i.to_numpy(float)*sign
  if np.std(ii)/max(np.mean(ii),1e-9)<.05:
   f.update(cc_duration_s=float(np.ptp(tt)),cc_throughput_ah=measured_integral(cc),cc_current_mean_a=float(ii.mean()),cc_current_std_a=float(ii.std()),cc_initial_voltage_v=float(vv[0]),cc_final_voltage_v=float(vv[-1]),cc_voltage_slope_v_s=slope(tt,vv),cc_voltage_increase_v=float(vv[-1]-vv[0]),charge_c_rate=float(ii.mean()/1.1))
 if cvstart is not None:
  tt=t[cvstart:];ii=i[cvstart:];cq=measured_integral(c.iloc[cvstart:])
  f.update(cv_duration_s=float(np.ptp(tt)),cv_throughput_ah=cq,cv_current_mean_a=float(ii.mean()),cv_current_decay_a_s=slope(tt,ii),cv_duration_ratio=float(np.ptp(tt)/np.ptp(t)),cv_throughput_ratio=cq/total if total>0 else np.nan)
  fit=np.polyfit(tt-tt[0],np.log(ii),1);pred=np.polyval(fit,tt-tt[0]);den=np.sum((np.log(ii)-np.log(ii).mean())**2)
  if fit[0]<0 and den>0 and 1-np.sum((np.log(ii)-pred)**2)/den>.9:f['cv_decay_tau_s']=-1/fit[0]
 for a,b,tag in [(3.8,4.,'3p8_4p0'),(3.9,4.1,'3p9_4p1')]:
  if v[0]<=a and v.max()>=b:
   idx1=np.flatnonzero((v[:-1]<=a)&(v[1:]>=a)&(np.diff(v)>0));idx2=np.flatnonzero((v[:-1]<=b)&(v[1:]>=b)&(np.diff(v)>0))
   if len(idx1) and len(idx2) and idx2[0]>idx1[0]:
    ja,jb=idx1[0],idx2[0];ta=np.interp(a,v[ja:ja+2],t[ja:ja+2]);tb=np.interp(b,v[jb:jb+2],t[jb:jb+2]);f['time_'+tag+'_s']=tb-ta
    f['throughput_bracket_'+tag+'_ah']=integral(t[ja:jb+2],i[ja:jb+2])/3600
 # ICA/DVA restricted to dense, monotone CC coverage. Require peak persistence across smoothing scales.
 if len(cc)>=100 and np.isfinite(f['cc_duration_s']):
  from scipy.signal import savgol_filter,find_peaks,peak_widths
  tt=cc.t.to_numpy(float);vv=cc.v.to_numpy(float);ii=cc.i.to_numpy(float)*sign
  dq=np.r_[0,np.cumsum(np.diff(tt)*(ii[:-1]+ii[1:])/7200)]
  if vv.min()<=3.8 and vv.max()>=4.15 and np.mean(np.diff(vv)<-.002)<.01:
   uq,ix=np.unique(np.maximum.accumulate(vv),return_index=True)
   if len(uq)>80 and np.max(np.diff(uq))<.02:
    grid=np.arange(max(3.8,uq.min()),min(4.15,uq.max()),.002);curve=np.interp(grid,uq,dq[ix]);der=np.gradient(savgol_filter(curve,11,2),grid);der2=np.gradient(savgol_filter(curve,17,2),grid)
    peaks,_=find_peaks(der,prominence=max(.05,.1*np.ptp(der)),distance=8)
    stable=[j for j in peaks if any(abs(j-k)<=4 for k in find_peaks(der2,prominence=max(.05,.1*np.ptp(der2)),distance=8)[0])]
    if stable:
     j=max(stable,key=lambda x:der[x]);w=peak_widths(der,[j],rel_height=.5);lo,hi=int(w[2][0]),int(w[3][0])+1
     f.update(ica_peak_height_ah_v=float(der[j]),ica_peak_voltage_v=float(grid[j]),ica_peak_width_v=float(w[0][0]*.002),ica_peak_area_ah=float(np.trapezoid(der[lo:hi],grid[lo:hi])),ica_peak_count=len(stable));q['ica_status']='stable_across_11_17_point_smoothing'
   if dq[-1]>.2:
    grid=np.linspace(dq[0],dq[-1],150);dv=np.gradient(savgol_filter(np.interp(grid,dq,vv),11,2),grid);pk,_=find_peaks(dv,prominence=.1*np.ptp(dv),distance=8)
    if len(pk):
     j=max(pk,key=lambda x:dv[x]);w=peak_widths(dv,[j]);f.update(dva_peak_height_v_ah=float(dv[j]),dva_peak_location_ah=float(grid[j]),dva_peak_width_ah=float(w[0][0]*np.diff(grid)[0]),dva_curve_area_v=float(np.trapezoid(dv,grid)));q['dva_status']='exploratory_single_smoothing_not_selection_eligible'
 if 'temp' in c and c.temp.notna().all():f.update(charge_temperature_mean_c=float(c.temp.mean()),charge_temperature_max_c=float(c.temp.max()),charge_temperature_rise_c=float(c.temp.iloc[-1]-c.temp.iloc[0]))
 return f,q

def reconstruct(p,frames,observer=None):
 if not frames:return []
 d=pd.concat(frames,ignore_index=True);d=d.drop_duplicates(subset=[c for c in ['t','i','v','cycle','step'] if c in d]);d=d.sort_values('t',kind='stable').reset_index(drop=True)
 # Sign evidence comes from within-step voltage evolution, excluding gaps and transitions.
 dt=d.t.diff();dv=d.v.diff();same=(d.i*d.i.shift()>0)&(dt>0)&(dt<=120)&(d.i.abs()>.03)&(dv.abs()>.0001)
 evidence=np.sign(d.i[same])*np.sign(dv[same]);alignment=evidence.mean() if len(evidence)>=20 else 0
 sign=1 if alignment>.15 else -1 if alignment<-.15 else 0
 if not sign:return [dict(battery=battery_id(p),source_file=str(p),reason='ambiguous_current_polarity',label_valid=False)]
 state=np.where(d.i*sign>.03,1,np.where(d.i*sign<-.03,-1,0));bound=np.r_[True,(state[1:]!=state[:-1])|(np.diff(d.t)<=0)|(~interval_evidence(d)[1:])]
 blocks=[g for _,g in d.groupby(np.cumsum(bound),sort=False)];rows=[];last_charge=None;last_charge_end=None
 for bi,g in enumerate(blocks):
  mode=int(state[g.index[0]])
  if mode==1:
   # Link CC/CV phases only when intervening rows are verified near-zero-current rest.
   if last_charge is not None:
    between=d.loc[last_charge.index[-1]+1:g.index[0]-1]
    if (between.i.abs()<=.01).all() and interval_evidence(d.loc[last_charge.index[-1]:g.index[0]]).all():
     last_charge=pd.concat([last_charge,g])
    else:last_charge=g
   else:last_charge=g
   last_charge_end=bi
   continue
  if mode!=-1:continue
  t=g.t.to_numpy(float);v=g.v.to_numpy(float);ii=-sign*g.i.to_numpy(float);capacity=integral(t,ii)/3600
  c=last_charge if last_charge is not None else d.iloc[:0]
  reasons=[]
  if len(g)<12:reasons.append('insufficient_discharge_points')
  if len(g)<2 or not np.all((np.diff(t)>0)&(np.diff(t)<=120)):reasons.append('invalid_discharge_time')
  if not np.isfinite(v).all() or not np.isfinite(ii).all():reasons.append('invalid_discharge_measurements')
  if v[-1]>2.72:reasons.append('partial_discharge_or_missing_cutoff')
  full_charge=len(c)>=12 and c.v.iloc[-1]>=4.18 and abs(c.i.iloc[-1])<=.055
  if not full_charge:reasons.append('no_verified_full_charge_before_discharge')
  if capacity<=0 or capacity>1.65:reasons.append('implausible_integrated_capacity')
  gap=np.nan;agreement=np.nan;recorded=np.nan
  if 'qd' in g:
   a=g.qd.to_numpy(float);recorded=a[-1]-a[0]
   if np.isfinite(recorded) and recorded>0:
    agreement=abs(recorded-capacity)/max(capacity,1e-9)
    if agreement>.03:reasons.append('recorded_integrated_capacity_disagreement')
  if len(c):
   gap=t[0]-c.t.iloc[-1]
   mid=d.loc[c.index[-1]+1:g.index[0]-1]
   if len(mid) and not (mid.i.abs()<=.01).all():reasons.append('intervening_active_measurement')
   combined=d.loc[c.index[-1]:g.index[0]]
   if not interval_evidence(combined).all():reasons.append('unknown_charge_discharge_gap')
  if len(c):
   c=c.copy();c.attrs['linked_charge']=True
  features,qual=charge_features(c,sign)
  row=dict(battery=battery_id(p),source_file=str(p),source_sheet='|'.join(g.source_sheet.unique()),cycle_raw=str(g.cycle.iloc[0]) if 'cycle' in g else '',episode_in_file=len(rows)+1,discharge_start_s=float(t[0]),discharge_end_s=float(t[-1]),timestamp_start=str(g.stamp.iloc[0]),timestamp_end=str(g.stamp.iloc[-1]),charge_end_timestamp=str(c.stamp.iloc[-1]) if len(c) else '',current_sign=sign,polarity_alignment=float(alignment),discharge_start_v=float(v[0]),discharge_end_v=float(v[-1]),discharge_current_mean_a=float(ii.mean()),discharge_capacity_integrated_ah=capacity,recorded_capacity_delta_ah=recorded,capacity_relative_disagreement=agreement,label_valid=not reasons,soh=capacity/1.1 if not reasons else np.nan,reason=';'.join(reasons) if reasons else 'validated',capacity_label_confidence='high' if not reasons else 'unlabeled',cycle_confidence='source_local_episode',unknown_source_boundary=True,n_discharge_points=len(g),n_charge_points=len(c),verified_rest_duration_s=0.)
  if len(c) and last_charge_end is not None:
   rests=blocks[last_charge_end+1:bi]
   row['verified_rest_duration_s']=sum(float(np.ptp(r.t)) for r in rests if (r.i.abs()<=.01).all() and len(r)>1)
  row['within_source_continuity']=bool(interval_evidence(d.loc[blocks[bi-1].index[-1]:g.index[0]]).all()) if bi else False
  row['charge_start_s']=float(c.t.iloc[0]) if len(c) else np.nan
  row.update(features);row.update(qual)
  if observer is not None:observer(c,sign,row)
  rows.append(row)
  last_charge=None;last_charge_end=None
 return rows

def normalize_feature_schema(data):
 for old,new in {'throughput_3p8_4p0_ah':'throughput_bracket_3p8_4p0_ah','throughput_3p9_4p1_ah':'throughput_bracket_3p9_4p1_ah'}.items():
  if old in data:
   data[new]=data[new].combine_first(data[old]) if new in data else data[old]
   data=data.drop(columns=old)
 return data

def extract(limit=None,representative=False):
 OUT.mkdir(exist_ok=True);CACHE.mkdir(exist_ok=True)
 paths=discover()
 if representative:
  paths=[next(p for p in paths if battery_id(p)==b and p.suffix=='.xlsx') for b in ['CS2_33','CS2_34','CS2_3','CS2_5']]
 if limit:paths=paths[:limit]
 allrows=[];inventory=[]
 for p in tqdm(paths,desc='Raw files',mininterval=5):
  key=hashlib.sha256((str(p)+str(p.stat().st_size)+str(p.stat().st_mtime_ns)+json.dumps(CONFIG,sort_keys=True)+('cadex-clock-v1' if p.suffix=='.txt' else '')).encode()).hexdigest()[:24];cache=CACHE/(key+'.json')
  if cache.exists(): obj=json.loads(cache.read_text())
  else:
   try:
    frames,inv=read_workbook(p);rows=reconstruct(p,frames);obj=dict(rows=rows,inventory=inv)
   except Exception as e:
    logging.exception('File failed %s',p);obj=dict(rows=[],inventory=[dict(path=str(p),battery=battery_id(p),status='read_error',error=str(e))])
   cache.write_text(json.dumps(obj,default=str,allow_nan=True),encoding='utf8')
  allrows.extend(obj['rows']);inventory.extend(obj['inventory'])
 data=normalize_feature_schema(pd.DataFrame(allrows)).drop(columns=['charge_voltage_integral_vs'],errors='ignore');inv=pd.DataFrame(inventory)
 prefix='representative_' if representative else ''
 inv.to_csv(OUT/(prefix+'schema_inventory.csv'),index=False)
 if not data.empty:
  data['timestamp_start']=pd.to_datetime(data.timestamp_start,errors='coerce');data['timestamp_end']=pd.to_datetime(data.timestamp_end,errors='coerce')
  # Exact repeated discharges across source exports are deduplicated; never infer continuity from filename order.
  keys=['battery','timestamp_start','timestamp_end','discharge_capacity_integrated_ah']
  dup=data.duplicated(keys,keep='first')&data.timestamp_start.notna()
  data.loc[dup].to_csv(OUT/(prefix+'duplicate_episodes.csv'),index=False);data=data.loc[~dup].sort_values(['battery','timestamp_start','source_file','episode_in_file']).reset_index(drop=True)
  data['cycle_id']=[f'{b}:{j}' for j,b in enumerate(data.battery)]
  data['cycle_ordinal']=data.groupby('battery').cumcount()+1
  data.to_csv(OUT/(prefix+'cs2_cycle_features_all.csv'),index=False)
 print('FILES',len(paths),'BATTERIES',inv.battery.nunique(),'EPISODES',len(data),'LABELS',int(data.soh.notna().sum()) if 'soh' in data else 0,flush=True)
 return data,inv

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--representative',action='store_true');parser.add_argument('--limit',type=int);args=parser.parse_args()
 logging.basicConfig(filename=str(OUT/'pipeline.log'),level=logging.INFO)
 extract(args.limit,args.representative)



