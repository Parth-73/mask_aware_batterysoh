"""Training-only relevance, nested battery-held-out selection and forecasting."""
import json,warnings,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from scipy.stats import pearsonr,spearmanr,kendalltau
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from cs2_research_pipeline import OUT,CONFIG,charge_features,normalize_feature_schema
from threadpoolctl import threadpool_limits
threadpool_limits(limits=1)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore',category=RuntimeWarning)
SEED=42
FEATURES=[c for c in charge_features(pd.DataFrame(),1)[0] if c!='charge_voltage_integral_vs']
def save(df,name):df.to_csv(OUT/(name+'.csv'),index=False)
def eligible(d):
 return [c for c in FEATURES if c in d and not c.startswith('dva_') and d[c].notna().mean()>=.8 and d[c].nunique()>8 and sum(g[c].notna().sum()>=20 for _,g in d.groupby('battery'))>=3]
def phase_dependent_columns(selected):
 aggregates={'charge_duration_s','charge_throughput_ah','charge_energy_wh','charge_voltage_mean_v','charge_voltage_std_v','cv_duration_ratio','cv_throughput_ratio'}
 cc=[c for c in selected if c.startswith(('cc_','time_','throughput_','ica_','dva_')) or c=='charge_c_rate' or c in aggregates]
 cv=[c for c in selected if c.startswith('cv_') or c in aggregates]
 return cc,cv

def model(name='ridge'):
 return make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),StandardScaler(),Ridge(alpha=10) if name=='ridge' else HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=7,l2_regularization=5,random_state=SEED))
def metrics(y,p):return dict(rmse=float(np.sqrt(mean_squared_error(y,p))),mae=float(mean_absolute_error(y,p)),r2=float(r2_score(y,p)) if len(y)>1 else np.nan,n=len(y))
def relevance(d,cols):
 rows=[]
 for c in cols:
  g=d[[c,'soh','cycle_ordinal']].replace([np.inf,-np.inf],np.nan).dropna()
  if len(g)<12 or g[c].nunique()<3:continue
  pr=pearsonr(g[c],g.soh);sp=spearmanr(g[c],g.soh);ke=kendalltau(g[c],g.soh)
  # Residualize linear age within battery, avoiding between-cell age offsets.
  residuals=[]
  for _,bg in d.groupby('battery'):
   z=bg[[c,'soh','cycle_ordinal']].dropna()
   if len(z)>=12 and z[c].nunique()>3:
    age=np.c_[np.ones(len(z)),z.cycle_ordinal]
    residuals.append(np.c_[z[c]-age@np.linalg.lstsq(age,z[c],rcond=None)[0],z.soh-age@np.linalg.lstsq(age,z.soh,rcond=None)[0]])
  partial=np.nan
  if residuals:
   rr=np.concatenate(residuals);partial=pearsonr(rr[:,0],rr[:,1]).statistic
  mi=mutual_info_regression(g[[c]],g.soh,random_state=SEED,n_neighbors=5)[0]
  rows.append(dict(feature=c,n=len(g),pearson=pr.statistic,pearson_p=pr.pvalue,spearman=sp.statistic,spearman_p=sp.pvalue,kendall=ke.statistic,kendall_p=ke.pvalue,mutual_information=mi,partial_pearson_age=partial))
 return pd.DataFrame(rows)
def shortlist(d,cols,n=9):
 if not cols:return []
 rel=relevance(d,cols).set_index('feature');corr=d[cols].corr(method='spearman').abs().fillna(0)
 chosen=[];remaining=list(rel.index)
 while remaining and len(chosen)<n:
  def score(c):
   r=rel.loc[c];return (abs(r.spearman)+abs(r.kendall)+min(r.mutual_information,1))/3-.65*max([corr.loc[c,s] for s in chosen],default=0)
  c=max(remaining,key=score);chosen.append(c);remaining.remove(c)
 return chosen

def select(d,tag,experiments,stability_prior=None):
 cols=eligible(d);pool=shortlist(d,cols);chosen=[]
 bats=sorted(d.battery.unique())
 if len(bats)<3:return [],cols
 # The outer-training mRMR candidate search space is fixed; inner-fold eligibility and fitted transforms exclude that fold's held-out battery.
 folds=[]
 for b in bats:
  tr=d[d.battery!=b];va=d[d.battery==b]
  permitted=set(eligible(tr));folds.append((tr,va,permitted))
 for k in range(min(5,len(pool))):
  scores={}
  for c in pool:
   if c in chosen:continue
   if chosen and d[chosen+[c]].corr(method='spearman').loc[c,chosen].abs().max()>.98:continue
   subset=chosen+[c];errors=[]
   for tr,va,allowed in folds:
    if not set(subset).issubset(allowed):continue
    m=model();m.fit(tr[subset],tr.soh);errors.append(metrics(va.soh,m.predict(va[subset]))['rmse'])
   # Require every inner held-out battery to be evaluated for a comparable score.
   score=float(np.mean(errors)) if len(errors)==len(folds) else np.inf
   experiments.append(dict(experiment=tag,k=k+1,features='|'.join(subset),inner_battery_macro_rmse=score,inner_folds=len(errors)))
   scores[c]=score
  viable=[c for c in scores if np.isfinite(scores[c])]
  if not viable:
   # Expand the training-only shortlist if redundancy leaves fewer than five defensible choices.
   extras=[x for x in cols if x not in pool]
   if extras:
    pool.extend(extras)
    # Re-evaluate the same subset size using all eligible candidates; inner eligibility still applies.
    for c in extras:
     if chosen and d[chosen+[c]].corr(method='spearman').loc[c,chosen].abs().max()>.98:continue
     errors=[];subset=chosen+[c]
     for tr,va,allowed in folds:
      if not set(subset).issubset(set(eligible(tr))):continue
      m=model();m.fit(tr[subset],tr.soh);errors.append(metrics(va.soh,m.predict(va[subset]))['rmse'])
     if len(errors)==len(folds):
      scores[c]=float(np.mean(errors));experiments.append(dict(experiment=tag,k=k+1,features='|'.join(subset),inner_battery_macro_rmse=scores[c],inner_folds=len(errors)))
    viable=[c for c in scores if np.isfinite(scores[c])]
   if not viable:break
  best=min(scores[c] for c in viable)
  near=[c for c in viable if scores[c]<=best*1.01+1e-5]
  chosen.append(min(near,key=lambda c:(-(stability_prior or {}).get(c,0),scores[c])))
 return chosen,cols

def forecast_pairs(d,h):
 if h==0:return d.assign(target_soh=d.soh)
 pairs=[]
 # Only verified adjacent raw episode order in the same source; no bridging workbook boundaries.
 for _,g in d.dropna(subset=['episode_in_file']).groupby(['battery','source_file'],sort=False):
  g=g.sort_values('episode_in_file');lookup={int(r.episode_in_file):r for _,r in g.iterrows()}
  for j,r in lookup.items():
   seq=[lookup.get(j+k) for k in range(h+1)]
   if any(s is None for s in seq):continue
   good=True
   for a,b in zip(seq[:-1],seq[1:]):
    # Boundary includes observed rest samples but cannot be inferred solely from elapsed time.
    gap=b.get('charge_start_s',np.nan)-a.discharge_end_s
    try:
     jump=float(b.cycle_raw)-float(a.cycle_raw)
     if jump>1 or jump<0:good=False;break
    except (AttributeError,ValueError,TypeError):pass
    proof=b.get('prior_episode_continuity_verified',None)
    if not np.isfinite(gap) or gap<0 or (proof is not None and not bool(proof)) or (proof is None and gap>120):good=False;break
   if good and pd.notna(seq[-1].soh):
    out=r.to_dict();out['target_soh']=seq[-1].soh;out['target_cycle_id']=seq[-1].get('cycle_id','');out['target_episode_in_file']=seq[-1].episode_in_file;pairs.append(out)
 return pd.DataFrame(pairs) if pairs else pd.DataFrame(columns=list(d.columns)+['target_soh'])

def prepare_observations(data):
 data=normalize_feature_schema(data.copy())
 from cs2_continuity import audit_continuity
 contpath=OUT/'cs2_forecast_continuity.csv'
 cont=pd.read_csv(contpath) if contpath.exists() else audit_continuity(data)
 keys=['battery','source_file','episode_in_file'];extra=['prior_episode_continuity_verified','continuity_reason','observed_rest_samples_between_episodes']
 data=data.drop(columns=extra,errors='ignore').merge(cont[keys+extra],on=keys,how='left',validate='one_to_one')
 data['prior_episode_continuity_verified']=data.prior_episode_continuity_verified.isin([True,'True',1])
 data['missing_measurement_fraction']=np.nan;data['raw_cycle_index_numeric']=pd.to_numeric(data.cycle_raw,errors='coerce');data['raw_cycle_index_jump']=data.groupby(['battery','source_file']).raw_cycle_index_numeric.diff().gt(1);data[FEATURES]=data[FEATURES].replace([np.inf,-np.inf],np.nan)
 start=pd.to_datetime(data.timestamp_start,errors='coerce');end=pd.to_datetime(data.timestamp_end,errors='coerce')
 timing_invalid=(end<start)
 if timing_invalid.any():
  data.loc[timing_invalid,'soh']=np.nan;data.loc[timing_invalid,'label_valid']=False
  data.loc[timing_invalid,'reason']=data.loc[timing_invalid,'reason']+';invalid_absolute_timestamp_order'
 data['absolute_timestamp_order_valid']=~timing_invalid
 previous_end=data.groupby(['battery','source_file']).discharge_end_s.shift(1)
 gap=data.charge_start_s-previous_end
 data['history_gap_unknown']=~data.prior_episode_continuity_verified|data.raw_cycle_index_jump
 data['history_segment']=data.groupby(['battery','source_file']).history_gap_unknown.cumsum()
 save(data,'cs2_cycle_features_all')
 return data

def analyze(data,inv):
 data=prepare_observations(data)
 labels=data[data.soh.notna()].copy()
 save(labels[['cycle_id','battery','soh','discharge_capacity_integrated_ah','recorded_capacity_delta_ah','capacity_relative_disagreement','capacity_label_confidence','source_file']],'cs2_validated_soh_labels')
 save(data[['cycle_id','battery']].join(data[FEATURES].notna().astype(int)),'cs2_feature_availability')
 save(data[data.soh.isna()],'cs2_rejected_cycles')
 history=data[['cycle_id','battery','source_file','episode_in_file','cycle_ordinal','verified_rest_duration_s','unknown_source_boundary','history_gap_unknown','history_segment']].copy()
 groups=data.groupby(['battery','source_file','history_segment'])[FEATURES]
 lag1,lag2,lag3,lag4,lag5=[groups.shift(k) for k in range(1,6)]
 delta=(lag1-lag2).add_prefix('previous_delta_')
 slopes=(-2*lag5-lag4+lag2+2*lag1)/10
 slopes=slopes.where(lag1.notna()&lag2.notna()&lag3.notna()&lag4.notna()&lag5.notna()).add_prefix('prior_rolling_slope_')
 history=pd.concat([history,delta,slopes],axis=1)
 history['cumulative_prior_observed_charge_ah']=data.groupby(['battery','source_file']).charge_throughput_ah.transform(lambda x:x.fillna(0).cumsum().shift(1))
 history['elapsed_test_time_s']=data.discharge_start_s-data.groupby(['battery','source_file']).discharge_start_s.transform('min')
 save(history,'cs2_historical_features')
 source=data.groupby(['battery','source_file']).agg(start=('timestamp_start','min'),end=('timestamp_end','max'),episodes=('cycle_id','size')).reset_index().sort_values(['battery','start'])
 source['previous_source_end']=source.groupby('battery').end.shift(1)
 source['possible_overlap']=source.start<=source.previous_source_end
 source['unknown_gap_to_previous']=~source.possible_overlap & source.previous_source_end.notna()
 save(source,'source_boundary_audit')
 manual=inv[~inv.status.isin(['raw','raw_cadex_clock_verified','metadata','statistics_excluded'])].copy();manual['review_scope']='source'
 cycle_review=data[data.reason.str.contains('ambiguous|unknown|invalid|disagreement',case=False,na=False)][['cycle_id','battery','source_file','reason']].copy();cycle_review['review_scope']='episode'
 manual=pd.concat([manual,cycle_review],ignore_index=True);save(manual,'cs2_manual_review')
 summary=data.groupby('battery').agg(episodes=('cycle_id','size'),validated_labels=('soh','count'),mean_soh=('soh','mean'),sources=('source_file','nunique')).reindex(sorted(inv.battery.unique()),fill_value=0).reset_index();save(summary,'cs2_battery_summary')
 quality=[];perdist=[]
 for c in FEATURES:
  x=data[c].dropna();q1,q3=x.quantile([.25,.75]) if len(x) else (np.nan,np.nan);iq=q3-q1
  quality.append(dict(feature=c,valid_n=len(x),missing_percent=100*data[c].isna().mean(),mean=x.mean(),median=x.median(),std=x.std(),iqr=iq,extreme_fraction=float(((x<q1-3*iq)|(x>q3+3*iq)).mean()),supporting_batteries=data.loc[data[c].notna(),'battery'].nunique(),coverage_requirement='Observed continuous charging; CC/CV evidence for phase features; ICA dense monotone coverage',failure_reason='absent_phase_or_insufficient_coverage_or_unreliable_curve'))
  for b,g in data.groupby('battery'):perdist.append(dict(battery=b,feature=c,n=g[c].count(),mean=g[c].mean(),median=g[c].median(),std=g[c].std()))
 failures=[]
 for c in FEATURES:
  miss=data[data[c].isna()].copy()
  reason='ica_status' if c.startswith('ica_') else 'dva_status' if c.startswith('dva_') else 'charge_quality'
  for (b,status),g in miss.groupby(['battery',reason],dropna=False):failures.append(dict(feature=c,battery=b,failure_context=status,n_missing=len(g)))
 save(pd.DataFrame(failures),'feature_extraction_failure_counts')
 save(pd.DataFrame(quality),'feature_quality_report');save(pd.DataFrame(perdist),'per_battery_feature_distributions')
 bats=sorted(b for b,g in labels.groupby('battery') if len(g)>=30)
 # Fixed split by cell identifier before feature relevance inspection.
 trainb=bats[:-4] if len(bats)>=9 else bats;valb=bats[-4:-2] if len(bats)>=9 else [];testb=bats[-2:] if len(bats)>=9 else []
 train=labels[labels.battery.isin(trainb)];val=labels[labels.battery.isin(valb)];test=labels[labels.battery.isin(testb)]
 # Battery split is fixed here; export the five-feature tables only after training-only selection.
 cols=eligible(train);rel=relevance(train,cols);save(rel,'pearson_spearman_kendall');save(rel[['feature','mutual_information','n']] if len(rel) else pd.DataFrame(columns=['feature','mutual_information','n']),'mutual_information')
 pc=[]
 for b,g in train.groupby('battery'):
  r=relevance(g,cols)
  if len(r):r['battery']=b;pc.append(r)
 pc=pd.concat(pc,ignore_index=True) if pc else pd.DataFrame();save(pc,'per_battery_feature_correlations')
 if len(pc):
  stability=pc.groupby('feature').agg(mean_spearman=('spearman','mean'),std_spearman=('spearman','std'),n_batteries=('battery','nunique'),direction_consistency=('spearman',lambda x:max((x>0).mean(),(x<0).mean())));save(stability,'per_battery_correlation_stability')
 reds=[]
 for method in ['pearson','spearman']:
  corr=train[cols].corr(method=method);corr.index.name='feature';long=corr.reset_index().melt(id_vars='feature',var_name='other',value_name='correlation');long['method']=method;reds.append(long)
 save(pd.concat(reds),'feature_redundancy_matrix')
 experiments=[];foldselections=[];results=[]
 selection_pairs=forecast_pairs(data,1)
 selection_pairs=selection_pairs[selection_pairs.battery.isin(trainb)].copy()
 selection_task='one_cycle_forecasting'
 if len(selection_pairs)>=100 and selection_pairs.battery.nunique()>=4:
  selection_data=selection_pairs.copy();selection_data['origin_soh']=selection_data.soh;selection_data['soh']=selection_data.target_soh
 else:
  selection_data=train;selection_task='current_cycle_estimation_fallback_insufficient_forecast_pairs'
 print('SELECTION TASK',selection_task,'ROWS',len(selection_data),flush=True)
 for b in trainb:
  tr=train[train.battery!=b];te=train[train.battery==b];selected,allcols=select(selection_data[selection_data.battery!=b],'outer_'+b,experiments)
  foldselections.extend(dict(heldout_battery=b,feature=c) for c in selected)
  print('OUTER',b,'selected',selected,flush=True)
  for subset,features in [('selected',selected),('all_eligible',allcols),('charging_baseline',['charge_duration_s','charge_throughput_ah'])]:
   if not features:continue
   for kind in ['ridge','hgb']:
    m=model(kind);m.fit(tr[features],tr.soh);results.append(dict(task='estimation',horizon=0,battery=b,model=kind,subset=subset,**metrics(te.soh,m.predict(te[features]))))
 selected,_=select(selection_data,'final_training_only',experiments,Counter(r['feature'] for r in foldselections))
 save(pd.DataFrame(experiments),'feature_subset_experiments');save(pd.DataFrame(foldselections),'fold_selected_features')
 counts=Counter(r['feature'] for r in foldselections);freq=pd.DataFrame([dict(feature=c,selected_count=counts[c],outer_folds=len(trainb),frequency=counts[c]/len(trainb)) for c in cols]);save(freq,'feature_selection_stability');save(freq,'feature_importance')
 # Multiplicity-adjusted descriptive p-values remain subject to serial-dependence caveats.
 if len(rel):
  from scipy.stats import false_discovery_control
  for stat in ['pearson','spearman','kendall']:
   pp=rel[stat+'_p'];ok=pp.notna();rel.loc[ok,stat+'_bh_q']=false_discovery_control(pp[ok].to_numpy())
  save(rel,'pearson_spearman_kendall')
 top=pd.DataFrame([dict(rank=k+1,feature=c,selection='training-only mRMR shortlist + inner battery-held-out sequential forward selection') for k,c in enumerate(selected)]);save(top,'top_5_features')
 exportcols=['cycle_id','battery','source_file','source_sheet','episode_in_file','soh']+selected
 save(train[exportcols],'train');save(val[exportcols],'validation');save(test[exportcols],'test')
 if selected:
  imp=SimpleImputer(strategy='median',keep_empty_features=True);xt=imp.fit_transform(train[selected]);sc=StandardScaler().fit(xt)
  (OUT/'selected_feature_scaler.json').write_text(json.dumps(dict(feature_order=selected,training_batteries=trainb,median_imputation=imp.statistics_.tolist(),mean=sc.mean_.tolist(),scale=sc.scale_.tolist(),notes='Fit on training cells only; raw physical-unit CSVs retain missing values and availability masks are separate'),indent=2))
 # Test labels are only accessed after freezing training features; report per-battery test metrics.
 for b in testb+valb:
  te=labels[labels.battery==b]
  for kind in ['ridge','hgb']:
   for subset,features in [('selected',selected),('all_eligible',cols),('charging_baseline',['charge_duration_s','charge_throughput_ah'])]:
    if not features:continue
    m=model(kind);m.fit(train[features],train.soh);results.append(dict(task='estimation',horizon=0,battery=b,model=kind,subset=subset,split='test' if b in testb else 'validation',**metrics(te.soh,m.predict(te[features]))))
 save(pd.DataFrame(results),'estimation_results')
 importance=[]
 if selected and len(val):
  from sklearn.inspection import permutation_importance
  for kind in ['ridge','hgb']:
   m=model(kind);m.fit(train[selected],train.soh)
   for b,vg in val.groupby('battery'):
    imp=permutation_importance(m,vg[selected],vg.soh,scoring='neg_root_mean_squared_error',random_state=SEED,n_repeats=5)
    importance.extend(dict(feature=c,model=kind,battery=b,rmse_increase_mean=imp.importances_mean[j],rmse_increase_std=imp.importances_std[j]) for j,c in enumerate(selected))
 save(pd.DataFrame(importance,columns=['feature','model','battery','rmse_increase_mean','rmse_increase_std']),'feature_importance')
 forecasting=[];robust=[];paircounts=[]
 for h in [1,3,5]:
  pairs=forecast_pairs(data,h)
  if len(pairs):
   pairs['split']=np.select([pairs.battery.isin(trainb),pairs.battery.isin(valb),pairs.battery.isin(testb)],['train','validation','test'],default='insufficient_labeled_cell_support')
   export=[c for c in ['cycle_id','target_cycle_id','battery','source_file','episode_in_file','target_episode_in_file','split','soh','target_soh']+selected if c in pairs]
   save(pairs[export],'forecast_pairs_h'+str(h))
  else:save(pd.DataFrame(columns=['cycle_id','target_cycle_id','battery','split','target_soh']+selected),'forecast_pairs_h'+str(h))
  paircounts.append(dict(horizon=h,n_pairs=len(pairs),batteries=pairs.battery.nunique()))
  for b in testb+trainb:
   fitb=[x for x in trainb if x!=b]
   chosen_b=selected if b in testb else [r['feature'] for r in foldselections if r['heldout_battery']==b]
   tr=pairs[pairs.battery.isin(fitb)].dropna(subset=['target_soh']);te=pairs[pairs.battery==b].dropna(subset=['target_soh'])
   if len(tr)<30 or len(te)<5 or not chosen_b:continue
   for kind in ['ridge','hgb']:
    for subset,features in [('selected',chosen_b),('all_eligible',eligible(tr)),('charging_baseline',['charge_duration_s','charge_throughput_ah'])]:
     if not features:continue
     m=model(kind);m.fit(tr[features],tr.target_soh);forecasting.append(dict(task='forecasting',horizon=h,battery=b,split='test' if b in testb else 'outer_cv',model=kind,subset=subset,**metrics(te.target_soh,m.predict(te[features]))))
    if b not in testb:continue
    # Train missingness-aware baselines using shared deterministic augmentations.
    rng=np.random.default_rng(SEED);xx=tr[selected].copy();aug=xx.copy();aug[rng.random(aug.shape)<.25]=np.nan
    cc_aug=xx.copy();cc_aug.loc[:,phase_dependent_columns(selected)[0]]=np.nan
    cv_aug=xx.copy();cv_aug.loc[:,phase_dependent_columns(selected)[1]]=np.nan
    all_aug=xx.copy();all_aug.iloc[len(all_aug)//3:len(all_aug)//3+max(1,len(all_aug)//5),:]=np.nan
    m=model(kind);m.fit(pd.concat([xx,aug,cc_aug,cv_aug,all_aug]),pd.concat([tr.target_soh]*5))
    rng=np.random.default_rng(SEED+h);mask=rng.random((len(te),len(selected)))<.25
    for scenario in ['complete','missing_cc','missing_cv','random_missing','consecutive_missing_origins']:
     tx=te[selected].copy()
     if scenario=='missing_cc':tx.loc[:,phase_dependent_columns(selected)[0]]=np.nan
     if scenario=='missing_cv':tx.loc[:,phase_dependent_columns(selected)[1]]=np.nan
     if scenario=='random_missing':tx[mask]=np.nan
     if scenario=='consecutive_missing_origins':tx.iloc[len(tx)//3:len(tx)//3+max(1,len(tx)//5),:]=np.nan
     robust.append(dict(horizon=h,battery=b,model=kind,scenario=scenario,**metrics(te.target_soh,m.predict(tx))))
   pers=te.dropna(subset=['soh'])
   if len(pers):forecasting.append(dict(task='forecasting',horizon=h,battery=b,split='test' if b in testb else 'outer_cv',model='persistence_origin_soh',subset='origin_soh_available_after_discharge',**metrics(pers.target_soh,pers.soh)))
 save(pd.DataFrame(paircounts),'forecast_pair_counts');save(pd.DataFrame(forecasting,columns=['task','horizon','battery','split','model','subset','rmse','mae','r2','n']),'forecasting_results');save(pd.DataFrame(robust,columns=['horizon','battery','model','scenario','rmse','mae','r2','n']),'robustness_results')
 if results:
  res=pd.DataFrame(results);res['split']=res.get('split',pd.Series(index=res.index,dtype=str)).fillna('outer_cv');macro=res.groupby(['task','horizon','split','model','subset'])[['rmse','mae','r2']].mean().reset_index();save(macro,'estimation_battery_macro')
 if forecasting:save(pd.DataFrame(forecasting).groupby(['horizon','split','model','subset'])[['rmse','mae','r2']].mean().reset_index(),'forecasting_battery_macro')
 else:save(pd.DataFrame(columns=['horizon','split','model','subset','rmse','mae','r2']),'forecasting_battery_macro')
 if robust:save(pd.DataFrame(robust).groupby(['horizon','model','scenario'])[['rmse','mae','r2']].mean().reset_index(),'robustness_battery_macro')
 else:save(pd.DataFrame(columns=['horizon','model','scenario','rmse','mae','r2']),'robustness_battery_macro')
 (OUT/'robustness_mask_definition.json').write_text(json.dumps(dict(missing_cc=phase_dependent_columns(selected)[0],missing_cv=phase_dependent_columns(selected)[1],random_feature_probability=.25,consecutive_missing_fraction=.2,evaluation_seed='42 + horizon',training='complete + random + CC + CV + contiguous all-feature missing augmentation; identical across models',interpretation='Feature group masking; physical partial curves evaluated separately'),indent=2))
 manifest=dict(implementation_sha256={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in ['cs2_research_pipeline.py','cs2_research_analysis.py','cs2_continuity.py']},status='executed_conservative_pipeline',raw_files=inv.path.nunique(),processed_batteries=int(inv.battery.nunique()),batteries_with_extracted_episodes=int(data.battery.nunique()),validated_soh_labels=len(labels),extracted_features=len(FEATURES),features_with_any_observed_value=int(data[FEATURES].notna().any().sum()),eligible_features=cols,selected_features=selected,input_columns=selected,estimation_target='soh',forecasting_target='target_soh',selection_task=selection_task,selection_training_rows=len(selection_data),train_batteries=trainb,validation_batteries=valb,test_batteries=testb,outer_cv=[dict(train=[x for x in trainb if x!=b],heldout=b) for b in trainb],limitations=['CADEX uses independently cross-checked Duration seconds; absolute timestamps unavailable','Source boundaries not merged without verified experiment continuity','SOH depends on discharge rate; different protocols not normalized','Forecasting restricted to observed same-file adjacency','DVA exploratory, excluded from selection','No Transformer, PINN, GWO or AIDO neural architecture trained','Feature masking does not establish physical acquisition invariance'])
 (OUT/'dataset_manifest.json').write_text(json.dumps(manifest,indent=2));(OUT/'preprocessing_config.json').write_text(json.dumps(dict(**CONFIG,continuity_audit_version=1,cadex_clock_ratio=60,cadex_max_clock_residual_s=5,minimum_feature_availability=.8,minimum_supporting_batteries=3,minimum_valid_per_supporting_battery=20,minimum_unique_values=9,mrmr_redundancy_penalty=.65,shortlist_size=9,maximum_selected_redundancy=.98,ridge_alpha=10,hgb=dict(max_iter=100,max_leaf_nodes=7,l2_regularization=5),forecast_horizons=[1,3,5]),indent=2))
 plots(data,train,cols,selected,freq)
 report(data,inv,summary,rel,selected,manifest,results,forecasting)
 return manifest

def plots(data,train,cols,selected,freq):
 fig,ax=plt.subplots(figsize=(11,5))
 for b,g in data.groupby('battery'):ax.plot(g.cycle_ordinal,g.soh,'.',ms=2,label=b)
 ax.set(xlabel='Observed episode ordinal (not absolute cycle age)',ylabel='Validated SOH');ax.legend(fontsize=7,ncol=3);fig.tight_layout();fig.savefig(OUT/'soh_trajectories.png',dpi=150);plt.close(fig)
 if cols:
  corr=train[cols].corr(method='spearman');fig,ax=plt.subplots(figsize=(12,10));im=ax.imshow(corr,vmin=-1,vmax=1,cmap='coolwarm');ax.set_xticks(range(len(cols)),cols,rotation=90,fontsize=6);ax.set_yticks(range(len(cols)),cols,fontsize=6);fig.colorbar(im);fig.tight_layout();fig.savefig(OUT/'feature_correlation_heatmap.png',dpi=150);plt.close(fig)
 available=[c for c in FEATURES if data[c].notna().any()]
 if available:
  fig,axes=plt.subplots(int(np.ceil(len(available)/4)),4,figsize=(16,3*int(np.ceil(len(available)/4))))
  for ax,c in zip(axes.ravel(),available):ax.hist(data[c].dropna(),bins=35);ax.set_title(c,fontsize=7)
  for ax in axes.ravel()[len(available):]:ax.axis('off')
  fig.tight_layout();fig.savefig(OUT/'all_feature_distributions.png',dpi=120);plt.close(fig)
 if selected:
  fig,axes=plt.subplots(2,len(selected),figsize=(4*len(selected),7),squeeze=False)
  for j,c in enumerate(selected):
   axes[0,j].hist(data[c].dropna(),bins=40);axes[0,j].set_title(c,fontsize=8)
   for b,g in train.groupby('battery'):axes[1,j].scatter(g[c],g.soh,s=3,label=b)
   axes[1,j].set_ylabel('SOH')
  fig.tight_layout();fig.savefig(OUT/'selected_feature_distributions_and_scatter.png',dpi=150);plt.close(fig)
 if len(freq):
  f=freq.sort_values('frequency');fig,ax=plt.subplots(figsize=(9,7));ax.barh(f.feature,f.frequency);ax.set_xlabel('Selection frequency across outer held-out batteries');fig.tight_layout();fig.savefig(OUT/'feature_selection_stability.png',dpi=150);plt.close(fig)

def report(data,inv,summary,rel,selected,manifest,results,forecasting):
 def table(df):return df.to_string(index=False)
 strongest=rel.reindex(rel.pearson.abs().sort_values(ascending=False).index).head(5) if len(rel) else rel
 nonlinear=rel.sort_values('mutual_information',ascending=False).head(5) if len(rel) else rel
 report=f'''# CALCE CS2 feature research â€” executed results

Processed {manifest['raw_files']} raw files across {manifest['processed_batteries']} discovered batteries. Extracted {len(data)} discharge episodes, {len(data[data.soh.notna()])} validated labels and {len(FEATURES)} charging feature definitions. Exactly {len(selected)} features selected.

## 1. Valid labels per battery
```
{table(summary)}
```
## 2. Previous pipeline failures
The previous battery parser extracted the first digit run in `CS2_33`, which is `2`, assigning all cells CS2_2. It grouped a full workbook by exported Cycle_Index, even when one index contained repeated operating episodes. It evaluated the fifth percentile of discharge voltage against 2.85 V rather than the observed terminal cutoff: a steep low-voltage tail can be complete while its fifth percentile remains above 2.85 V. It split raw sheets and did not account for cumulative capacity counters. The new pipeline reconstructs contiguous current episodes across channel sheets and preserves partial episodes. These mechanisms are verified from code and raw examples; their individual contribution to all old rejections has not been causally quantified.

## 3. Strongest linear/rank associations (training cells only)
```
{table(strongest)}
```
Full Pearson and Spearman rankings are available in pearson_spearman_kendall.csv.

## 4. Nonlinear associations
```
{table(nonlinear)}
```
Mutual information is an exploratory dependence estimate, not proof of a nonlinear physical mechanism. Age-residual associations are also reported.

## 5. Redundancy
See Pearson and Spearman matrices in feature_redundancy_matrix.csv. Throughput, duration and energy are physically related; mRMR penalizes their redundancy before sequential selection.

## 6. Unseen-cell predictivity
See estimation_results.csv and estimation_battery_macro.csv. Outer folds redo selection on training cells only. Inner folds fit imputation and scaling on their training cells. Selection optimizes the one-cycle forecasting target where sufficient pairs exist and is a Ridge surrogate; HGB is an independent predictive comparison. Metrics use SOH fractions, so multiply errors by 100 for percentage points. Battery-macro means give each cell equal weight.

## 7â€“8. Final selected features and rationale
{', '.join(selected) or 'No defensible features selected'}.
Selection requires >=80% training availability, variation and >=3 supported cells; DVA is excluded as insufficiently validated. A relevance-versus-redundancy shortlist is followed by sequential forward selection to five using inner battery-macro RMSE. Selection task: {manifest['selection_task']}. Features with >0.98 absolute Spearman redundancy to an already selected feature are excluded. The unvalidated voltage-time-integral implementation is excluded from published features. Among candidates within 1% (+0.00001 SOH) of the best inner score, final selection prefers the higher outer-fold selection frequency. Thus selection does not follow the highest pooled correlation. See feature_subset_experiments.csv for every executed subset and feature_selection_stability.csv for outer-fold frequency. Availability and feature stability still require external experimental replication.

## 9. Five versus all eligible features
Estimation battery-macro results (SOH fraction units):
```
{table(pd.read_csv(OUT/'estimation_battery_macro.csv')) if (OUT/'estimation_battery_macro.csv').exists() else 'No supported estimation experiment.'}
```
See separate selected/all_eligible/charging_baseline rows in estimation_battery_macro.csv and forecasting_results.csv. No pooled-cycle score is presented as independent-cell evidence.

## 10. Forecasting and missing acquisition
Executed forecasting battery-macro results:
```
{table(pd.read_csv(OUT/'forecasting_battery_macro.csv'))}
```
Executed feature-masking battery-macro results:
```
{table(pd.read_csv(OUT/'robustness_battery_macro.csv'))}
```
Forecasting uses horizons 1/3/5 only across verified same-file episode adjacency. See forecast_pair_counts.csv and forecasting_results.csv. Missingness experiments use identical seeded masks for Ridge and HGB and augmentation during training. Missing historical observations is represented as contiguous missing origin features, not a trained sequence model. Physical partial-curve diagnostics are tracked separately in physical_partial_observation_manifest.json when executed. Feature masking alone cannot support physical acquisition-invariance claims.

## 11. Robust findings and limitations
The identifier bug, raw-sheet distinctions and episode-level reconstruction are supported directly. All selected features precede the associated discharge; no target discharge measurement is a predictor. Same-cycle estimation is separated from forecasting. Labels retain genuine fluctuations and are not smoothed or interpolated. Rates vary by protocol; SOH here is protocol-conditioned delivered capacity, not rate-independent intrinsic capacity. Source-boundary merging, CADEX absolute chronology and status-code semantics, comprehensive physical partial-observation validation and full architecture training remain unresolved. Exploratory p-values treat cycles as independent and must not be interpreted confirmatorily; serial dependence and multiple testing inflate evidence. No multiplicity-adjusted inferential claim is made. See the audit and protocol.

Metadata verified from [CALCE](https://calce.umd.edu/battery-data): nominal 1100 mAh; standard 4.2 V CC/CV charging with 0.05 A termination; default 2.7 V discharge cutoff. Cells have full, variable-rate and partial-cycling protocols. Retrieved 2026-10-31.
'''
 (OUT/'FEATURE_ENGINEERING_REPORT.md').write_text(report,encoding='utf8')
 audit=f'''# Data quality audit

## Label validation reasons
```
{data.reason.value_counts().to_string()}
```

## Source inventory
```
{inv.status.value_counts().to_string()}
```

Discharge completeness requires preceding observed charge ending >=4.18 V and <=0.055 A, an observed discharge endpoint <=2.72 V, >=12 points, finite monotone discharge time with no >120 s interval, a plausible positive integrated capacity <=1.65 Ah and <=3% disagreement with the recorded counter increment where available. Voltage/current tolerances accommodate instrument resolution; the previous threshold was not relaxed to increase counts. Sparse charging intervals require matching operating-step clocks and increasing charge counters. Their throughput uses recorded counter increments; observed rests separate CC and CV and are excluded from active charging throughput. No discharge threshold is relaxed. The first unobserved fraction of each step is not integrated. Recorded counters are differenced per episode, never treated as per-cycle absolute capacity. Unusual SOH is retained when measurement checks pass.

Raw channel sheets are concatenated within each workbook, sorted by test time, and duplicate measurement tuples removed. Identical discharge timestamp/capacity signatures across files are excluded and exported. Other overlapping source intervals remain a manual-review concern. No cross-workbook fragment is joined. This loses some legitimate boundary cycles conservatively. Cycle ordinals measure observed episodes, not verified absolute cycle age. No missing interval is called rest; rest durations sum only near-zero-current blocks supported by observed samples and matching step clocks. CADEX clock units are accepted only when Duration versus Time fits a 60:1 slope within 0.1 and maximum residual below 5 seconds, with monotone Duration. Current/voltage use explicit mA/mV headers. Capacity is percentage-like and never treated as Ah. Absolute CADEX timestamps remain unknown; accepted capacity integrates measured current against the validated seconds clock.
'''
 (OUT/'DATA_QUALITY_AUDIT.md').write_text(audit,encoding='utf8')
 (OUT/'EXPERIMENTAL_PROTOCOL.md').write_text('''# Experimental protocol

Seed 42. All raw files and old preprocessing outputs remain unchanged. Run the representative gate before the complete extraction. Checkpoints are keyed by source path, size, modification time and extraction configuration. New extraction logic must increment the configuration version to invalidate checkpoints.

Prediction origin is the end of charging for estimation, and the end of the origin discharge for forecasting/persistence comparisons. Feature candidates use charging samples only. Target-current, target-capacity, discharge-derived proxies and retrospective quantities are excluded from the primary features. Historical variables and observation masks remain separate.

Identifiers define battery-disjoint training/validation/test sets before computing associations. Training outer leave-one-battery-out folds redo feature selection. Inner folds independently check eligibility, fit imputation/scaling, and score SFS subsets. No outer-held-out or test labels select features. mRMR shortlist is bounded at nine for runtime. Ridge drives SFS; HGB compares selected/all features. MLP and exhaustive subset search are not executed. Within-cell cycle samples are serially dependent; p-values are descriptive. No neural architecture performance is claimed. Electrical resistance proxies are omitted because transition timing has not been validated as instantaneous; temperature-adjusted indicators are omitted without a validated compensation model. DVA features remain exploratory. Historical features are exported separately and excluded from the main five.

Forecast windows never cross files, missing episodes or unknown intervals. Episode adjacency requires raw-sample endpoint matching, intervening near-zero-current samples, and consistent operating-step clocks. Long gaps are accepted only when these measurements verify rest; elapsed time alone is insufficient. SOH fractions are the metric unit. Reporting uses per-battery metrics and battery-macro averages. No SOH interpolation or smoothing. Different discharge rates remain a material label confounder.

CADEX absolute chronology, cross-source experiment linkage and comprehensive physical partial-curve robustness remain follow-up work. These outputs are an audited research baseline, not a claim of full acquisition invariance or publication-ready Transformer validation.
''',encoding='utf8')
 (OUT/'top_5_features_justification.md').write_text('# Selected charging features\n\n'+'\n'.join(f'{i+1}. `{c}` â€” available before target discharge; passed availability/coverage checks and added by battery-held-out SFS after mRMR screening.' for i,c in enumerate(selected))+'\n\nInspect subset RMSE and outer-fold frequencies before interpreting any feature as universal. Detailed selection is in FEATURE_ENGINEERING_REPORT.md.',encoding='utf8')

if __name__=='__main__':
 data=pd.read_csv(OUT/'cs2_cycle_features_all.csv',low_memory=False);inv=pd.read_csv(OUT/'schema_inventory.csv');print(json.dumps(analyze(data,inv),indent=2))


