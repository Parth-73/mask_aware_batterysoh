"""Validate final artifacts and append measured diagnostic summaries."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from cs2_research_pipeline import OUT,source_snapshot_sha256
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def finalize():
 manifest=json.loads((OUT/'dataset_manifest.json').read_text());d=pd.read_csv(OUT/'cs2_cycle_features_all.csv',low_memory=False);v=d[d.soh.notna()];selected=manifest['selected_features']
 assert len(v)==manifest['validated_soh_labels']
 assert v.discharge_end_v.le(2.72).all()
 assert v.capacity_relative_disagreement.dropna().le(.03).all()
 assert set(manifest['train_batteries']).isdisjoint(manifest['test_batteries'])
 assert set(manifest['train_batteries']).isdisjoint(manifest['validation_batteries'])
 assert set(manifest['validation_batteries']).isdisjoint(manifest['test_batteries'])
 for split in ['train','validation','test']:
  x=pd.read_csv(OUT/(split+'.csv'))
  assert all(c in x for c in selected)
  assert 'discharge_capacity_integrated_ah' not in x
 for h in [1,3,5]:
  pairs=pd.read_csv(OUT/f'forecast_pairs_h{h}.csv')
  if len(pairs):
   assert (pairs.target_episode_in_file-pairs.episode_in_file).eq(h).all()
   tp=pairs[pairs.split=='train']
   if len(tp):
    imp=SimpleImputer(strategy='median',keep_empty_features=True);values=imp.fit_transform(tp[selected]);sc=StandardScaler().fit(values)
    (OUT/f'selected_feature_scaler_h{h}.json').write_text(json.dumps(dict(feature_order=selected,horizon=h,training_batteries=sorted(tp.battery.unique()),n_training_origins=len(tp),median_imputation=imp.statistics_.tolist(),mean=sc.mean_.tolist(),scale=sc.scale_.tolist()),indent=2))
 extreme=v[(v.soh<.2)|(v.soh>1.2)].copy();extreme['review_scope']='extreme_capacity_measurement';extreme['review_note']='Kept as measured: inspect raw source and operating rate; statistical extremeness alone is not a rejection reason.';extreme.to_csv(OUT/'cs2_extreme_capacity_review.csv',index=False)
 inv=pd.read_csv(OUT/'schema_inventory.csv');source=inv[~inv.status.isin(['raw','raw_cadex_clock_verified','metadata','statistics_excluded'])].copy();source['review_scope']='source'
 cyc=d[d.reason.str.contains('ambiguous|unknown|invalid|disagreement',case=False,na=False)][['cycle_id','battery','source_file','reason']].copy();cyc['review_scope']='episode'
 pd.concat([source,cyc,extreme[['cycle_id','battery','source_file','reason','review_scope','review_note']]],ignore_index=True).to_csv(OUT/'cs2_manual_review.csv',index=False)
 summary=dict(validated_labels=len(v),unique_batteries=d.battery.nunique(),selected_features=selected,all_label_endpoints_pass=True,all_available_counter_crosschecks_pass=True,disjoint_splits=True,forecast_horizon_indices_pass=True,retained_extreme_soh_labels=len(extreme),unresolved_cadex_files=int((inv.status=='manual_review_cadex_clock').sum()),read_error_files=int(inv.loc[inv.status=='read_error','path'].nunique()))
 (OUT/'artifact_validation.json').write_text(json.dumps(summary,indent=2))
 from cs2_statistics import write_full_statistics
 rel,freq=write_full_statistics(d,manifest);train=pd.read_csv(OUT/'train.csv')
 importance=pd.read_csv(OUT/'feature_importance.csv');importance['task']='current_cycle_estimation_validation_cells';importance.to_csv(OUT/'feature_importance.csv',index=False)
 corr=train[selected].corr(method='spearman');corr.to_csv(OUT/'selected_feature_redundancy_spearman.csv')
 descriptions={'throughput_bracket_3p9_4p1_ah':'Measured charge over sample intervals bracketing 3.9–4.1 V. This is a sample-edge approximation, not exactly interpolated threshold-to-threshold charge.','throughput_bracket_3p8_4p0_ah':'Measured charge over sample intervals bracketing 3.8–4.0 V; endpoint approximation depends on sampling.','charge_duration_s':'Active charging duration, excluding measured rest between charging phases.','charge_throughput_ah':'Measured charging throughput; sparse intervals use validated cumulative charging counters.','charge_energy_wh':'Charging energy computed from observed current/voltage and supported throughput intervals.','charge_voltage_mean_v':'Mean observed charging voltage; depends on sampling and operating protocol.','charge_voltage_std_v':'Dispersion of observed charging voltage; summarizes charging-curve shape.','cc_duration_s':'Duration of the identified stable-current charging region.','cc_throughput_ah':'Charge supplied in the identified constant-current region.','cc_current_mean_a':'Mean CC operating current; may encode protocol rather than intrinsic health.','cc_current_std_a':'CC current variability; may include instrument and control-loop effects.','cc_initial_voltage_v':'Voltage at the start of observed CC charging; affected by prior operating conditions.','cc_final_voltage_v':'Voltage at the end of the identified CC region.','cc_voltage_slope_v_s':'Linear slope of voltage during CC charging.','cc_voltage_increase_v':'Observed CC voltage rise.','cv_duration_s':'Duration of the voltage plateau with observed current decay.','cv_throughput_ah':'Charge transferred during verified CV charging.','cv_current_mean_a':'Mean current during verified CV charging.','cv_current_decay_a_s':'Observed linear CV current decay rate.','cv_duration_ratio':'CV duration divided by observed active charging duration.','cv_throughput_ratio':'CV throughput divided by observed charging throughput.','cv_decay_tau_s':'CV exponential decay time constant retained only for a sufficiently good log-current fit.','charge_c_rate':'CC charging current divided by nominal capacity; an operating-condition feature.'}
 text='# Five provisional features: measured evidence and physical interpretation\n\nThe selected subset did not beat the simple charging or persistence baseline on the untouched test cells. Selection is provisional; cross-protocol generalization is not established.\n\n'
 experiments=pd.read_csv(OUT/'feature_subset_experiments.csv')
 for j,c in enumerate(selected):
  rr=rel[rel.feature==c];ff=freq[freq.feature==c]
  desc=descriptions.get(c,'Observed fixed-voltage-window or derivative feature; coverage requirements and reliability flags are exported separately.')
  text+=f'## {j+1}. `{c}`\n\n{desc}\n\n'
  if len(rr):text+=f"Training current-cycle Pearson {rr.pearson.iloc[0]:.4f}, Spearman {rr.spearman.iloc[0]:.4f}, mutual information {rr.mutual_information.iloc[0]:.4f}. "
  if len(ff):text+=f"Selected in {ff.selected_count.iloc[0]:.0f}/{ff.outer_folds.iloc[0]:.0f} outer folds. "
  path='|'.join(selected[:j+1]);step=experiments[(experiments.experiment=='final_training_only')&(experiments.features==path)]
  if len(step):text+=f"Inner battery-macro RMSE for this prefix: {step.inner_battery_macro_rmse.iloc[0]:.5f}. "
  text+='Added using inner held-out-cell predictive error, with redundancy screening; this is an empirical selection, not proof of a unique physical mechanism.\n\n'
 toppearson=rel.loc[rel.pearson.abs().nlargest(5).index,'feature'].tolist();topspearman=rel.loc[rel.spearman.abs().nlargest(5).index,'feature'].tolist()
 text+=f"Highest absolute Pearson five: {', '.join(toppearson)}.\n\nHighest absolute Spearman five: {', '.join(topspearman)}.\n\nFinal features optimize {manifest['selection_task']}; pooled correlation alone does not measure cross-cell error or incremental value. The complete subset search log, outer selection frequencies and selected-feature redundancy matrix provide the numerical basis.\n"
 (OUT/'top_5_features_justification.md').write_text(text,encoding='utf8')
 audit='\n\n## Final artifact validation and extreme-value investigation\n\n'+json.dumps(summary,indent=2)+'\n\n'
 audit+='Voltage-window throughput features use measured intervals bracketing the requested voltage endpoints, hence their explicit `throughput_bracket_` names. They include finite sample-edge contributions outside the exact thresholds. Crossing-time features interpolate the threshold crossing times. This distinction matters for acquisition robustness.\n\n'
 audit+='Very low measured SOH values are retained, not clipped. Their source records reach the required voltage cutoff after observed full-charge termination and pass available counter/integration checks. These checks establish delivered capacity under the recorded operating rate, not rate-independent intrinsic health. Low-capacity cycles therefore require physical review before publication; see cs2_extreme_capacity_review.csv. The pipeline.log contains earlier development-run failures as well as later runs; the final schema inventory and artifact_validation.json report the completed run.\n'
 p=OUT/'DATA_QUALITY_AUDIT.md';s=p.read_text(encoding='utf8');s=s.split('\n\n## Final artifact validation')[0];p.write_text(s+audit,encoding='utf8')
 p=OUT/'FEATURE_ENGINEERING_REPORT.md';s=p.read_text(encoding='utf8').split('\n\n## Executed physical partial-charge diagnostic')[0]
 pp=OUT/'physical_partial_observation_manifest.json'
 if pp.exists():
  part=json.loads(pp.read_text());s+='\n\n## Executed physical partial-charge diagnostic\n\n'+part['scope']+'\n\n'
  metric=OUT/'physical_partial_observation_battery_macro.csv'
  if metric.exists():s+='```\n'+pd.read_csv(metric).to_string(index=False)+'\n```\n'
  s+='This bounded experiment re-extracts features from actual measured charging prefixes/suffixes. It is distinct from masking extracted features. It does not establish acquisition invariance across all voltage windows or protocols.\n'
 historymetric=OUT/'historical_observation_robustness_battery_macro.csv'
 if historymetric.exists():
  s+='\n\n## Missing historical observation vectors\n\nFour preceding observations are required within a verified continuity segment. The last two complete historical feature vectors are masked in one scenario. Both models train on complete and masked windows using identical evaluation masks.\n\n```\n'+pd.read_csv(historymetric).to_string(index=False)+'\n```\n'
 s+='\n\n## Final validation\n\n'+f'{len(extreme)} unusually low/high labeled capacities are retained and flagged for review. {summary["unresolved_cadex_files"]} CADEX sources fail the independent clock-consistency check. See artifact_validation.json and the audit for exact checks.\n'
 # Refresh masking metrics after the dependency-aware masking experiment.
 a10=s.find('## 10.');b11=s.find('## 11.',a10)
 if a10>=0 and b11>=0:
  fmacro=pd.read_csv(OUT/'forecasting_battery_macro.csv');rm=pd.read_csv(OUT/'robustness_battery_macro.csv')
  section='## 10. Forecasting and acquisition robustness\n\nForecasting battery-macro metrics (SOH fraction):\n```\n'+fmacro.to_string(index=False)+'\n```\n\nMissingness-aware feature-mask baselines:\n```\n'+rm.to_string(index=False)+'\n```\n\nCC/CV masking removes every selected feature requiring that phase, including whole-charge aggregate features. Identical evaluation masks are used for Ridge and HGB, and both train on missingness augmentation. Feature masking does not establish physical acquisition invariance. Actual raw prefixes/suffixes and missing historical vectors are reported separately below.\n\n'
  s=s[:a10]+section+s[b11:]
 # Explicitly report both strongest Pearson and strongest Spearman associations.
 a=s.find('## 3.');b=s.find('## 4.',a);c=s.find('## 5.',b)
 if a>=0 and b>=0 and c>=0:
  pr=rel.loc[rel.pearson.abs().nlargest(5).index];sp=rel.loc[rel.spearman.abs().nlargest(5).index];mi=rel.nlargest(5,'mutual_information')
  section='## 3. Strongest training-cell associations\n\nPearson:\n```\n'+pr.to_string(index=False)+'\n```\n\nSpearman:\n```\n'+sp.to_string(index=False)+'\n```\n\nEligibility flags distinguish exploratory associations from reliable primary predictors.\n\n'
  section+='## 4. Nonlinear dependence estimates\n\n```\n'+mi.to_string(index=False)+'\n```\n\nMutual information measures dependence, not proof of a nonlinear mechanism. P-values/BH q-values remain exploratory because cycles are serially dependent.\n\n'
  s=s[:a]+section+s[c:]
 verdict='\n\n## Generalization verdict\n\n'
 forecast=pd.read_csv(OUT/'forecasting_battery_macro.csv');comparison=forecast[(forecast.split=='test')&(forecast.model.isin(['ridge','hgb','persistence_origin_soh']))]
 verdict+='Untouched test cells are CS2_7 and CS2_9. The selected five-feature subset does not beat the simple charging baseline or measured-SOH persistence on these test cells. Thus this study does not establish a superior or acquisition-invariant five-feature forecaster. Protocol-conditioned capacity, aging extremes and the small number of independent cells limit transfer; these are explanations to investigate, not demonstrated causal attributions. No feature selection was retuned using these test results.\n\n```\n'+comparison.to_string(index=False)+'\n```\n'
 s=s.split('\n\n## Generalization verdict')[0]+verdict
 p.write_text(s,encoding='utf8')
 manifest['source_snapshot_sha256']=source_snapshot_sha256()
 manifest['selection_status']='provisional_generalization_not_established'
 top=pd.read_csv(OUT/'top_5_features.csv');top['generalization_status']='provisional: did not beat charging/persistence baselines on untouched test cells';top.to_csv(OUT/'top_5_features.csv',index=False)
 predictions=pd.read_csv(OUT/'forecasting_results.csv')
 if len(predictions):
  families={'CS2_3':'variable_rate','CS2_9':'variable_rate','CS2_7':'random_cutoff'}
  predictions['protocol_family']=predictions.battery.map(families).fillna('fixed_rate')
  predictions.groupby(['horizon','split','protocol_family','model','subset']).agg(rmse=('rmse','mean'),mae=('mae','mean'),r2=('r2','mean'),n_batteries=('battery','nunique')).reset_index().to_csv(OUT/'forecasting_protocol_family_macro.csv',index=False)
 manifest['implementation_sha256']={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in ['cs2_research_pipeline.py','cs2_research_analysis.py','cs2_continuity.py','cs2_statistics.py','cs2_acquisition_robustness.py','cs2_history_robustness.py','cs2_partial_views.py','finalize_cs2_research.py']}
 manifest['artifact_validation']=summary;manifest['output_directory']=str(OUT);manifest['notebook']=str(OUT.parents[1]/'CALCE_CS2_Feature_Research.ipynb');manifest['physical_partial_observation_executed']=pp.exists()
 manifest['historical_observation_robustness_executed']=(OUT/'historical_observation_robustness_protocol.json').exists()
 (OUT/'dataset_manifest.json').write_text(json.dumps(manifest,indent=2))
 files=[p for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='output_index.csv']
 pd.DataFrame([dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]).to_csv(OUT/'output_index.csv',index=False)
 print(json.dumps(summary,indent=2))
 return summary
if __name__=='__main__':finalize()
