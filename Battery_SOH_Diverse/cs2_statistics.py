"""Complete training-cell exploratory statistics, including ineligible measured candidates."""
from collections import Counter
import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control
from cs2_research_pipeline import OUT
from cs2_research_analysis import FEATURES,relevance,eligible,forecast_pairs

def write_full_statistics(data,manifest):
 train=data[data.battery.isin(manifest['train_batteries'])&data.soh.notna()].copy()
 rel=relevance(train,FEATURES);rel['eligible_current_cycle']=rel.feature.isin(eligible(train))
 for stat in ['pearson','spearman','kendall']:
  valid=rel[stat+'_p'].notna();rel.loc[valid,stat+'_bh_q']=false_discovery_control(rel.loc[valid,stat+'_p'].to_numpy())
 rel.to_csv(OUT/'pearson_spearman_kendall.csv',index=False);rel[['feature','mutual_information','n','eligible_current_cycle']].to_csv(OUT/'mutual_information.csv',index=False)
 cells=[]
 for b,g in train.groupby('battery'):
  r=relevance(g,FEATURES)
  if len(r):r['battery']=b;cells.append(r)
 pc=pd.concat(cells,ignore_index=True);pc.to_csv(OUT/'per_battery_feature_correlations.csv',index=False)
 stability=pc.groupby('feature').agg(mean_pearson=('pearson','mean'),std_pearson=('pearson','std'),mean_spearman=('spearman','mean'),std_spearman=('spearman','std'),n_batteries=('battery','nunique'),direction_consistency=('spearman',lambda x:max((x>0).mean(),(x<0).mean())))
 stability.to_csv(OUT/'per_battery_correlation_stability.csv')
 matrices=[]
 for method in ['pearson','spearman']:
  corr=train[FEATURES].corr(method=method);corr.index.name='feature';long=corr.reset_index().melt(id_vars='feature',var_name='other',value_name='correlation');long['method']=method;matrices.append(long)
 pd.concat(matrices).to_csv(OUT/'feature_redundancy_matrix.csv',index=False)
 fold=pd.read_csv(OUT/'fold_selected_features.csv');counts=Counter(fold.feature);den=len(manifest['train_batteries'])
 freq=pd.DataFrame([dict(feature=c,selected_count=counts[c],outer_folds=den,frequency=counts[c]/den) for c in FEATURES]);freq.to_csv(OUT/'feature_selection_stability.csv',index=False)
 selection=forecast_pairs(data,1);selection=selection[selection.battery.isin(manifest['train_batteries'])].copy()
 manifest['eligible_features_for_selection']=eligible(selection) if manifest['selection_task']=='one_cycle_forecasting' else eligible(train)
 if len(freq[freq.selected_count>0]):
  import matplotlib.pyplot as plt
  f=freq[freq.selected_count>0].sort_values('frequency');fig,ax=plt.subplots(figsize=(10,max(4,len(f)*.35)));ax.barh(f.feature,f.frequency);ax.set_xlabel('Selection frequency across outer held-out-cell folds');fig.tight_layout();fig.savefig(OUT/'feature_selection_stability.png',dpi=150);plt.close(fig)
 return rel,freq
