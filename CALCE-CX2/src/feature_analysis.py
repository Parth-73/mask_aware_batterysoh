"""Frozen battery splits and training-only statistics; no predictive models are fitted."""
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr,spearmanr
from sklearn.feature_selection import mutual_info_regression
from .common import CONFIG, save_json
from .extract_features import FEATURES

# Frozen a priori from battery identifiers, not feature/target distributions.
SPLITS={'train':['CX2_16','CX2_33','CX2_34','CX2_35','CX2_36'],
        'validation':['CX2_37'], 'test':['CX2_38'],
        'protocol_holdout':['CX2_3','CX2_8','CX2_32'],
        'unresolved_text_sessions':['CX2_4','CX2_31']}
F1=['ic_peak_height','partial_charge_ah','partial_dv_dq']
F2=F1+['ic_peak_voltage','ic_peak_area','ic_peak_width','dva_mean','dva_peak','partial_charge_time_s','cc_duration_s','cv_duration_s','cv_decay_rate_s_inv','cv_capacity_ah']
F3=F2+['historical_ic_peak_shift_v','historical_partial_change_ah','charge_crate','charge_protocol_cv','relaxation_60s_v']


def split_name(battery):
    return next((name for name,values in SPLITS.items() if battery in values),'unassigned')


def safe_corr(x,y,method):
    ok=np.isfinite(x)&np.isfinite(y)
    if ok.sum()<8 or np.ptp(np.asarray(x)[ok])<1e-10 or np.ptp(np.asarray(y)[ok])<1e-10:return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter('ignore');return float((pearsonr if method=='pearson' else spearmanr)(np.asarray(x)[ok],np.asarray(y)[ok]).statistic)


def heatmap(matrix,path,title):
    if matrix.empty:return
    fig,ax=plt.subplots(figsize=(12,10));im=ax.imshow(matrix,cmap='coolwarm',vmin=-1,vmax=1)
    ax.set_xticks(range(len(matrix)),matrix.columns,rotation=90,fontsize=7);ax.set_yticks(range(len(matrix)),matrix.index,fontsize=7)
    ax.set_title(title);fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(path,dpi=140);plt.close(fig)


def compute_vif(train,variable):
    """VIF only where complete cases retain variance; never divide by zero."""
    if not variable:return []
    complete=train[variable].replace([np.inf,-np.inf],np.nan).dropna()
    active=[f for f in variable if len(complete) and complete[f].std(ddof=0)>1e-10]
    result=[dict(feature=f,vif=np.nan,n=len(complete),status='constant_or_absent_in_complete_cases') for f in variable if f not in active]
    if len(complete)<=len(active)+2:
        return result+[dict(feature=f,vif=np.nan,n=len(complete),status='insufficient_complete_cases') for f in active]
    if not active:return result
    a=complete[active].to_numpy();a=(a-a.mean(axis=0))/a.std(axis=0)
    for j,f in enumerate(active):
        other=np.delete(a,j,axis=1);design=np.c_[np.ones(len(a)),other]
        fit=np.linalg.lstsq(design,a[:,j],rcond=None)[0]
        fraction=float(np.var(a[:,j]-design@fit))
        result.append(dict(feature=f,vif=1/fraction if fraction>1e-12 else np.inf,n=len(a),status='ok' if fraction>1e-12 else 'rank_deficient'))
    return result


def analyze(frame,output):
    save_json(output/'battery_splits.json',dict(seed=42,policy='A priori identifier-based split; conventional eligible cycles only for selection.',splits=SPLITS))
    train=frame[(frame.split=='train')&frame.eligible_analysis].copy()
    rows=[];stability=[]
    for feature in FEATURES:
        x=train[feature].to_numpy(float);y=train.soh.to_numpy(float);ok=np.isfinite(x)&np.isfinite(y)
        mi=np.nan
        if ok.sum()>=10 and np.ptp(x[ok])>1e-10 and np.ptp(y[ok])>1e-10:
            mi=float(mutual_info_regression(x[ok,None],y[ok],random_state=42,n_neighbors=min(5,int(ok.sum())-1))[0])
        ps=[];ss=[]
        for battery,g in train.groupby('battery'):
            p=safe_corr(g[feature],g.soh,'pearson');s=safe_corr(g[feature],g.soh,'spearman')
            stability.append(dict(feature=feature,battery=battery,n=int(g[feature].notna().sum()),pearson=p,spearman=s))
            if np.isfinite(p):ps.append(p)
            if np.isfinite(s):ss.append(s)
        signs=np.sign(ss)
        consistency=float(max(np.mean(signs>=0),np.mean(signs<=0))) if len(signs) else np.nan
        rows.append(dict(feature=feature,n=int(ok.sum()),training_missing_percent=float(100*(1-ok.mean())) if len(ok) else np.nan,
                         pearson=safe_corr(x,y,'pearson'),spearman=safe_corr(x,y,'spearman'),mutual_information=mi,
                         battery_mean_pearson=float(np.mean(ps)) if ps else np.nan,battery_mean_spearman=float(np.mean(ss)) if ss else np.nan,
                         battery_std_spearman=float(np.std(ss)) if ss else np.nan,sign_consistency=consistency,batteries_with_correlation=len(ss)))
    rank=pd.DataFrame(rows)
    # Fixed physical/stability/availability gates, applied exclusively to training data.
    rank['selection_score']=(rank.battery_mean_spearman.abs()*rank.sign_consistency*(1-rank.training_missing_percent/100)/(1+rank.battery_std_spearman)).fillna(0)
    rank=rank.sort_values('selection_score',ascending=False)
    rank.to_csv(output/'feature_correlations.csv',index=False)
    pd.DataFrame(stability).to_csv(output/'feature_battery_stability.csv',index=False)
    variable=[f for f in FEATURES if train[f].notna().sum()>=10 and train[f].std()>1e-10]
    corr=train[variable].corr(method='spearman',min_periods=10)
    corr.to_csv(output/'feature_feature_correlations.csv')
    heatmap(corr,output/'figures'/'training_feature_heatmap.png','Training batteries only: Spearman feature correlation')
    # VIF on complete training rows only, no imputation. Rank deficiency is explicit.
    vifs=compute_vif(train,variable)
    pd.DataFrame(vifs,columns=['feature','vif','n','status']).to_csv(output/'feature_vif.csv',index=False)
    selected=[];selection=[]
    for r in rank.itertuples():
        reason='candidate'
        if r.feature=='charge_crate' and train.charge_crate.quantile(.95)-train.charge_crate.quantile(.05)<.05:reason='insufficient_physical_charge_protocol_variation_below_0.05C'
        elif r.training_missing_percent>30 or not np.isfinite(r.training_missing_percent):reason='availability_below_70_percent'
        elif r.batteries_with_correlation<2:reason='fewer_than_two_training_batteries'
        elif r.sign_consistency<.75:reason='inconsistent_between_batteries'
        elif not np.isfinite(r.battery_mean_spearman) or abs(r.battery_mean_spearman)<.3:reason='weak_within_battery_rank_association'
        elif any(np.isfinite(corr.loc[r.feature,s]) and abs(corr.loc[r.feature,s])>.9 for s in selected):reason='redundant_rank_correlation_above_0.90'
        elif len(selected)>=6:reason='compact_limit_six'
        else:selected.append(r.feature);reason='provisionally_selected'
        selection.append(dict(feature=r.feature,decision=reason))
    pd.DataFrame(selection).to_csv(output/'feature_selection_decisions.csv',index=False)
    save_json(output/'feature_sets.json',dict(F1=F1,F2=F2,F3=F3,F4=selected,
        unavailable_pending_measurement_verification=['resistance_ohm','historical_resistance_change_ohm','temperature_c'],
        F4_status='provisional_training_statistics_only; validation-performance selection deferred because no models are allowed',
        baseline_status='Operational three-feature baseline; exact Wang et al. study interval/smoothing not supplied or verified.',
        exclusions=['soh','soh_candidate','discharge_crate_median','discharge_crate_max','discharge_integral_ah','discharge_counter_ah','battery','lifetime_cycle','split'],
        missing_data_policy='Physical measurements remain NaN. Any later learned imputer/scaler must fit training batteries only.',
        architecture_policy='Use identical feature columns and temporal availability for all architectures; reshape only.',
        same_cycle_discharge_policy='Audit only; an independently logged pre-charge scheduled protocol may be added in a future dataset version.'))
    return rank,selected


def plot_evolution(frame,output):
    specs=[('soh','Validated SOH'),('ic_peak_voltage','IC peak voltage (V)'),('cc_duration_s','CC duration (s)'),('cv_duration_s','CV duration (s)')]
    for col,label in specs:
        fig,ax=plt.subplots(figsize=(10,5))
        for battery,g in frame.groupby('battery'):
            x=g[g.validated_complete] if col=='soh' else g
            ax.plot(x.lifetime_cycle,x[col],'.-',ms=2,lw=.5,label=battery)
        ax.set(xlabel='Observed reconstructed cycle ordinal (includes rejected candidates)',ylabel=label);ax.legend(fontsize=7,ncol=3);fig.tight_layout();fig.savefig(output/'figures'/f'{col}_evolution.png',dpi=140);plt.close(fig)
    # Separate per-battery feature distributions, never used in test feature selection.
    for features,name in [(F1,'baseline'),(['cc_duration_s','cv_duration_s','cv_capacity_ah'],'charging')]:
        fig,axes=plt.subplots(1,3,figsize=(14,5))
        for ax,f in zip(axes,features):
            groups=[(b,g[f].dropna()) for b,g in frame.groupby('battery') if g[f].notna().any()]
            if groups:ax.boxplot([g for _,g in groups],tick_labels=[b for b,_ in groups],showfliers=False)
            ax.tick_params(axis='x',rotation=90);ax.set_title(f)
        fig.tight_layout();fig.savefig(output/'figures'/f'{name}_distributions.png',dpi=130);plt.close(fig)


def plot_quality(frame,output):
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for b,g in frame.groupby('battery'):
        axes[0].scatter(g.discharge_counter_ah,g.discharge_integral_ah,s=3,label=b)
        valid=g.ic_smoothing_relative_spread.dropna()
        if len(valid):axes[1].hist(valid,bins=30,alpha=.4,label=b)
    axes[0].plot([0,1.8],[0,1.8],'k--',lw=1)
    axes[0].set(xlabel='Discharge counter delta (Ah)',ylabel='Current integration (Ah)')
    axes[1].axvline(.2,color='k',linestyle='--');axes[1].set(xlabel='IC smoothing relative peak spread',ylabel='Cycles')
    axes[0].legend(fontsize=6);axes[1].legend(fontsize=6);fig.tight_layout();fig.savefig(output/'figures'/'capacity_and_ic_quality.png',dpi=140);plt.close(fig)
    summary=frame.groupby('battery').agg(validated=('validated_complete','sum'),total=('cycle_id','size'))
    summary['rejected']=summary.total-summary.validated
    ax=summary[['validated','rejected']].plot.bar(stacked=True,figsize=(10,5));ax.set_ylabel('Reconstructed candidates');ax.figure.tight_layout();ax.figure.savefig(output/'figures'/'cycle_completeness.png',dpi=140);plt.close(ax.figure)
    missing=frame.groupby('battery')[FEATURES].agg(lambda x:100*x.isna().mean())
    fig,ax=plt.subplots(figsize=(14,6));im=ax.imshow(missing,vmin=0,vmax=100,cmap='magma')
    ax.set_xticks(range(len(missing.columns)),missing.columns,rotation=90,fontsize=7);ax.set_yticks(range(len(missing)),missing.index);fig.colorbar(im,ax=ax,label='Missing %');fig.tight_layout();fig.savefig(output/'figures'/'feature_missingness.png',dpi=140);plt.close(fig)
