"""Independent raw-curve spot checks and publication diagnostics; no model selection."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.common import ROOT
from src.training.config import json_write
from src.reconstruct_cycles import prepare_battery,reconstruct
from .prepare import OUT,PARENT,FEATURES,BATTERIES,GRID,curve_tensor,validate_cached_data


def save(fig,name):
    fig.tight_layout();fig.savefig(OUT/'plots'/f'{name}.png',dpi=300,bbox_inches='tight')
    fig.savefig(OUT/'plots'/f'{name}.pdf',bbox_inches='tight');plt.close(fig)


def main():
    f,curves=validate_cached_data()
    original=pd.read_csv(ROOT/'outputs/soh_four_model_benchmarks/metrics/all_runs.csv')
    original=original[original.experiment.eq('primary')].copy()
    old_ids=set(pd.read_csv(ROOT/'outputs/soh_four_model_benchmarks/splits/windows_L5_matchedFalse.csv').cycle_id)
    new_ids=set(pd.read_csv(OUT/'splits/targets_L5.csv').cycle_id)
    original['comparison_status']='historical_same_target_ids_additional_raw_information_and_different_training_budget'
    original['model_family']='historical_engineered_feature_baseline'
    original.to_csv(OUT/'metrics/stage0_reference_baselines.csv',index=False)
    json_write(OUT/'data_audit/baseline_comparability.json',{'same_primary_target_ids':old_ids==new_ids,
        'historical_target_count':len(old_ids),'new_target_count':len(new_ids),'historical_epochs':10,
        'historical_features':FEATURES,'new_information':'raw voltage-axis Q/dQdV/I plus engineered features',
        'no_retraining_of_completed_baselines':True,'caution':'Target matching does not equal information or training-budget matching.'})
    # Per-cycle event availability uses observation/cutoff information only. Discharge-rate
    # comparability is retained solely in the event audit, never in this predictor metadata.
    e=pd.read_csv(OUT/'rest_events/rest_event_inventory.csv')
    v=e[e.category.eq('verified_measured_rest') & e.available_for_cached_target]
    uncertain=e[e.category.eq('uncertain_recording_gap') & e.available_for_cached_target]
    durations=v.groupby('following_cycle_id').duration_s.sum()
    observability=f[['battery','cycle_id','prediction_cutoff']].copy()
    observability['observed_pre_cutoff_rest_s']=observability.cycle_id.map(durations)
    observability['verified_short_wait_observed']=observability.observed_pre_cutoff_rest_s.notna()
    observability['uncertain_gap_present']=observability.cycle_id.isin(uncertain.following_cycle_id)
    observability['category']=np.where(observability.verified_short_wait_observed,'verified_measured_rest',
        np.where(observability.uncertain_gap_present,'uncertain_recording_gap','no_verified_rest'))
    observability.to_csv(OUT/'rest_events/per_cycle_observability.csv',index=False)
    rechecks=[]
    manifest=pd.read_csv(PARENT/'data_audit/inspection_manifest.csv')
    for battery in BATTERIES:
        group=f[f.battery.eq(battery)]
        source=group.sources.value_counts().index[0]
        candidates=group[group.sources.eq(source)]
        row=candidates.iloc[len(candidates)//2]
        source_file,sheet=source.split('::')
        subset=manifest[manifest.battery.eq(battery)&manifest.source_file.eq(source_file)&manifest.sheet.eq(sheet)]
        d,_=prepare_battery(subset,battery)
        cycles,_=reconstruct(d,battery)
        matches=[(r,parts) for r,parts in cycles if pd.Timestamp(r['start'])==pd.Timestamp(row.start)]
        assert len(matches)==1
        r,parts=matches[0]
        fresh=curve_tensor(parts)
        cached=np.asarray(curves[f.index[f.cycle_id.eq(row.cycle_id)][0]])
        assert np.allclose(fresh,cached,atol=1e-6,rtol=1e-6)
        assert np.isclose(r['soh'],row.soh,atol=1e-9,rtol=0)
        cutoff=max(g.timestamp.max() for g in parts if g.phase.iloc[0]=='charge')
        assert cutoff==pd.Timestamp(row.prediction_cutoff)
        rechecks.append({'battery':battery,'cycle_id':row.cycle_id,'source':source,
                        'raw_reconstruction_and_curve_match':True,'end_of_charge_cutoff':str(cutoff)})
    json_write(OUT/'data_audit/raw_curve_rechecks.json',rechecks)
    fig,axes=plt.subplots(1,3,figsize=(12,3.8))
    labels=['Q (Ah)','dQ/dV (Ah/V)','I (A)']
    for i in np.linspace(0,len(f)-1,7,dtype=int):
        for channel,ax in enumerate(axes):ax.plot(GRID,curves[i,channel],label=f.cycle_id.iloc[i],alpha=.8)
    for ax,label in zip(axes,labels):ax.set(xlabel='Charge voltage (V)',ylabel=label)
    axes[0].legend(fontsize=5);fig.suptitle('Measured full-coverage CNN inputs; no per-curve capacity normalization')
    save(fig,'raw_curves')
    fig,axes=plt.subplots(3,2,figsize=(11,10))
    for battery,g in f.groupby('battery'):
        for ax,feature in zip(axes.flat,FEATURES):ax.plot(g.lifetime_cycle,g[feature],label=battery,linewidth=.6)
        axes.flat[-1].plot(g.lifetime_cycle,g.soh*100,label=battery,linewidth=.7)
    units=['s','s','V/Ah','V','V']
    for ax,feature,unit in zip(axes.flat,FEATURES,units):ax.set(xlabel='Observed reconstructed cycle',ylabel=f'{feature} ({unit})')
    axes.flat[-1].set(xlabel='Observed reconstructed cycle',ylabel='Validated SOH (%)');axes.flat[-1].legend(fontsize=6)
    save(fig,'features_and_soh')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].hist(v.duration_s,bins=40,color='#267b91');axes[0].set(xlabel='Verified observed rest interval (s)',ylabel='Intervals',title='Short protocol waits; no verified hour-scale rests')
    counts=observability.category.value_counts();axes[1].bar(range(len(counts)),counts)
    axes[1].set_xticks(range(len(counts)),counts.index,rotation=20,ha='right');axes[1].set(ylabel='Cached cycles',title='Pre-cutoff observation availability')
    save(fig,'rest_feasibility')
    print('Seven raw re-extraction checks passed; original baselines indexed separately.',flush=True)


if __name__=='__main__':main()
