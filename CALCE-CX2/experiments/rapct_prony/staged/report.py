"""Artifact-derived stage reports; unavailable experiments receive status, never scores."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.training.config import json_write
from src.benchmark.engine import metrics
from .prepare import OUT,PARENT,SOURCE
from .diagnostics import save


def label(row):
    if row['stage']=='A3':return 'A3_frozen' if row.get('freeze_cnn',False) else 'A3_joint'
    if row['stage']=='A4':return 'A4_zero' if row.get('physics_weight',0.)==0 else 'A4_guided'
    return row['stage']


def markdown(frame):
    if frame.empty:return 'No completed results in this domain.'
    def fmt(v):return f'{v:.6g}' if isinstance(v,(float,np.floating)) else str(v)
    return '\n'.join(['| '+' | '.join(frame.columns)+' |','| '+' | '.join(['---']*len(frame.columns))+' |']+
        ['| '+' | '.join(fmt(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)])


def generate():
    validation=[];outer=[];all_validation=[];all_outer=[];regions=[];predictions=[]
    for directory in sorted((OUT/'checkpoints').iterdir()):
        file=directory/'validation_result.json'
        if not file.exists():continue
        row=json.loads(file.read_text())
        row.update(model=label(row),dataset='CALCE-CX2',domain='experimental_current_charge_SOH')
        history=pd.read_csv(directory/'training.csv')
        if 'effective_physics_weight' in history:
            row['effective_physics_weight_at_best']=float(history.loc[history.epoch.eq(row['best_epoch']),'effective_physics_weight'].iloc[0])
            row['best_checkpoint_prior_active']=row['effective_physics_weight_at_best']>0
        lineage_seconds=row['training_seconds'];parent=row.get('parent')
        while parent:
            ancestor=Path(parent).parent
            predecessor=json.loads((ancestor/'validation_result.json').read_text())
            lineage_seconds+=predecessor['training_seconds'];parent=predecessor.get('parent')
        row['cumulative_lineage_training_seconds']=lineage_seconds
        all_validation.append(row.copy())
        if row.get('purpose')=='final':validation.append(row)
        file=directory/'outer_result.json'
        if file.exists():
            row={**json.loads(file.read_text()),'model':label(row),'dataset':'CALCE-CX2','domain':'experimental_current_charge_SOH'}
            row['cumulative_lineage_training_seconds']=lineage_seconds
            all_outer.append(row.copy())
            if row.get('purpose')!='final':continue
            outer.append(row)
            p=pd.read_csv(directory/'predictions_outer.csv');p['model']=row['model'];predictions.append(p)
            for region,mask in [('above_90',p.actual_soh>.9),('80_to_90',p.actual_soh.between(.8,.9)),('below_80',p.actual_soh<.8)]:
                regions.append({k:row[k] for k in ['model','stage','configuration_id','seed','outer','dataset']}|
                    {'region':region}|(metrics(p.actual_soh[mask].to_numpy(),p.predicted_soh[mask].to_numpy()) if mask.any() else {'n_samples':0}))
    v=pd.DataFrame(validation);o=pd.DataFrame(outer)
    data=pd.concat([v,o],ignore_index=True)
    if data.empty:return
    pd.DataFrame(all_validation+all_outer).to_csv(OUT/'metrics/all_runs.csv',index=False)
    o.to_csv(OUT/'metrics/outer_fold_metrics.csv',index=False)
    numbers=['mae_fraction','mse_fraction_squared','rmse_fraction','mae_percentage_points','rmse_percentage_points',
             'mape_percent','smape_percent','r2','training_seconds','parameter_count','peak_gpu_memory_mb']
    grouping=['model','stage','evaluation_split','outer','epochs']
    summary=data.groupby(grouping)[numbers].agg(['mean','std','count'])
    summary.columns=['_'.join(c) for c in summary.columns];summary=summary.reset_index()
    summary['dataset']='CALCE-CX2';summary['source_seeds']='42,123,2026'
    summary['scope']='one executed outer battery; seed sample SD is algorithmic variability, not population uncertainty'
    summary.to_csv(OUT/'metrics/stage_comparison.csv',index=False)
    summary.to_csv(OUT/'metrics/seed_stability.csv',index=False)
    summary.to_csv(OUT/'metrics/battery_macro_summary.csv',index=False)
    summary[summary.evaluation_split.eq('outer')].to_csv(OUT/'metrics/final_comparison.csv',index=False)
    summary[summary.stage.eq('A4')].to_csv(OUT/'metrics/physics_ablation.csv',index=False)
    summary[summary.stage.isin(['A0','A1','A2'])].to_csv(OUT/'metrics/feature_ablation.csv',index=False)
    pd.DataFrame(regions).to_csv(OUT/'metrics/soh_region_metrics.csv',index=False)
    columns=[c for c in ['model','stage','configuration_id','dataset','outer','validation','seed','evaluation_split',
                         'epochs','training_seconds','inference_seconds','inference_ms_per_sample','parameter_count','peak_gpu_memory_mb'] if c in data]
    columns+=[c for c in ['cumulative_lineage_training_seconds'] if c in data]
    data[columns].to_csv(OUT/'metrics/model_complexity.csv',index=False)
    unavailable=pd.DataFrame([{'stage':stage,'dataset':'CALCE-CX2','status':'not_executed_recovery_feasibility_gate',
        'reason':'Only short protocol waits verified; irregular long-rest recovery and latent-state identification unsupported'} for stage in ['A5','A6','A7','A8']])
    unavailable.to_csv(OUT/'metrics/rest_event_metrics.csv',index=False)
    activation_rows=[r for r in all_validation if r.get('module')=='stage3']
    if activation_rows:
        pd.DataFrame(activation_rows)[['configuration_id','purpose','outer','validation','seed','physics_weight','smooth_weight',
            'best_epoch','epochs_completed','effective_physics_weight_at_best','best_checkpoint_prior_active']].to_csv(OUT/'metrics/physics_checkpoint_activation.csv',index=False)
    synthetic=OUT/'latent_states/synthetic/ablation.csv'
    if synthetic.exists():pd.read_csv(synthetic).to_csv(OUT/'metrics/prony_ablation.csv',index=False)
    else:pd.DataFrame(columns=['domain','modes','learnable','status','synthetic_mse']).to_csv(OUT/'metrics/prony_ablation.csv',index=False)
    trials=[]
    for path in (OUT/'optuna_studies').glob('*_trials.csv'):
        t=pd.read_csv(path);t['study']=path.stem;trials.append(t)
        fig,ax=plt.subplots(figsize=(6,4))
        completed=t[t.state.eq('COMPLETE')]
        ax.plot(completed.number,completed.value,marker='o')
        ax.set(xlabel='Trial number',ylabel='Inner battery-macro RMSE (SOH fraction)',title=path.stem)
        save(fig,'optimization_'+path.stem)
        fig,ax=plt.subplots(figsize=(6,3));ax.axis('off')
        ax.text(.05,.5,f'{len(completed)} completed pilot trials.\nHyperparameter importance is not estimated\nfrom this insufficient trial population.',transform=ax.transAxes)
        save(fig,'importance_unavailable_'+path.stem)
    if trials:pd.concat(trials,ignore_index=True).to_csv(OUT/'metrics/hyperparameter_trials.csv',index=False)
    for directory in (OUT/'checkpoints').iterdir():
        file=directory/'validation_result.json'
        if not file.exists() or json.loads(file.read_text()).get('purpose')!='final':continue
        h=pd.read_csv(directory/'training.csv')
        fig,axes=plt.subplots(1,2,figsize=(10,3.5))
        axes[0].plot(h.epoch,h.data_loss,label='Train data MSE')
        axes[0].plot(h.epoch,h.validation_battery_macro_rmse**2,label='Validation MSE')
        axes[0].set(xlabel='Epoch',ylabel='SOH fraction squared',title=directory.name);axes[0].legend(fontsize=7)
        for name in ['physics_loss','smoothness_loss']:
            if name in h:axes[1].plot(h.epoch,h[name],label=name)
        axes[1].set(xlabel='Epoch',ylabel='Unweighted prior loss',title='Separate prior components');axes[1].legend(fontsize=7)
        save(fig,'learning_'+directory.name)
    if predictions:
        p=pd.concat(predictions,ignore_index=True)
        p.to_csv(OUT/'predictions/all_outer_predictions.csv',index=False)
        observed=pd.read_csv(OUT/'rest_events/per_cycle_observability.csv').set_index('cycle_id')
        short_wait=[]
        for (model,seed,fold),group in p.groupby(['model','seed','outer']):
            lookup=observed.reindex(group.cycle_id)
            indicator=lookup.verified_short_wait_observed.to_numpy()
            for category,mask in [('verified_observed_short_wait',indicator),('no_verified_wait_observation',~indicator)]:
                if not mask.any():continue
                short_wait.append({'model':model,'seed':seed,'outer':fold,'dataset':'CALCE-CX2',
                    'event_class':category,'status':'descriptive_current_cycle_estimation',
                    'interpretation':'Ordinary short protocol waits; no independent capacity-recovery or non-rest claim',
                    **metrics(group.actual_soh.to_numpy()[mask],group.predicted_soh.to_numpy()[mask])})
        pd.concat([unavailable,pd.DataFrame(short_wait)],ignore_index=True).to_csv(OUT/'metrics/rest_event_metrics.csv',index=False)
        pd.DataFrame(short_wait).to_csv(OUT/'metrics/short_wait_conditional_errors.csv',index=False)
        for battery,g in p.groupby('battery'):
            fig,ax=plt.subplots(figsize=(10,5))
            true=g[['cycle_id','lifetime_cycle','actual_soh']].drop_duplicates().sort_values('lifetime_cycle')
            ax.plot(true.lifetime_cycle,true.actual_soh*100,'k',label='Validated SOH',linewidth=1.4)
            for name,h in g.groupby('model'):
                means=h.groupby('lifetime_cycle').predicted_soh.mean()
                ax.plot(means.index,means*100,label=name,linewidth=.8)
            ax.set(xlabel='Reconstructed observed cycle ordinal',ylabel='SOH (%)',title=battery+' outer predictions: seed means');ax.legend(fontsize=7)
            save(fig,'outer_'+battery)
        for metric in ['mae_percentage_points','rmse_percentage_points']:
            scores=o.groupby('model')[metric].agg(['mean','std'])
            fig,ax=plt.subplots(figsize=(8,4));ax.bar(range(len(scores)),scores['mean'],yerr=scores['std'],capsize=3)
            ax.set_xticks(range(len(scores)),scores.index,rotation=20);ax.set(ylabel=metric.replace('_',' '),title='Executed outer fold: seed mean ± sample SD')
            save(fig,'comparison_'+metric)
        fig,ax=plt.subplots(figsize=(7,4))
        for name,g in o.groupby('model'):ax.scatter(g.parameter_count.mean(),g.rmse_percentage_points.mean(),label=name)
        ax.set(xlabel='Parameters',ylabel='RMSE (SOH percentage points)',title='Observed accuracy versus model size');ax.legend(fontsize=7)
        save(fig,'complexity_accuracy')
        # Meaningful latent aging audits are limited to valid segments; offsets across resets
        # are unknown histories and never counted as physical negative increments.
        for row in o[o.stage.eq('A4')].itertuples():
            directory=Path(row.checkpoint_path).parent
            latent=directory/'latent_outer.csv'
            if not latent.exists():continue
            l=pd.read_csv(latent);fig,axes=plt.subplots(1,2,figsize=(10,4))
            for _,g in l.groupby('segment_id'):
                axes[0].plot(g.lifetime_cycle,g.degradation,linewidth=.6)
                axes[1].plot(g.lifetime_cycle,g.observation_residual,linewidth=.6)
            axes[0].set(xlabel='Observed cycle ordinal',ylabel='Latent D (fraction)',title='Unknown histories reinitialized at boundaries')
            axes[1].set(xlabel='Observed cycle ordinal',ylabel='Bounded observation residual (fraction)',title='Not a measured recovery state')
            save(fig,'latent_'+row.configuration_id)
    manifest=json.loads((OUT/'processed/manifest.json').read_text())
    environment=json.loads((OUT/'configs/environment.json').read_text())
    feasibility=json.loads((OUT/'rest_events/feasibility.json').read_text())
    gates={path.stem:json.loads(path.read_text()) for path in (OUT/'configs').glob('*_gate_stage*.json')}
    valrank=v.groupby('model').rmse_fraction.mean().sort_values()
    recommendation_files=list((OUT/'configs').glob('*_validation_recommendation.json'))
    recommendation=json.loads(recommendation_files[0].read_text())['recommendation'] if recommendation_files else valrank.index[0]
    public=summary[summary.evaluation_split.eq('outer')][['model','outer','epochs','mae_percentage_points_mean','rmse_percentage_points_mean','mape_percent_mean','smape_percent_mean','r2_mean']]
    status={'status':'partial_nested_reduced_budget_pilot','completed_final_training_runs':len(v),'completed_outer_evaluations':len(o),
            'outer_batteries':sorted(o.outer.unique()) if len(o) else [],'eligible_batteries':manifest['batteries'],
            'primary_windows':manifest['window_counts']['5'],'recovery_experimental_status':'unavailable',
            'validation_recommendation':recommendation if len(valrank) else None,
            'completed_search_inner_and_secondary_fits':len(all_validation)-len(v),
            'final_comparison':json.loads(public.to_json(orient='records')),'full_seven_fold_nested_evaluation_completed':False}
    json_write(OUT/'reports/final_results.json',status)
    text=f'''# RAPCT-Prony staged research report

## 1. Related work and research gap
See [verified primary-source review](../../../../experiments/rapct_prony/research/RELATED_WORK_VERIFIED.md), including all supplied papers, access limitations, reference-code revision, assumptions and a comparison matrix. CNN-Transformer fusion, degradation decomposition, physics-guided SOH and exponential memory are established ingredients. Neither publishable novelty nor the central recovery benefit is established.

## 2. Environment and data
Python {environment['python']}. Installed versions and actual current GPU/CUDA/driver/VRAM are in configs/environment.json and requirements.txt. Deterministic seeds are 42, 123, 2026; experimental training is FP32 by default. Optuna and the Excel parser are workspace-local dependencies. Actual raw content hashes, original feature/configuration snapshots and seven independent raw reconstruction/curve/cutoff checks are saved in data_audit. There are {manifest['samples']} complete raw-curve/feature cycles across seven eligible conventional cells and {manifest['window_counts']['5']} eligible length-5 targets. Length 10/20 eligibility is {manifest['window_counts']['10']}/{manifest['window_counts']['20']}; a matched length-20 target control is supported separately. Full raw reconstruction was not repeated; validated caches were checked and reused.

## 3. Preprocessing, features and prediction contract
Original quality checks and measured-current capacity integration are preserved. SOH=validated discharge Ah/1.35, fractional, unsmoothed and unclipped. Current-cycle end-of-charge is the cutoff; no same-cycle discharge measurement or later rest enters an encoder. Primary engineered features are cc_duration_s, cv_duration_s, dva_peak, ic_peak_voltage, ic_peak_width. Original IC/DVA/CC/CV definitions are unchanged. Raw channels are Q(V) in Ah, dQ/dV in Ah/V, I(V) in A on 176 measured-coverage points at 0.002 V over 3.80–4.15 V. No extrapolation or per-curve capacity normalization. Training-only feature/channel StandardScalers and fit cycle IDs are saved per run. Rejected/missing cycles, source changes, recorder resets, overlaps and gaps never become artificial consecutive windows.

## 4. Raw-curve CNN and Stage-1 comparisons
Independent voltage-axis branches use kernels 3/5/7, GroupNorm, GELU and two convolutions. Concatenated pooled features project to 64; a separate 5→32→32 MLP feeds a 96→64 fusion for initial dimensions. A0, A1 and A2 use the same target cohort, but A0/A1/A2 only observe current charge. Their actual input ID lists reflect that. Stage-1 final pilot budget is 15 epochs rather than the requested full 150. Selected channels/embedding and all trial records are in configs and optuna_studies. Existing unrelated full-cycle prototype and four-model experiments are preserved and indexed separately; identical targets do not make their information and training budgets equivalent.

## 5. Temporal encoder
The separate Stage-2 model uses chronological sinusoidal positions, 64-dimensional representations, four heads, two layers, FFN 128 and dropout 0.1 initially. Every attention layer uses triangular causal and padding masks, with explicit tests. No token attends to a later charge cycle. Frozen versus jointly trained curve encoders are compared with inner validation only. Encoder pretraining excludes the relevant validation and outer battery. Attention arrays are descriptive, not causal feature importance.

## 6. Irreversible degradation and losses
See staged/ARCHITECTURE.md and Stage-3 gate/configuration for exact executed equations. A positive encoder-estimated initial D handles unknown prehistory; nonnegative rates accumulate only within verified chronological segments. Detached state carry across chunks and encoder reinitialization at discontinuities are explicit. Nonnegative increments are parameterized, not double-penalized with an identically zero loss. An independent encoder state proposal versus accumulated state and rate-variation terms are mathematical priors; no recurrence is compared against itself. A bounded observation residual permits fluctuations and is not labeled recovered capacity. No Arrhenius, impedance, SOC or calibrated SEI equation is invented. The initial 0.01/0.001 coefficients use the requested epoch-10/30 final warm-up; any shorter search schedule is explicitly saved. Zero-weight controls retain the same head/recurrence. Latent D and the learned reference have offset-identifiability limitations.

## 7. Rest-event feasibility and memory mechanisms
Re-audited verified intervals: {feasibility['verified_observed_intervals']}; uncertain intervals/gaps: {feasibility['uncertain_gaps']}. Maximum verified duration is {feasibility['duration_quantiles_s']['1.0']} seconds; verified hour-scale rests: {feasibility['verified_rests_over_one_hour']}. Active boundary observations and inconsistent clocks were downgraded. The prior inventory's fragments are not independent interventions. Category and cutoff availability are retained per event and per cycle. Discharge-condition matching is retrospective audit information and never an event trigger. Measured capacity increases never define rest inputs. Ordinary short waits cannot establish the irregular long-rest recovery hypothesis or identifiable transient mechanisms; this does not imply that short waits have no capacity effect.

Single-mode and multi-mode recovery results, when present, are labeled SYNTHETIC. Official NPSSL code and MIT attribution are pinned; no oscillator PDE is imported. The adaptation accepts/returns state, maintains memory during active operation, handles unknown gaps explicitly and uses positive bounded time constants and bounded nonnegative penalty. Synthetic time ranges are artificial; no experimental battery time constants are inferred. Experimental A5–A8 results are unavailable under this gate. No synthetic success is described as battery recovery validation.

## 8. Hyperparameter search and evaluation
SQLite-backed Optuna TPE, MedianPruner, stage-wise searches and battery-macro inner RMSE. The resource-limited pilot uses two trials per component and two inner validation cells; this is below the requested 10–15/8–12-trial budgets. Pruning reports compare equal completed epoch progress across the same two inner folds. Trial CSVs include parameters, scores, runtime and states; study DBs resume and complete unchanged fits are reused. Two trials do not support stable importance estimates; plots state this rather than inventing importance. Full search-space helpers are provided, but broad simultaneous search is not claimed.

Only outer CX2_38 is executed in this pilot; inner validation cells are CX2_16 and CX2_33 during search. Final architecture recommendation equally averages three-seed validation means over both inner batteries; outer seed checkpoints use CX2_16. All selected final configurations use seeds 42/123/2026. A0–A3 final pilot fits use 15 epochs; A4 uses 35 to activate the full warm-up schedule. Across-stage differences are confounded by that budget difference; the A4 zero/nonzero pair has equal budget. AdamW starts at 5e-4, decay 1e-3, batch 32, clipping 1, scheduler factor .5/patience 5/floor 1e-6 and stopping patience 20. All stage choices are frozen before any new outer predictions. These explored batteries support cross-battery benchmarking, not untouched external validation. Full seven-fold nested confirmation and full 150/200-epoch budgets are unexecuted.

## 9. Quantitative comparison
{markdown(public)}

MAE/RMSE above are SOH percentage points; all fractional metrics, MSE, relative MAPE/sMAPE and R2 are in metrics/all_runs.csv. MAPE uses max(abs(actual),1e-8); sMAPE uses max(abs(actual)+abs(prediction),1e-8). Signed bias is prediction minus actual. Seed SD uses ddof=1 and reflects optimization variability on one outer cell, not a battery-population confidence interval. Region tables include counts. Timing scope and measured parameter/device memory metadata are retained. Incremental training duration and cumulative parent-lineage training cost are distinguished; outer timing includes frozen-CNN encoding, model/state forward and CPU transfer. A cached Transformer-only forward is not presented as the full raw-curve inference cost. All completed search/pretraining/inner-validation fits remain individually labeled in all_runs.csv.

## 10. Ablations, physical audits and limitations
Feature pathways, frozen/joint encoders and zero/nonzero priors have separate labels and configurations. Only matched target cycles enter controlled new-stage comparisons. State consistency, negative-increment frequency, residual magnitude and saturation are reported only where computed from actual latent outputs. physics_checkpoint_activation.csv records the effective prior weight at each validation-selected best epoch: a configured nonzero coefficient is not proof that the selected checkpoint was trained with active priors. Early stopping can prevent completion of the full warm-up. Descriptive errors following sampled short waits are separated from unavailable independent recovery results; absence of a verified observation is never called confirmed non-rest. Synthetic mode-count/fixed-versus-learned ablations are in a separate domain. They do not identify unique electrochemical relaxation times. Seven prior-explored cells, complete-case selection, repeated windows, ordinary protocol rests, unknown history and protocol-dependent capacity labels limit generalization. No causal recovery interpretation or uncertainty calibration is claimed.

## 11. Conclusions and completion status
Completed final fits: {len(v)}; outer evaluations: {len(o)}. Budget-specific validation recommendation: {status['validation_recommendation']}. The central rest-recovery advantage is **unresolved**, not demonstrated or refuted by unavailable experiments. See reports/final_results.json for machine-readable status and stage gate files for exact completed work. Models, complete resume states, predictions, scalers, latent trajectories, figures and Optuna databases are preserved. Later independent cells and verified irregular rest interventions are needed for the central claim.
'''
    (OUT/'reports/RAPCT_PRONY_FINAL_REPORT.md').write_text(text,encoding='utf-8')
    print(json.dumps(status,indent=2),flush=True)


if __name__=='__main__':generate()
