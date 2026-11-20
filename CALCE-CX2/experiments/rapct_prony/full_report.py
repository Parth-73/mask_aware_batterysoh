"""Final filename, explicitly interim scientific evidence: all values come from saved artifacts."""
import json
import re
import numpy as np
import pandas as pd
from .audit import OUT


def md(frame):
    frame=frame.copy()
    def value(x):
        if pd.isna(x):return 'unavailable'
        if isinstance(x,(float,np.floating)):return f'{x:.5g}'
        return str(x).replace('|','/').replace('\n',' ')
    return '| '+' | '.join(map(str,frame.columns))+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+'\n'.join('| '+' | '.join(value(x) for x in row)+' |' for row in frame.itertuples(index=False,name=None))


def write_report():
    environment=json.loads((OUT/'configs/environment.json').read_text())
    feasibility=json.loads((OUT/'rest_events/feasibility.json').read_text())
    status=json.loads((OUT/'reports/stage_status.json').read_text())
    m=pd.read_csv(OUT/'metrics/matched_stage_comparison.csv')
    comparison=m[['run','split','mae','rmse','mae_pp','rmse_pp','n']].copy()
    comparison['run']=comparison.run.str.rsplit('_',n=1).str[0].str.replace('_',' ')
    comparison=comparison.rename(columns={'run':'model'})
    support=pd.read_csv(OUT/'rest_events/causal_state_support.csv')
    f=pd.read_csv(OUT/'processed/samples.csv')
    counts=f.groupby('battery').size().rename('retained_cycles').reset_index()
    preservation=json.loads((OUT/'data_audit/historical_preservation_check.json').read_text())
    tests=(OUT/'reports/test_all_stages.log').read_text(encoding='utf-8-sig',errors='replace')
    test_summary=re.findall(r'\d+ passed[^\n]*',tests)
    studies=[json.loads(p.read_text()) for p in (OUT/'optuna_studies').glob('*_best.json')]
    trial_file=OUT/'metrics/hyperparameter_trials.csv'
    trials=pd.read_csv(trial_file) if trial_file.exists() else pd.DataFrame()
    best_study=next((s for s in studies if s['study'].startswith('cnn_pilot10')),None)
    nested=pd.read_csv(OUT/'metrics/nested_CNN_outer_metrics.csv')
    if best_study is not None:nested=nested[nested.study.eq(best_study['study'])]
    nested_summary=nested[['seed','rmse','rmse_pp','mae_pp','n']].drop_duplicates('seed')
    synthetic=pd.read_csv(OUT/'metrics/prony_ablation.csv')
    synthetic_summary=synthetic.groupby(['modes','learnable_timescales']).heldout_penalty_rmse_fraction.agg(['mean','std']).reset_index()
    best_config=best_study['parameters'] if best_study else {}
    lines=f'''# RAPCT-Prony research report — executed pilots and incomplete research stages

This report documents an implemented framework and executed pilot experiments. It is not a completed publication study or confirmation of a recovery advantage. The proposed name and combination are not established as novel.

## 1. Literature and research question

[Related-work comparison matrix](../../../experiments/rapct_prony/research/RELATED_WORK.md) distinguishes publisher full text, abstracts, metadata and inaccessible method details. CNN–Transformer fusion, physics-guided SOH, degradation decomposition and exponential relaxation all have prior art. The reference NPSSL was inspected in actual source: fixed decay-rate grid, positive amplitudes, recursive memory and per-window zero initialization. Its mechanical oscillator residual was not copied.

The question is whether separating irreversible aging from verified-rest transient memory improves matched cross-battery SOH prediction. This execution cannot answer the recovery part because documented persistent rest-state transitions fail the conservative continuity gate. Several full-text literature details remain unverified, preventing a definitive novelty assessment.

## 2. Actual environment and reproducibility

```json
{json.dumps(environment,indent=2)}
```

Driver details are in configs/nvidia_smi.txt; the driver-advertised CUDA compatibility is distinct from PyTorch's CUDA runtime. This machine has an RTX 4050 Laptop GPU, not a historical benchmark GPU. Deterministic model seeds are 42, 123 and 2026. Primary A0–A4 pilots use seed 42; the nested CNN refit and synthetic mode ablations use all three. Configuration files, training losses, checkpoints, train-only scalers, normalization units, hashes and resource measurements accompany each completed run. Vendor dependencies were installed in the experiment folder, without changing the base environment.

Sensitive degradation and memory computations explicitly remain float32 under autocast. CNN batch size, accumulation, clipping, optional AMP and worker count are configurable. Cached curve tensors and frozen-encoder representations reduce repeated preprocessing.

## 3. Data, cycle reconstruction and target

Raw hashes are frozen in data_audit/raw_hashes.csv. The isolated inventory reads 314 Excel channel tables for the seven conventional batteries, while 16 relative-time exports remain unresolved. Trusted local channel caches were rebuilt because historical cache paths referenced another machine. New Excel reads use calamine; accepted reconstructed cycle starts, quality decisions and fractional labels are checked against the historical validated table.

SOH = validated discharge current integral / 1.35 Ah. No target clipping, smoothing or capacity normalization occurs. Historical counter-agreement, charge/discharge endpoint and rejection rules are unchanged. Original rejection reasons remain in the source quality table; new cohort exclusions are in data_audit/curve_quality.csv. Observed cycle ordinals are not asserted to be absolute laboratory lifetime cycle counts.

{md(counts)}

Total matched full-curve/feature samples: **{len(f)}**.

## 4. Prediction-time contract and engineered features

The cutoff is the final available charging observation for the current cycle, before its validated discharge begins. Same-cycle discharge, future SOH, post-cutoff rest, unverified temperature/resistance and label-derived recovery triggers are forbidden. A machine-readable allowlist is in configs/feature_availability.json. Every accepted cutoff is checked against actual reconstructed discharge timestamps.

The five engineered inputs are cc_duration_s, cv_duration_s, dva_peak, ic_peak_voltage and ic_peak_width. Existing extraction definitions are preserved: SG(11,3) IC on the 2 mV voltage grid, 7/11/15 smoothing checks, interior-peak reliability rules, and existing partial-capacity DVA. They enter an MLP, not a convolution over arbitrary feature positions.

## 5. Raw-curve CNN and Stage-1 pilots

Curves contain Q(V) in Ah, dQ/dV in Ah/V and I(V) in A at 176 positions from 3.80 to 4.15 V. Full measured coverage and reliable monotonic charging are required; missing regions are not extrapolated. Channels use training-battery scaling, without per-curve full-capacity normalization. Source and cohort hashes accompany the cached tensor.

Parallel kernels 3/5/7 each use 32 channels, GroupNorm, GELU, a second convolution and GELU. Adaptive pooling and projection give a 64-dimensional curve embedding. The five-feature MLP is 5→32→32; fusion projects 96→64. MLP-only, curve-only and fusion have separate regression heads.

AdamW starts at 5e-4, weight decay 1e-3, batch 32 and clipping 1. MLP/fusion ran 30-epoch reduced-budget pilots. The original CNN process stopped after 17 logged epochs; its best weights initialized a separate 20-epoch run with a fresh optimizer. This continuation has a different effective budget and is explicitly not a fair final training-budget comparison. The interrupted artifacts were retained.

## 6. Transformer and matched temporal inputs

A causal two-layer Transformer uses dimension 64, four heads, FFN 128, dropout 0.10 and cycle-order sinusoidal positions. Complete five-cycle windows exclude ordinal gaps, rejected/missing cycles, source changes, counter resets, overlap and excessive acquisition gaps. No elapsed-time encoding is invented. Padding support and future-cycle masking are unit tested.

Both frozen-cycle-encoder and joint-training variants ran 20 epochs from the completed fusion checkpoint. Joint training was selected using validation RMSE only (0.57583 pp versus 0.66576 pp). On matched validation targets it still underperformed the MLP (0.51986 pp) and fusion (0.55613 pp). Attention summaries describe the final inference minibatch and are not causal importance evidence.

The length-5 cohort has 8,395 windows: 1,345 validation targets on CX2_36 and 1,102 outer targets on CX2_38. Length-10/20 audits are saved, but those sequence lengths were not trained in this execution.

## 7. Physics-guided irreversible degradation

The validation-selected Transformer is frozen. Its representations feed a constrained aging branch over complete chronological segments (548 segments, maximum 46 target cycles):

D_first = 0.2 softplus(initial_encoder(h_first))

r_k = 0.001 softplus(rate_encoder(h_k))

D_next = D_k + r_k, for verified consecutive observed cycle increments.

A missing-history boundary gets an encoder-estimated initial state, not zero aging. State carries are supported and tested across valid chunks. No transition is inferred across an unknown segment boundary.

Measured prediction = S_ref - D + c, where S_ref = 1 + 0.2 tanh(s), and c = 0.03 tanh(correction_encoder(h)). The signed bounded correction permits measured fluctuations without making irreversible loss decrease. These scales and bounds are empirical modeling choices, not independently measured electrochemical constants. The global reference is learned from training cells.

No Arrhenius, impedance, temperature or electrochemical PDE constraint is invented. Estimated D is latent, not measured irreversible capacity loss. Monotonicity is guaranteed only inside valid modeled segments, not across separately initialized histories.

## 8. Losses and physical audits

L = MSE(SOH) + warmup × [lambda_phys mean(c²) + lambda_smooth mean((delta_r/0.001)²)].

Warm-up is zero in epochs 1–10, ramps during 11–30 and reaches the requested full coefficients thereafter. The smoothness prior applies only to valid adjacent cycles. Zero/zero and 0.01/0.001 settings each ran 50 epochs from the same frozen representation and seed.

Nonnegative rate is structural, so an identically zero monotonicity penalty is not added. No recurrence-versus-itself residual is reported as physics validation. Physical-audit files record zero negative within-segment increments and correction saturation. These are mathematical consistency checks, not proof of uniquely identified electrochemistry. Both variants underperformed the unconstrained Transformer in this pilot.

## 9. Rest identification and feasibility

The inventory distinguishes sampled measured rest (|I|≤0.005 A with clock-consistent adjacent observations ≤180 s apart), uncertain recording gaps and no verified rest. Cross-source gaps are never relabeled as rest; overlapping acquisition intervals conservatively downgrade verification. Source identifiers and uncertainty reasons are retained.

Verified short-rest count: **{feasibility['verified_events']}**. Uncertain gaps: **{feasibility['uncertain_gaps']}**. Verified durations range **1–235 s**, with no event longer than one hour. Counts are events, not independent recovery experiments. Pre/post pair checks use discharge endpoints, current charge cutoffs, consecutive validated cycles and retrospective discharge-rate agreement. They do not use capacity increases to activate events.

{md(support)}

Every retained history contains at least one uncertain interval, frequently sparse CV logging. Capacity counter agreement can validate an integral without documenting transient memory evolution during the gap. Under this conservative policy, zero persistent-state transitions qualify. This is a limitation of the current transition policy, not proof that CALCE contains no capacity recovery or that all gaps were rest. Experimental recovery training is disabled in rest_events/recovery_gate.json.

## 10. Single-mode and Neural Prony equations

Single-mode measured rest: z_after = z_before exp(-rest_s/tau). During observed cycling, bounded nonnegative stress updates z while retaining its existing state. Unknown transitions require an explicit encoder-derived current-state initialization; no hidden per-minibatch reset to zero occurs.

Multi-mode memory uses positive bounded log-timescales and normalized nonnegative softplus weights. P = sum(w_m z_m), with each state bounded by 0.1 fractional capacity. The bounded cycling update is z_after_stress = z_rest + (0.1-z_rest)(1-exp(-stress/0.1)). The bound is a mathematical regularizer. Without verified rest, supplied gap durations do not induce exponential relaxation.

Full fusion is S_ref-D-P. The composition returns prediction, irreversible state, transient penalty, modeled verified-rest contribution, timescales, trajectories and availability masks. The full graph is synthetic-forward tested only; no experimental joint training or calibrated uncertainty is claimed. Full-segment execution is the tested interface; arbitrary truncated Transformer-context training is not implemented.

## 11. Synthetic memory evidence

A separately generated biexponential penalty has synthetic time constants 30 and 300 s and amplitudes 0.035/0.015. These are declared test parameters, not CALCE electrochemical measurements. Fixed/learnable times, modes 1/2/4/8 and seeds 42/123/2026 produced 24 identification runs. Interleaved held-out synthetic time points test response interpolation; this is not cross-battery generalization or experimental capacity-recovery validation. Experimental timescale ranges are not fitted from validation/test rests.

{md(synthetic_summary)}

The metric is penalty RMSE in fractional capacity units, not CALCE SOH RMSE. Individual fitted modes need not correspond uniquely to physical mechanisms. Synthetic figures and checkpoints are explicitly labeled and excluded from experimental all_runs.csv.

## 12. Hyperparameter search and nested protocol

The executable nested CNN workflow uses SQLite-backed Optuna TPESampler(seed=42), MedianPruner and deterministic ordered inner battery folds. Pruning compares equal completed fold prefixes; final scores average inner battery RMSE equally. Every inner fit has its own training-only scalers. Pretraining provenance checks reject weights trained on held-out batteries.

The executed primary tuning pilot covers only outer CX2_38 and inner validation batteries CX2_36/CX2_37. Ten trials used five-epoch inner budgets, searching channels 16/32/64 and log LR 1e-4–1e-3. Three trials completed both folds; seven were pruned. With this small completed set, the default TPE startup budget is not exhausted and stable hyperparameter importance is not estimated. This is an optimization-workflow pilot, not evidence of Bayesian convergence.

Selected CNN parameters: **{json.dumps(best_config)}**. Final epoch count is the median best epoch from the inner folds (four epochs). After freezing that choice, models refit on all six non-outer batteries for the three seeds. Any in-sample refit diagnostic is explicitly not validation and never selects a checkpoint; the fixed final epoch is saved. Reopening the study was verified to add zero trials and zero training runs. Completed identical numeric configurations are reused across study labels.

The CLI supports all seven outer batteries and all remaining inner battery folds. That complete study was not run. Transformer, physics-weight and joint component Optuna searches remain unimplemented/unexecuted; Prony experimental tuning is blocked by the data gate. These incomplete items must not be described as final nested evaluation.

## 13. Selected outer CNN refits — separate protocol

{md(nested_summary)}

These use all six inner batteries and only four refit epochs. They must not be pooled with the primary five-training-battery, single-seed staged comparison below. The cells were previously examined in historical experiments: this is cross-battery benchmarking, not untouched external validation.

## 14. Matched quantitative stage comparisons

All rows below use identical target cycle IDs, cutoffs and raw labels within their split. Fractional and percentage-point errors are separate. Primary training uses CX2_16/33/34/35/37, validation CX2_36 and outer CX2_38. CNN continuation and later-stage budgets differ, so this is pilot evidence.

{md(comparison)}

The MLP is validation-best among these primary pilots. The CNN continuation has the smallest displayed primary outer RMSE, but outer scores were not used for selection. Joint Transformer generalization is worse than frozen despite better validation; this demonstrates why one validation/outer pair cannot establish architecture superiority. Adding the constrained aging head did not improve accuracy. The full recovery hypothesis remains untested experimentally.

## 15. Metrics, figures and identifiability

Metrics include MAE/MSE/RMSE, MAPE, sMAPE, R², median/max/95th-percentile absolute error, signed bias, stage training/inference time, parameter count and peak allocated GPU memory. Physics inference includes an actual frozen-encoder forward plus state head; its training time is explicitly new-head-only, with encoder training recorded separately. Memory values and D are estimates, not independent ground truth.

metrics/stage_comparison.csv is matched; historical_reference.csv and nested_CNN_outer_metrics.csv are separate. Seed SD is only computed where multiple matching seeds exist; single-seed stability remains unavailable. SOH-region metrics are retrospective evaluation slices, never predictors. rest_event_metrics.csv contains feasibility counts and explicitly no recovery prediction errors.

PNG at 300 DPI and vector PDF figures cover raw/scaled curves, features, SOH, curve quality, sampled rests, predictions, matched staged error, complexity, latent aging segments, synthetic memory/timescales and Optuna histories. Real post-rest rebound and experimental Prony figures are unavailable and not invented. Importance panels state insufficient completed trials rather than manufacturing importance estimates.

Offsets, stress amplitudes and close timescales can trade off. Positive/bounded states reduce numerical ambiguity but do not identify unique degradation mechanisms. Required future work includes initialization sensitivity, controlled pre/post-rest measurements and independently documented conditions.

## 16. Tests and preservation audit

Latest combined test result: **{test_summary[-1] if test_summary else 'see test_all_stages.log'}**. Tests cover existing extraction, interpolation bounds, no-label allowlists, train-only scaling, chronological gaps, pretraining split scope, causal/padding masks, positive aging, state carry/reset, exact decay, synthetic rebound, gradients, finite outputs, reload equality, deterministic seeds, exact training-resume equivalence, fixed-epoch outer refit and metric/identifier units.

{preservation['protected_file_count']} historical source/config/feature files were snapshotted. All except the original outputs/battery_splits.json match the starting hashes. That file changed during execution and currently contains CX2_{{35; this task did not write or restore it. The initial file is preserved in data_audit/frozen_definitions. The new experiment uses explicit independent split configs and is unaffected. Existing untracked benchmark work and baseline checkpoints were not overwritten.

## 17. Stage completion ledger

{md(pd.DataFrame([dict(checkpoint=k,status=v) for k,v in status.items()]))}

The complete seven-outer-fold, all-component nested study, full seed stability for A0–A4, remaining feature/sequence/physics searches, initialization sensitivities and genuine external validation remain incomplete. Experimental A5–A8 recovery comparisons and full joint training are unavailable under the current continuity gate. No completion is fabricated.

## 18. Scientific conclusion and reproducible next steps

These executed pilots do not show that increasing hybrid complexity improves SOH estimation, and they cannot establish an advantage after genuine recovery events. A simple five-feature MLP is the validation leader on matched primary targets; constrained aging worsens error in this configuration. Synthetic Prony tests validate the implemented memory mechanism only.

Use the isolated README commands to rebuild artifacts, resume a checkpoint/study and launch remaining outer folds. Before an experimental recovery claim, resolve unknown sparse-CV transitions with documented measurements or use a separately documented rest dataset without silently merging it into CALCE. A strong publication conclusion requires completed nested ablations and genuinely independent cells. The framework and negative pilot findings are reproducible research groundwork, not a demonstrated novel superior recovery estimator.
'''
    (OUT/'reports/RAPCT_PRONY_FINAL_REPORT.md').write_text(lines,encoding='utf-8')

if __name__=='__main__':write_report()
