"""Tables and 300-DPI/vector figures from completed runs only."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .runtime import atomic_csv
from .data import FEATURES
from ..training.config import ROOT, json_write

METRICS = ['mae_fraction', 'rmse_fraction', 'mse_fraction_squared', 'mape_percent', 'smape_percent',
           'r2', 'mae_percentage_points', 'rmse_percentage_points', 'training_seconds',
           'inference_ms_per_sample', 'parameter_count', 'peak_gpu_memory_mb']
KEYS = ['architecture', 'feature_set', 'length', 'experiment', 'physics_weight', 'reg_weight', 'matched_targets']


def summarize(frame, keys):
    out = frame.groupby(keys, dropna=False)[METRICS].agg(['mean', 'std', 'count'])
    out.columns = ['_'.join(x) for x in out.columns]
    return out.reset_index()


def save(fig, output, name):
    fig.tight_layout()
    fig.savefig(output/f'{name}.png', dpi=300, bbox_inches='tight')
    fig.savefig(output/f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)


def bars(data, group, metric, output, name, title):
    if data.empty:
        return
    g = data.groupby(group)[metric].agg(['mean', 'std'])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(range(len(g)), g['mean'], yerr=g['std'].fillna(0), capsize=3, color='#267b91')
    ax.set_xticks(range(len(g)), g.index.astype(str), rotation=20, ha='right')
    ax.set(ylabel=metric.replace('_', ' '), title=title)
    save(fig, output, name)


def generate(output):
    output = Path(output)
    runs = [json.loads(p.read_text()) for p in sorted((output/'runs').glob('*/result.json'))]
    if not runs:
        raise ValueError('No completed experiments; no result can be reported')
    data = pd.DataFrame(runs)
    config = json.loads((output/'experiment_manifest.json').read_text())['config']
    for name in ['metrics', 'reports', 'figures']:
        (output/name).mkdir(exist_ok=True)
    tables, figures = output/'metrics', output/'figures'
    atomic_csv(data, tables/'all_runs.csv')
    atomic_csv(data, tables/'per_battery_metrics.csv')
    cross = data[data.fold.ne('historical')]
    # First average seeds per battery. Across-battery summaries give each cell equal weight.
    per_fold = cross.groupby(KEYS+['fold'], dropna=False)[METRICS].mean().reset_index()
    cv = summarize(per_fold, KEYS)
    atomic_csv(cv, tables/'cross_validation_summary.csv')
    atomic_csv(summarize(cross, KEYS+['fold']), tables/'seed_stability.csv')
    atomic_csv(cv[cv.experiment.isin(['primary', 'feature_ablation']) & cv.architecture.eq('lstm')], tables/'feature_ablation.csv')
    atomic_csv(cv[cv.architecture.str.endswith('_pinn')], tables/'physics_ablation.csv')
    atomic_csv(cv[cv.experiment.eq('sequence_ablation')], tables/'sequence_ablation.csv')
    complexity = data[['configuration_id', 'dataset', 'fold', 'seed']+KEYS+['parameter_count', 'training_seconds', 'inference_ms_per_sample', 'peak_gpu_memory_mb']]
    atomic_csv(complexity, tables/'model_complexity.csv')
    primary = cross[cross.experiment.eq('primary')]
    final = cv[cv.experiment.eq('primary')].copy()
    atomic_csv(final, tables/'final_comparison.csv')
    regions = pd.concat([pd.read_csv(p) for p in (output/'runs').glob('*/regions.csv')], ignore_index=True)
    atomic_csv(regions, tables/'soh_region_metrics.csv')
    if data.fold.eq('historical').any():
        atomic_csv(data[data.fold.eq('historical')], tables/'historical_reference.csv')
    # The recommendation uses only inner-validation errors, never outer errors.
    recommendation_scores = primary.groupby('architecture').validation_rmse_fraction.agg(['mean', 'std', 'count']).sort_values(['mean'], kind='stable')
    atomic_csv(recommendation_scores.reset_index(), tables/'validation_recommendation.csv')
    best = recommendation_scores.index[0]
    primary_ids = set(primary.configuration_id)
    predictions = [pd.read_csv(output/'predictions'/f'{ident}_test.csv') for ident in sorted(primary_ids)]
    prediction = pd.concat(predictions, ignore_index=True)
    atomic_csv(prediction, output/'predictions/primary_outer_predictions.csv')
    expected = json.loads((output/'planned_runs.json').read_text())
    complete = len(data)==len(expected)
    # Uncertainty: seed SD is reported per fold, not mislabeled as a population CI.
    for battery, g in prediction.groupby('battery'):
        truth = g[['cycle_id', 'lifetime_cycle', 'actual_soh']].drop_duplicates().sort_values('lifetime_cycle')
        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.plot(truth.lifetime_cycle, truth.actual_soh*100, color='black', label='Validated capacity SOH', linewidth=1.5)
        for architecture, h in g.groupby('architecture'):
            h = h.groupby('lifetime_cycle').predicted_soh.agg(['mean', 'std']).reset_index()
            ax.plot(h.lifetime_cycle, h['mean']*100, label=architecture, linewidth=1)
            ax.fill_between(h.lifetime_cycle, (h['mean']-h['std'])*100, (h['mean']+h['std'])*100, alpha=.12)
        ax.set(xlabel='Reconstructed observed cycle ordinal', ylabel='SOH (%)', title=f'{battery}: outer predictions (seed mean ± sample SD)')
        ax.legend(fontsize=8)
        save(fig, figures, f'trajectory_{battery}')
    for metric in ['rmse_percentage_points', 'mae_percentage_points']:
        bars(primary, 'architecture', metric, figures, f'comparison_{metric}', 'Cross-battery primary comparison')
    fig, ax = plt.subplots(figsize=(7, 5))
    for architecture, g in prediction.groupby('architecture'):
        # Plot a deterministic thinning only; metrics always use every sample.
        h = g.iloc[::max(1, len(g)//2000)]
        ax.scatter(h.actual_soh*100, h.predicted_soh*100, s=4, alpha=.25, label=architecture)
    lo, hi = prediction.actual_soh.min()*100, prediction.actual_soh.max()*100
    ax.plot([lo, hi], [lo, hi], 'k--')
    ax.set(xlabel='Validated SOH (%)', ylabel='Predicted SOH (%)', title='Outer-fold prediction agreement')
    ax.legend()
    save(fig, figures, 'predicted_vs_actual')
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for architecture, g in prediction.groupby('architecture'):
        ax.hist(g.signed_error*100, bins=80, density=True, histtype='step', label=architecture)
    ax.set(xlabel='Prediction minus actual (SOH percentage points)', ylabel='Density', title='Outer-fold residual distributions')
    ax.legend()
    save(fig, figures, 'residual_distributions')
    pivot = primary.groupby(['fold', 'architecture']).rmse_percentage_points.mean().unstack()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(pivot, aspect='auto', cmap='viridis')
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=20)
    ax.set_yticks(range(len(pivot)), pivot.index)
    ax.set_title('Battery generalization: seed-mean RMSE')
    fig.colorbar(im, ax=ax, label='SOH percentage points')
    save(fig, figures, 'cross_battery_generalization')
    for experiment, group, name in [('feature_ablation', 'feature_set', 'feature_ablation'),
                                    ('sequence_ablation', 'length', 'sequence_ablation')]:
        subset = cross[(cross.experiment.eq(experiment)) | ((experiment=='feature_ablation') & cross.experiment.eq('primary') & cross.architecture.eq('lstm'))]
        bars(subset, group, 'rmse_percentage_points', figures, name, f'{experiment}: matched eligible targets')
    physics = cross[cross.architecture.str.endswith('_pinn')].copy()
    physics['label'] = physics.architecture + ' λ=' + physics.physics_weight.astype(str)
    bars(physics, 'label', 'rmse_percentage_points', figures, 'physics_ablation', 'Shared latent-aging regularizer ablation')
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for architecture, g in primary.groupby('architecture'):
        ax.scatter(g.parameter_count.mean(), g.rmse_percentage_points.mean(), label=architecture)
    ax.set(xlabel='Trainable parameters', ylabel='Outer RMSE (SOH percentage points)', title='Model complexity versus cross-battery accuracy')
    ax.legend()
    save(fig, figures, 'complexity_accuracy')
    rg = regions[regions.configuration_id.isin(primary_ids) & regions.n_samples.gt(0)]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for a, g in rg.groupby('architecture'):
        h = g.groupby('region').rmse_percentage_points.mean().reindex(['above_90', '80_to_90', 'below_80'])
        ax.plot(range(3), h, marker='o', label=a)
    ax.set_xticks(range(3), ['SOH >90%', '80–90%', '<80%'])
    ax.set(ylabel='RMSE (SOH percentage points)', title='Degradation-region performance (counts in CSV)')
    ax.legend()
    save(fig, figures, 'degradation_regions')
    # All learning curves are saved individually with their run identity.
    for r in data.itertuples():
        h = pd.read_csv(output/'training_logs'/f'{r.configuration_id}.csv')
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
        axes[0].plot(h.epoch, h.data_loss, label='Train data MSE')
        axes[0].plot(h.epoch, h.validation_mse, label='Inner validation MSE')
        axes[0].set(xlabel='Epoch', ylabel='SOH fraction squared', title=f'{r.architecture}: {r.fold}, seed {r.seed}')
        axes[0].legend(fontsize=7)
        axes[1].plot(h.epoch, h.physics_loss, label='Monotonicity penalty')
        axes[1].plot(h.epoch, h.regularization_loss, label='Transient correction penalty')
        axes[1].set(xlabel='Epoch', ylabel='Unweighted loss (fraction squared)', title='Shared physics components')
        axes[1].legend(fontsize=7)
        save(fig, figures, f'learning_{r.configuration_id}')
    counts = pd.read_csv(output/'data_audit/cycle_counts.csv')
    windows = pd.read_csv(output/'splits/windows_L5_matchedFalse.csv')
    environment = json.loads((output/'environment.json').read_text())
    hardware = json.loads((output/'hardware_info.json').read_text())
    raw = json.loads((ROOT/'outputs/run_metadata.json').read_text())
    machine = {'status': config['status'], 'completed_runs': len(data), 'planned_runs': len(expected),
               'matrix_complete': complete, 'full_200_epoch_protocol_completed': complete and config['epochs']==200,
               'eligible_batteries': counts.battery.tolist(), 'eligible_windows_L5': len(windows),
               'final_comparison': json.loads(final.to_json(orient='records')),
               'validation_recommendation': best, 'rest_verified': False}
    json_write(output/'reports/final_comparison.json', machine)
    show = final[['architecture', 'mae_percentage_points_mean', 'rmse_percentage_points_mean', 'mape_percent_mean', 'smape_percent_mean', 'r2_mean']]
    text = f'''# Four-model CALCE CX2 baseline comparison

## 1. Dataset information

Status: **{config['status']}**. Completed {len(data)} of {len(expected)} planned runs in this budget. Full 200-epoch protocol completed: {machine['full_200_epoch_protocol_completed']}. Smoke outputs are not scientific benchmark evidence.

Historical preprocessing inventory: {raw['raw_files']} raw files, {raw['batteries_inventoried']} inventoried cells, {raw['candidates']} candidate cycles, {raw['validated_complete']} validated complete, {raw['conventional_validated']} conventional validated. These counts describe frozen preprocessing exports, not a newly executed reconstruction. Local raw hash coverage is reported separately in reproducibility_manifest.json.

Seven eligible conventional batteries; {len(windows)} eligible length-5 targets on the common nine-feature complete-case population. Per-battery counts:

{counts.to_string(index=False)}

## 2. Computational environment

Python: {environment['python']}. PyTorch: {environment['packages']['torch']}. CUDA runtime: {hardware['torch_cuda_runtime']}. Device: {hardware.get('gpu', 'CPU')}. VRAM bytes: {hardware.get('vram_bytes')}. Exact packages, driver report, precision, deterministic flags, hashes and commit are stored in the environment/hardware/reproducibility manifests. FP32 only; no AMP.

## 3. Data preprocessing

Reused and audited source reconstruction: current threshold 0.02 A, within-step trapezoidal integration with bounded initial rectangular interval, agreement with endpoint counter differences within max(0.01 Ah, 5%), verified charge/discharge endpoints, explicit rejection reasons. SOH = validated discharge Ah / 1.35 Ah, fractional, unclipped and unsmoothed. Quality/feature/rejection tables cross-checked. Source-same, strictly consecutive original ordinals, no rejected or missing observations, no time overlap and inter-cycle gaps <=3600 seconds. These gaps are continuity gates, never rest labels. The target is the final current cycle's eventual discharge SOH, estimated from charge measurements available through that cycle. No next-cycle forecasting.

## 4. Feature engineering

Primary exact order: {FEATURES['five']}. CC/CV active duration in seconds uses 4.18 V threshold and verified CV current decay; rests excluded. DVA peak in V/Ah uses measured 3.8–4.0 V partial-charge interval and 151-point capacity grid. IC peak position/width in V use 3.80–4.15 V on a 0.002 V grid, cubic Savitzky–Golay smoothing (11 points, with 7/15 stability checks), reliable interior peaks and width at half prominence. Definitions are unchanged. Three/nine-feature alternatives use identical common complete-case targets. Five features are prespecified, not claimed universally optimal. No resistance, temperature, discharge-derived input, battery/cycle ID, target, or split label is encoded. Rest-duration evidence is insufficient: NaN, no zero imputation, no rest-event metric or six-feature experiment.

## 5. Model architectures and physical hypotheses

Vanilla LSTM: one layer, 64 hidden units, final chronological representation and linear output. Transformer: linear 64-dimensional embedding, sinusoidal chronological positions, four heads, two layers, feed-forward 128, dropout 0.1, final-token regression. All attention tokens are available through prediction cutoff. CNN: time-axis Conv1D 32/64 channels, kernel 3, padding 1, GELU and temporal mean pooling.

Both PINNs use the identical head: D=softplus(h_D), c=0.03*tanh(h_c), prediction=1-D+c. D is nonnegative. L_data=mean((prediction-y)^2); L_phys=mean(ReLU(D_previous-D_current)^2) over verified consecutive target-window pairs; L_reg=mean(c^2). L_total=L_data+0.1*L_phys+0.1*L_reg. Zero-weight ablations set BOTH coefficients to zero while preserving architecture and head. Paired samples use earlier charge measurements only; no future data in individual inference. This is a soft aging hypothesis, **not a validated electrochemical law**. D is not guaranteed globally nondecreasing and decomposition is not identifiable from SOH alone. The residual is bounded at three percentage points and penalized; it cannot arbitrarily bypass the latent branch. The 1-D reference assumes initial irreversible SOH 1 and can introduce bias, especially SOH >1. No Neural Prony model is implemented.

## 6. Experimental methodology

Outer leave-one-battery-out ordered cells: CX2_16,33,34,35,36,37,38. Inner validation is the next listed cell, wrapping to CX2_16. Five remaining cells train. Historical split is separately labeled when executed (train 16/33/34/35/37, validation 36, test 38). All cells have prior exploratory exposure; these are cross-battery benchmarks, not untouched external validation. CX2_3/8 transfer remains unverified, CX2_32 excluded. Train-only StandardScaler on eligible inner-training cycles, unscaled target, complete cases. Length 5 is primary; matched length-20 targets for separate length 5/10/20 ablations.

AdamW lr 0.0005, decay 0.001, batch 64, clipping 1.0. ReduceLROnPlateau factor 0.5, patience 5, floor 1e-6. Early stopping patience 20, minimum inner-validation MSE, seed 42/123/2026. Executed epoch budget: {config['epochs']}; full requested maximum: 200. Fixed coefficient singleton (0.1,0.1) predetermined; no tuning against outer outcomes. Optimizer/scheduler and Python/NumPy/CPU/CUDA RNG states are saved for resume. Configuration hashes include code, dataset, runtime and budget. Incompatible output directories are rejected. Atomic files protect completed outputs.

## 7. Quantitative performance

Outer metrics are first averaged over seeds within battery, then equally across seven batteries. Columns below are SOH percentage points for MAE/RMSE, relative percent for MAPE/sMAPE. Complete fractional MAE/MSE/RMSE and R2 are in metrics/final_comparison.csv. MAPE denominator=max(abs(actual),1e-8); sMAPE=200*mean(abs(error)/max(abs(actual)+abs(prediction),1e-8)). Bias is prediction minus actual. Seed SD uses ddof=1 and is reported per fold; battery variability is separately reported in cross_validation_summary.csv. Overlapping windows are not independent replicates. GPU inference timing is synchronized and excludes host transfer; training includes checkpoint overhead. Device peak allocation is measured during training.

{show.to_string(index=False)}

## 8. Feature-set ablations

Three/five/nine inputs are compared on vanilla LSTM, common nine-feature complete-case windows, the same folds and seeds. This isolates feature information in one architecture; no claim of universality across architectures. See metrics/feature_ablation.csv. Empty rows mean unexecuted experiments, not zero error.

## 9. Physics ablations

See metrics/physics_ablation.csv and per-epoch separate unweighted physics/correction terms. Both encoders share definitions and hypotheses. Constraints may help or hurt; no superiority is assumed. Zero weights retain positivity, bounded correction and decomposition inductive bias; the ablation isolates the loss penalties, not all head assumptions.

## 10. Generalization and limitations

Inspect per-battery and SOH-region metrics with subgroup counts, maximum/median/95th percentile errors and signed bias in all_runs.csv. Tiny subgroups are descriptive only. Seven explored cells and pooled conventional 0.4C/0.5C/1C capacities limit external validity. Different discharge rates can confound capacity SOH despite exclusion from predictors. Missingness and source-boundary gates select a conservative cohort. Prior feature selection included some validation cells. No temperature, impedance, calendar-aging law or verified rest information is supported. Seeds are algorithmic replicates, not seven additional independent battery samples. No population confidence interval or superiority significance is inferred. Reduced budgets cannot establish converged model ranking. Full raw re-extraction and instrument calibration were not repeated.

## 11. Conclusions and execution completeness

Predetermined validation-only recommendation in this executed budget: **{best}** (mean inner-validation RMSE across folds and seeds). This is a budget-specific recommendation, not a winner chosen from outer-test error or one seed. It must be reassessed using the full prespecified training budget if this run is reduced. Outer metrics describe the fixed four configurations without feeding design changes. Full machine-readable status: reports/final_comparison.json. Completed checkpoints: checkpoints/*.pt; restart states: runs/*/resume.pt; metrics: metrics/*.csv; predictions preserve original cycle IDs and all window IDs under predictions/ and splits/. Unexecuted configurations have no fabricated metrics. Rest duration remains unavailable. Future work may verify rest events and implement a separate Neural Prony architecture; neither is claimed here.
'''
    (output/'reports/BASELINE_COMPARISON_REPORT.md').write_text(text, encoding='utf-8')
    json_write(output/'reports/artifact_inventory.json', {'completed_runs': len(data), 'planned_runs': len(expected),
        'png_count': len(list(figures.glob('*.png'))), 'pdf_count': len(list(figures.glob('*.pdf'))), 'dpi': 300})
    print(show.to_string(index=False), flush=True)
    print('Report:', output/'reports/BASELINE_COMPARISON_REPORT.md', flush=True)
