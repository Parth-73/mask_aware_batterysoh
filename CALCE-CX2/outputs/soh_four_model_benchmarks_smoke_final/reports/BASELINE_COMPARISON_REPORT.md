# Four-model CALCE CX2 baseline comparison

## 1. Dataset information

Status: **smoke**. Completed 4 of 4 planned runs in this budget. Full 200-epoch protocol completed: False. Smoke outputs are not scientific benchmark evidence.

Historical preprocessing inventory: 745 raw files, 12 inventoried cells, 25985 candidate cycles, 15822 validated complete, 13736 conventional validated. These counts describe frozen preprocessing exports, not a newly executed reconstruction. Local raw hash coverage is reported separately in reproducibility_manifest.json.

Seven eligible conventional batteries; 8395 eligible length-5 targets on the common nine-feature complete-case population. Per-battery counts:

battery  candidates  validated  complete_case_eligible
 CX2_16        2016       1939                    1899
 CX2_33        1700       1618                    1488
 CX2_34        1736       1657                    1554
 CX2_35        1814       1735                    1630
 CX2_36        1971       1893                    1743
 CX2_37        1281       1236                    1219
 CX2_38        1960       1877                    1477

## 2. Computational environment

Python: 3.14.6 | packaged by Anaconda, Inc. | (main, Jul  9 2026, 14:29:05) [MSC v.1942 64 bit (AMD64)]. PyTorch: 2.11.0+cu128. CUDA runtime: 12.8. Device: NVIDIA GeForce RTX 4050 Laptop GPU. VRAM bytes: 6438780928. Exact packages, driver report, precision, deterministic flags, hashes and commit are stored in the environment/hardware/reproducibility manifests. FP32 only; no AMP.

## 3. Data preprocessing

Reused and audited source reconstruction: current threshold 0.02 A, within-step trapezoidal integration with bounded initial rectangular interval, agreement with endpoint counter differences within max(0.01 Ah, 5%), verified charge/discharge endpoints, explicit rejection reasons. SOH = validated discharge Ah / 1.35 Ah, fractional, unclipped and unsmoothed. Quality/feature/rejection tables cross-checked. Source-same, strictly consecutive original ordinals, no rejected or missing observations, no time overlap and inter-cycle gaps <=3600 seconds. These gaps are continuity gates, never rest labels. The target is the final current cycle's eventual discharge SOH, estimated from charge measurements available through that cycle. No next-cycle forecasting.

## 4. Feature engineering

Primary exact order: ['cc_duration_s', 'cv_duration_s', 'dva_peak', 'ic_peak_voltage', 'ic_peak_width']. CC/CV active duration in seconds uses 4.18 V threshold and verified CV current decay; rests excluded. DVA peak in V/Ah uses measured 3.8–4.0 V partial-charge interval and 151-point capacity grid. IC peak position/width in V use 3.80–4.15 V on a 0.002 V grid, cubic Savitzky–Golay smoothing (11 points, with 7/15 stability checks), reliable interior peaks and width at half prominence. Definitions are unchanged. Three/nine-feature alternatives use identical common complete-case targets. Five features are prespecified, not claimed universally optimal. No resistance, temperature, discharge-derived input, battery/cycle ID, target, or split label is encoded. Rest-duration evidence is insufficient: NaN, no zero imputation, no rest-event metric or six-feature experiment.

## 5. Model architectures and physical hypotheses

Vanilla LSTM: one layer, 64 hidden units, final chronological representation and linear output. Transformer: linear 64-dimensional embedding, sinusoidal chronological positions, four heads, two layers, feed-forward 128, dropout 0.1, final-token regression. All attention tokens are available through prediction cutoff. CNN: time-axis Conv1D 32/64 channels, kernel 3, padding 1, GELU and temporal mean pooling.

Both PINNs use the identical head: D=softplus(h_D), c=0.03*tanh(h_c), prediction=1-D+c. D is nonnegative. L_data=mean((prediction-y)^2); L_phys=mean(ReLU(D_previous-D_current)^2) over verified consecutive target-window pairs; L_reg=mean(c^2). L_total=L_data+0.1*L_phys+0.1*L_reg. Zero-weight ablations set BOTH coefficients to zero while preserving architecture and head. Paired samples use earlier charge measurements only; no future data in individual inference. This is a soft aging hypothesis, **not a validated electrochemical law**. D is not guaranteed globally nondecreasing and decomposition is not identifiable from SOH alone. The residual is bounded at three percentage points and penalized; it cannot arbitrarily bypass the latent branch. The 1-D reference assumes initial irreversible SOH 1 and can introduce bias, especially SOH >1. No Neural Prony model is implemented.

## 6. Experimental methodology

Outer leave-one-battery-out ordered cells: CX2_16,33,34,35,36,37,38. Inner validation is the next listed cell, wrapping to CX2_16. Five remaining cells train. Historical split is separately labeled when executed (train 16/33/34/35/37, validation 36, test 38). All cells have prior exploratory exposure; these are cross-battery benchmarks, not untouched external validation. CX2_3/8 transfer remains unverified, CX2_32 excluded. Train-only StandardScaler on eligible inner-training cycles, unscaled target, complete cases. Length 5 is primary; matched length-20 targets for separate length 5/10/20 ablations.

AdamW lr 0.0005, decay 0.001, batch 64, clipping 1.0. ReduceLROnPlateau factor 0.5, patience 5, floor 1e-6. Early stopping patience 20, minimum inner-validation MSE, seed 42/123/2026. Executed epoch budget: 2; full requested maximum: 200. Fixed coefficient singleton (0.1,0.1) predetermined; no tuning against outer outcomes. Optimizer/scheduler and Python/NumPy/CPU/CUDA RNG states are saved for resume. Configuration hashes include code, dataset, runtime and budget. Incompatible output directories are rejected. Atomic files protect completed outputs.

## 7. Quantitative performance

Outer metrics are first averaged over seeds within battery, then equally across seven batteries. Columns below are SOH percentage points for MAE/RMSE, relative percent for MAPE/sMAPE. Complete fractional MAE/MSE/RMSE and R2 are in metrics/final_comparison.csv. MAPE denominator=max(abs(actual),1e-8); sMAPE=200*mean(abs(error)/max(abs(actual)+abs(prediction),1e-8)). Bias is prediction minus actual. Seed SD uses ddof=1 and is reported per fold; battery variability is separately reported in cross_validation_summary.csv. Overlapping windows are not independent replicates. GPU inference timing is synchronized and excludes host transfer; training includes checkpoint overhead. Device peak allocation is measured during training.

architecture  mae_percentage_points_mean  rmse_percentage_points_mean  mape_percent_mean  smape_percent_mean      r2_mean
    cnn_pinn                   57.457823                    57.465949          62.043191           89.949041 -4264.154665
        lstm                   76.629000                    76.629536          82.756938          141.187069 -7583.134351
   lstm_pinn                   60.409202                    60.415956          65.231258           96.806875 -4713.296819
 transformer                   24.240849                    24.353199          26.201560           23.140343  -764.994496

## 8. Feature-set ablations

Three/five/nine inputs are compared on vanilla LSTM, common nine-feature complete-case windows, the same folds and seeds. This isolates feature information in one architecture; no claim of universality across architectures. See metrics/feature_ablation.csv. Empty rows mean unexecuted experiments, not zero error.

## 9. Physics ablations

See metrics/physics_ablation.csv and per-epoch separate unweighted physics/correction terms. Both encoders share definitions and hypotheses. Constraints may help or hurt; no superiority is assumed. Zero weights retain positivity, bounded correction and decomposition inductive bias; the ablation isolates the loss penalties, not all head assumptions.

## 10. Generalization and limitations

Inspect per-battery and SOH-region metrics with subgroup counts, maximum/median/95th percentile errors and signed bias in all_runs.csv. Tiny subgroups are descriptive only. Seven explored cells and pooled conventional 0.4C/0.5C/1C capacities limit external validity. Different discharge rates can confound capacity SOH despite exclusion from predictors. Missingness and source-boundary gates select a conservative cohort. Prior feature selection included some validation cells. No temperature, impedance, calendar-aging law or verified rest information is supported. Seeds are algorithmic replicates, not seven additional independent battery samples. No population confidence interval or superiority significance is inferred. Reduced budgets cannot establish converged model ranking. Full raw re-extraction and instrument calibration were not repeated.

## 11. Conclusions and execution completeness

Predetermined validation-only recommendation in this executed budget: **transformer** (mean inner-validation RMSE across folds and seeds). This is a budget-specific recommendation, not a winner chosen from outer-test error or one seed. It must be reassessed using the full prespecified training budget if this run is reduced. Outer metrics describe the fixed four configurations without feeding design changes. Full machine-readable status: reports/final_comparison.json. Completed checkpoints: checkpoints/*.pt; restart states: runs/*/resume.pt; metrics: metrics/*.csv; predictions preserve original cycle IDs and all window IDs under predictions/ and splits/. Unexecuted configurations have no fabricated metrics. Rest duration remains unavailable. Future work may verify rest events and implement a separate Neural Prony architecture; neither is claimed here.
