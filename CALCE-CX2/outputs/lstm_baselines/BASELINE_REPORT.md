# CALCE CX2 LSTM baseline report

Completed 45 full seeded training runs; no hybrid model, physical loss, CNN or GWO was used.

Validation-selected configuration: **B1 (Vanilla), F3, length 5**.

| architecture | feature_set | sequence_length | rmse_fraction_mean | rmse_fraction_std | mae_fraction_mean | mae_fraction_std | mape_percent_mean | r2_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B1 | F3 | 5 | 0.00546748 | 0.000459733 | 0.00405817 | 0.000372305 | 0.578857 | 0.998035 |

The table reports means and sample standard deviations across seeds 42, 123 and 2026. SOH and fractional errors use [0,1] units; multiply MAE/RMSE by 100 for percentage points. MAPE is relative error expressed as percent, not percentage points.

## Battery split and dataset integrity

| split | batteries |
| --- | --- |
| train | CX2_16, CX2_33, CX2_34, CX2_35, CX2_37 |
| validation | CX2_36 |
| test | CX2_38 |

Labels were cross-checked against cycle_quality.csv and rejected_cycles.csv, including charge/discharge integration-counter agreement and SOH = validated discharge Ah / 1.35. Only complete, routine/conventional records were eligible. No raw file or earlier feature output was modified.

## Sample counts

| split | battery | sequence_length | samples |
| --- | --- | --- | --- |
| train | CX2_16 | 5 | 1507 |
| train | CX2_33 | 5 | 1116 |
| train | CX2_34 | 5 | 1125 |
| train | CX2_35 | 5 | 1233 |
| train | CX2_37 | 5 | 967 |
| validation | CX2_36 | 5 | 1345 |
| test | CX2_38 | 5 | 1102 |
| train | CX2_16 | 10 | 1127 |
| train | CX2_33 | 10 | 759 |
| train | CX2_34 | 10 | 722 |
| train | CX2_35 | 10 | 878 |
| train | CX2_37 | 10 | 709 |
| validation | CX2_36 | 10 | 979 |
| test | CX2_38 | 10 | 769 |
| train | CX2_16 | 20 | 620 |
| train | CX2_33 | 20 | 352 |
| train | CX2_34 | 20 | 306 |
| train | CX2_35 | 20 | 421 |
| train | CX2_37 | 20 | 336 |
| validation | CX2_36 | 20 | 463 |
| test | CX2_38 | 20 | 350 |

All feature comparisons use the same complete-case F2 feature union; missing inputs are rejected, never imputed or substituted. Reported individually usable cycles and common-cohort counts are in cycle_counts.csv. Rejections are in cycle_audit.csv and window_audit.csv. Windows require consecutive original ordinals, the same single source sheet, and nonoverlapping experimental times with gaps <=3600 s. No window is made by compacting rejected cycles.

Longer windows change the eligible targets. sequence_common_target_metrics.csv and its figure repeat the validation comparison on the intersection of target-cycle IDs. No cross-length improvement should be attributed only to architecture without checking this matched cohort.

## Experimental comparisons

| architecture | feature_set | sequence_length | rmse_fraction_mean | rmse_fraction_std | mae_fraction_mean | mae_fraction_std | mape_percent_mean | r2_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B1 | F1 | 5 | 0.00764933 | 0.000269827 | 0.00596204 | 0.000256282 | 0.812783 | 0.996169 |
| B1 | F2 | 5 | 0.00613274 | 0.000791636 | 0.00457269 | 0.000644536 | 0.658456 | 0.997512 |
| B1 | F3 | 5 | 0.00546748 | 0.000459733 | 0.00405817 | 0.000372305 | 0.578857 | 0.998035 |
| B1 | F3 | 10 | 0.00640759 | 0.0014825 | 0.00477739 | 0.00102326 | 0.685126 | 0.997144 |
| B1 | F3 | 20 | 0.00730283 | 0.00273377 | 0.00553543 | 0.00189334 | 0.793889 | 0.99601 |
| B2 | F1 | 5 | 0.00900068 | 0.000975167 | 0.00696658 | 0.000742215 | 0.967157 | 0.994659 |
| B2 | F3 | 5 | 0.0085549 | 0.000259104 | 0.00631486 | 0.000172732 | 0.911214 | 0.995209 |
| B2 | F3 | 10 | 0.00808923 | 0.000615354 | 0.00596539 | 0.000520579 | 0.848422 | 0.995589 |
| B2 | F3 | 20 | 0.00890112 | 0.00135538 | 0.00668149 | 0.00102263 | 0.926615 | 0.994495 |
| B3 | F3 | 5 | 0.00689587 | 0.0007256 | 0.0051803 | 0.000581542 | 0.740612 | 0.996866 |
| B3 | F3 | 10 | 0.00787518 | 0.000837881 | 0.00576717 | 0.000500493 | 0.83191 | 0.995804 |
| B3 | F3 | 20 | 0.00808612 | 0.000472741 | 0.00627117 | 0.000321348 | 0.884927 | 0.995516 |
| B4 | F3 | 5 | 0.00561247 | 0.000519573 | 0.00414847 | 0.000540504 | 0.590321 | 0.997928 |
| B4 | F3 | 10 | 0.00841377 | 0.00267139 | 0.00595832 | 0.00168917 | 0.865702 | 0.994927 |
| B4 | F3 | 20 | 0.0125104 | 0.00400994 | 0.00934765 | 0.00251221 | 1.35746 | 0.988557 |

A: B1/length 5 compares F1/F2/F3. B: architectures are selected using the chosen feature set at length 5. C: length 5/10/20 is selected for that architecture. The full four-architecture by three-length grid is also available. D: every configuration uses three seeds. The F1/B2/length-5 predictor-settings comparison is pre-specified. All choices were frozen in final_selection.json before any test checkpoint evaluation.

## Held-out test results

| architecture | feature_set | sequence_length | mae_fraction_mean | mae_fraction_std | rmse_fraction_mean | rmse_fraction_std | mse_fraction_squared_mean | mse_fraction_squared_std | mape_percent_mean | mape_percent_std | r2_mean | r2_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B1 | F3 | 5 | 0.0141683 | 0.000496354 | 0.0147363 | 0.000482525 | 0.000217312 | 1.42238e-05 | 1.90773 | 0.0717454 | 0.986882 | 0.000858641 |
| B2 | F1 | 5 | 0.00513915 | 0.000607682 | 0.00661688 | 0.000694415 | 4.41046e-05 | 9.33849e-06 | 0.706732 | 0.0843565 | 0.997338 | 0.000563734 |

Each frozen checkpoint is evaluated once on CX2_38; resumed reporting reads saved predictions. Test results never select features, epochs, architectures, lengths or seeds. Training/validation/test predictions and attention weights are saved separately and aggregated. The seed-mean trajectory figure shows seed variability, not a calibrated confidence interval.

## Compute and reproducibility

Total measured full-training-loop time: 466.1 s (0.129 h), including epoch validation and best-state copies, excluding dataset preparation and report generation. Parameter counts, per-run inference wall times and peak allocated/reserved GPU memory are in validation_results.csv. Inference is one GPU-synchronized forward pass after synthetic warmup; transfer/metric time is excluded, and short single-pass timing is noisy.

Training uses AdamW (lr 0.0005, weight decay 0.001), MSE, batch 64, at most 200 epochs, patience 20, gradient clipping 1.0 and ReduceLROnPlateau (factor 0.5, patience 5, minimum lr 1e-6). A 1e-6 fractional RMSE improvement is required to reset early stopping. B1/B4 use one 64-unit LSTM layer; B2/B3 use two layers and 0.2 inter-layer dropout. B3 is bidirectional within the available window. B4 uses learned additive temporal attention. Targets and predictions are not clipped.

Random/Python/NumPy/CPU/CUDA seeds are set; deterministic PyTorch algorithms and cuDNN are requested, benchmarking/TF32 disabled, CUBLAS_WORKSPACE_CONFIG=:4096:8 set before CUDA initialization. Reproducibility across different hardware/driver/library versions is not guaranteed. experiment_config.json records versions, source/code hashes, split, feature lists, thresholds and hyperparameters. Completed runs are cached; incompatible scientific configurations require a separate directory.

## Interpretation and limitations

The benchmark measures unseen-battery generalization using charging observations through the current cycle, not next-cycle forecasting. Smoothing is contained within each charge; none of the three feature sets uses historical or future-cycle feature engineering. Current-cycle discharge measurements are used exclusively for target verification and protocol audit, never as input columns.

F3 was fixed by the user from the prior exploratory feature pipeline. That earlier analysis included the battery now used for validation (CX2_36), so validation independence from exploratory feature selection is imperfect. The compact set is not claimed optimal, and the untouched test battery is the independent evaluation. No feature reranking was performed for this benchmark.

Only one validation and one test battery are available. Window overlap induces serial dependence; seed SD measures optimization variability and does not establish population uncertainty or statistical superiority. Strict continuity/complete-case filtering changes lifetime coverage. Future cross-protocol evaluation on CX2_3/CX2_8 and additional held-out batteries remains necessary.

## Literature reference

[Wang et al., Batteries 2026, DOI 10.3390/batteries12040115](https://www.mdpi.com/2313-0105/12/4/115) was checked against the user-supplied 21-page PDF. Table 3 (p. 9) confirms the F1/B2/length-5 predictor dimensions: input 3, sequence 5, layers 2, hidden size 64, output 1. The partial-charge interval is also confirmed as 3.8-4.0 V at the 4.2 V cutoff (p. 5). This verifies predictor dimensions and feature types, not the full AW-PINN or every published pure-LSTM comparison. The paper uses Gaussian IC smoothing (sigma 1), [0,1] feature normalization (p. 6), a within-battery chronological 50/50 split (p. 12), and an aging domain ending at 80% SOH (p. 4). Our Savitzky-Golay extraction, training-only StandardScaler, independent battery split and retained test targets extending below 80% are different. Dropout and optimizer details are not reported in the searchable predictor/training settings and are not assumed. See [verified methodological comparison](LITERATURE_COMPARISON.md), literature_comparison.csv and literature_verification.json for page references, protocol differences and metric-reporting caveats. The verification changed no model or result, and no inference was repeated. No direct published-performance superiority claim is made.

[PyTorch reproducibility documentation](https://docs.pytorch.org/docs/stable/notes/randomness.html). [CUDA RNN determinism](https://docs.pytorch.org/docs/stable/cudnn_rnn_determinism.html).

## Lifetime coverage

| split | battery | sequence_length | evaluated_targets | validated_routine_cycles | target_coverage_percent | window_target_min_soh_percent | window_target_max_soh_percent | full_validated_min_soh_percent | full_validated_max_soh_percent |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | CX2_16 | 5 | 1507 | 1924 | 78.3264 | 59.324 | 94.0813 | 59.324 | 94.3429 |
| train | CX2_33 | 5 | 1116 | 1610 | 69.3168 | 50.0085 | 95.738 | 41.3862 | 95.9999 |
| train | CX2_34 | 5 | 1125 | 1636 | 68.7653 | 48.0762 | 99.274 | 41.4116 | 100.366 |
| train | CX2_35 | 5 | 1233 | 1726 | 71.4368 | 47.5217 | 100.36 | 41.1523 | 101.183 |
| train | CX2_37 | 5 | 967 | 1231 | 78.554 | 60.7048 | 99.955 | 60.7048 | 101.21 |
| validation | CX2_36 | 5 | 1345 | 1888 | 71.2394 | 48.9864 | 99.4918 | 40.1054 | 100.775 |
| test | CX2_38 | 5 | 1102 | 1867 | 59.0252 | 47.2186 | 99.7025 | 20.5267 | 100.945 |

The result applies only to retained windows. Low-SOH and discontinuous records excluded by feature availability or continuity are not evaluated, and no accuracy claim covers those lifetime regions. test_lifetime_coverage.png shows this restriction directly.

## Conclusions from this experiment

The frozen validation choice B1/F3/L5 has mean test RMSE 1.4736 +/- 0.0483 percentage points. The pre-specified F1/two-layer predictor has 0.6617 +/- 0.0694 percentage points on the same retained test targets.

Validation rankings did not transfer to the one held-out test battery: the pre-specified literature-settings comparator has the lower observed test error. This does not change the validation-selected configuration and does not authorize test-driven redesign. The comparison changes both feature set and architecture, so the gap cannot isolate the effect of stacking or feature selection.

At length 5, additional architectural complexity did not consistently lower mean validation error relative to Vanilla LSTM. Longer windows changed the sample cohort and generally increased seed variation; consult the identical-target validation analysis before attributing any change to history length. These are descriptive findings for this split, not statistical population-level superiority claims.