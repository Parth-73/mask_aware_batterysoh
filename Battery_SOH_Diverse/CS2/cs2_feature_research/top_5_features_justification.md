# Five provisional features: measured evidence and physical interpretation

The selected subset did not beat the simple charging or persistence baseline on the untouched test cells. Selection is provisional; cross-protocol generalization is not established.

## 1. `cc_throughput_ah`

Charge supplied in the identified constant-current region.

Training current-cycle Pearson 0.8735, Spearman 0.9026, mutual information 2.1767. Selected in 8/9 outer folds. Inner battery-macro RMSE for this prefix: 0.08507. Added using inner held-out-cell predictive error, with redundancy screening; this is an empirical selection, not proof of a unique physical mechanism.

## 2. `throughput_bracket_3p9_4p1_ah`

Measured charge over sample intervals bracketing 3.9–4.1 V. This is a sample-edge approximation, not exactly interpolated threshold-to-threshold charge.

Training current-cycle Pearson 0.8377, Spearman 0.8127, mutual information 1.3814. Selected in 7/9 outer folds. Inner battery-macro RMSE for this prefix: 0.08448. Added using inner held-out-cell predictive error, with redundancy screening; this is an empirical selection, not proof of a unique physical mechanism.

## 3. `cc_voltage_increase_v`

Observed CC voltage rise.

Training current-cycle Pearson 0.6934, Spearman 0.7873, mutual information 1.5969. Selected in 6/9 outer folds. Inner battery-macro RMSE for this prefix: 0.08611. Added using inner held-out-cell predictive error, with redundancy screening; this is an empirical selection, not proof of a unique physical mechanism.

## 4. `cc_voltage_slope_v_s`

Linear slope of voltage during CC charging.

Training current-cycle Pearson -0.4513, Spearman -0.8208, mutual information 1.5926. Selected in 8/9 outer folds. Inner battery-macro RMSE for this prefix: 0.09078. Added using inner held-out-cell predictive error, with redundancy screening; this is an empirical selection, not proof of a unique physical mechanism.

## 5. `charge_voltage_mean_v`

Mean observed charging voltage; depends on sampling and operating protocol.

Training current-cycle Pearson -0.8332, Spearman -0.8444, mutual information 1.6585. Selected in 7/9 outer folds. Inner battery-macro RMSE for this prefix: 0.10596. Added using inner held-out-cell predictive error, with redundancy screening; this is an empirical selection, not proof of a unique physical mechanism.

Highest absolute Pearson five: cc_throughput_ah, cc_duration_s, charge_throughput_ah, charge_energy_wh, cv_duration_ratio.

Highest absolute Spearman five: charge_throughput_ah, charge_energy_wh, cc_throughput_ah, cc_duration_s, throughput_bracket_3p8_4p0_ah.

Final features optimize one_cycle_forecasting; pooled correlation alone does not measure cross-cell error or incremental value. The complete subset search log, outer selection frequencies and selected-feature redundancy matrix provide the numerical basis.
