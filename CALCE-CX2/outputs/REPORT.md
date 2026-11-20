# Feature pipeline execution report

| battery | total | validated | conventional |
| --- | --- | --- | --- |
| CX2_16 | 2016 | 1939 | 1924 |
| CX2_3 | 11069 | 1856 | 1854 |
| CX2_32 | 367 | 0 | 0 |
| CX2_33 | 1700 | 1618 | 1610 |
| CX2_34 | 1736 | 1657 | 1636 |
| CX2_35 | 1814 | 1735 | 1726 |
| CX2_36 | 1971 | 1893 | 1888 |
| CX2_37 | 1281 | 1236 | 1231 |
| CX2_38 | 1960 | 1877 | 1867 |
| CX2_8 | 2071 | 2011 | 0 |

Rejected/incomplete candidates: 10163
Provisional compact features: cc_duration_s, cv_duration_s, dva_peak, ic_peak_voltage, ic_peak_width

F4 uses training statistics only. Validation-performance confirmation is deferred; no predictive model has been fitted.
Same-cycle measured discharge C-rate is audit metadata, not a charging-time predictor.
Text sessions without absolute timestamps/verified capacity units are inventoried but not reconstructed.
Lifetime cycle numbers are observed reconstruction ordinals, not a claim of continuous lifetime coverage.

## Inventory

| status | sheets_or_files |
| --- | --- |
| ready | 457 |
| relative_time_only_counter_units_unverified | 218 |
| temperature_only_no_synchronized_current_session | 103 |
| temperature_schema_error | 5 |

## Training-only associations

| feature | n | pearson | spearman | mutual_information | battery_mean_spearman | sign_consistency |
| --- | --- | --- | --- | --- | --- | --- |
| cc_duration_s | 8783 | 0.97533 | 0.98542 | 2.6756 | 0.98463 | 1 |
| partial_charge_time_s | 8315 | 0.98794 | 0.98855 | 2.3715 | 0.99494 | 1 |
| partial_charge_ah | 8315 | 0.98794 | 0.98857 | 2.3742 | 0.99493 | 1 |
| partial_dv_dq | 8315 | -0.91179 | -0.98857 | 2.371 | -0.99493 | 1 |
| dva_mean | 8315 | -0.91078 | -0.98823 | 2.357 | -0.9948 | 1 |
| historical_partial_change_ah | 8311 | 0.97765 | 0.97534 | 2.1514 | 0.99494 | 1 |
| ic_peak_area | 8315 | 0.995 | 0.99592 | 2.8172 | 0.99594 | 1 |
| ic_peak_height | 8315 | 0.97532 | 0.9858 | 2.0529 | 0.99154 | 1 |
| cv_duration_s | 8782 | -0.78606 | -0.87964 | 1.1564 | -0.93893 | 1 |
| cv_decay_rate_s_inv | 8780 | 0.8043 | 0.85488 | 0.9121 | 0.91797 | 1 |
| dva_peak | 8315 | -0.79357 | -0.87593 | 1.2673 | -0.9221 | 1 |
| ic_peak_voltage | 8315 | -0.82393 | -0.84357 | 0.85852 | -0.897 | 1 |
| historical_ic_peak_shift_v | 8311 | -0.8558 | -0.90457 | 1.0232 | -0.89734 | 1 |
| cv_capacity_ah | 8782 | -0.73109 | -0.78061 | 0.7893 | -0.82235 | 1 |
| ic_peak_width | 8315 | -0.73028 | -0.79703 | 0.80238 | -0.77419 | 1 |
| charge_crate | 8784 | 0.046563 | 0.081647 | 0.67512 | 0.56925 | 1 |
| charge_protocol_cv | 8784 | 0.0008348 | 0.001958 | 4.5524e-05 | 0.0031387 | 0.5 |
| historical_resistance_change_ohm | 0 | NA | NA | NA | NA | NA |
| resistance_ohm | 0 | NA | NA | NA | NA | NA |
| temperature_c | 0 | NA | NA | NA | NA | NA |
| relaxation_60s_v | 0 | NA | NA | NA | NA | NA |

## Feature missingness among validated complete cycles

| feature | missing_percent |
| --- | --- |
| ic_peak_height | 8.6778 |
| partial_charge_ah | 8.6778 |
| partial_dv_dq | 8.6778 |
| ic_peak_voltage | 8.6778 |
| ic_peak_area | 8.6778 |
| ic_peak_width | 8.6778 |
| dva_mean | 8.6778 |
| dva_peak | 8.6778 |
| partial_charge_time_s | 8.6778 |
| cc_duration_s | 0.0063203 |
| cv_duration_s | 0.012641 |
| cv_decay_rate_s_inv | 0.025281 |
| cv_capacity_ah | 0.012641 |
| historical_ic_peak_shift_v | 8.722 |
| historical_partial_change_ah | 8.722 |
| resistance_ohm | 100 |
| historical_resistance_change_ohm | 100 |
| charge_crate | 0 |
| charge_protocol_cv | 0 |
| temperature_c | 100 |
| relaxation_60s_v | 87.29 |

Unresolved current-text sessions are not counted as zero-cycle batteries. Separate temperature logs cannot yet be synchronized to these current exports.

Unit/regression tests: run `python -B -m unittest discover -s tests -v`. See `tests/test_pipeline.py` for numerical, reconstruction, causal-history, protocol-family, and held-out-selection regression checks.