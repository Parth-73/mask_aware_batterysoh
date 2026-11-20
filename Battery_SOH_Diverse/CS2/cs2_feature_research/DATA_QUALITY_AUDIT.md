# Data quality audit

## Label validation reasons
```
reason
no_verified_full_charge_before_discharge                                                                                                                                                            13428
partial_discharge_or_missing_cutoff                                                                                                                                                                 12734
validated                                                                                                                                                                                            7793
partial_discharge_or_missing_cutoff;no_verified_full_charge_before_discharge                                                                                                                          715
insufficient_discharge_points                                                                                                                                                                         323
partial_discharge_or_missing_cutoff;no_verified_full_charge_before_discharge;recorded_integrated_capacity_disagreement                                                                                298
partial_discharge_or_missing_cutoff;recorded_integrated_capacity_disagreement                                                                                                                         166
insufficient_discharge_points;partial_discharge_or_missing_cutoff;no_verified_full_charge_before_discharge;recorded_integrated_capacity_disagreement                                                  163
insufficient_discharge_points;partial_discharge_or_missing_cutoff                                                                                                                                     138
insufficient_discharge_points;invalid_discharge_time;implausible_integrated_capacity                                                                                                                   54
insufficient_discharge_points;invalid_discharge_time;partial_discharge_or_missing_cutoff;no_verified_full_charge_before_discharge;implausible_integrated_capacity                                      14
partial_discharge_or_missing_cutoff;no_verified_full_charge_before_discharge;intervening_active_measurement                                                                                             6
partial_discharge_or_missing_cutoff;intervening_active_measurement                                                                                                                                      4
ambiguous_current_polarity                                                                                                                                                                              3
no_verified_full_charge_before_discharge;intervening_active_measurement                                                                                                                                 3
insufficient_discharge_points;invalid_discharge_time;partial_discharge_or_missing_cutoff;no_verified_full_charge_before_discharge;implausible_integrated_capacity;intervening_active_measurement        3
insufficient_discharge_points;invalid_discharge_time;partial_discharge_or_missing_cutoff;implausible_integrated_capacity                                                                                2
partial_discharge_or_missing_cutoff;no_verified_full_charge_before_discharge;unknown_charge_discharge_gap                                                                                               1
insufficient_discharge_points;partial_discharge_or_missing_cutoff;no_verified_full_charge_before_discharge                                                                                              1
no_verified_full_charge_before_discharge;unknown_charge_discharge_gap                                                                                                                                   1
insufficient_discharge_points;partial_discharge_or_missing_cutoff;no_verified_full_charge_before_discharge;unknown_charge_discharge_gap                                                                 1
```

## Source inventory
```
status
raw                          570
metadata                     549
statistics_excluded          127
raw_cadex_clock_verified      33
manual_review_cadex_clock     24
```

Discharge completeness requires preceding observed charge ending >=4.18 V and <=0.055 A, an observed discharge endpoint <=2.72 V, >=12 points, finite monotone discharge time with no >120 s interval, a plausible positive integrated capacity <=1.65 Ah and <=3% disagreement with the recorded counter increment where available. Voltage/current tolerances accommodate instrument resolution; the previous threshold was not relaxed to increase counts. Sparse charging intervals require matching operating-step clocks and increasing charge counters. Their throughput uses recorded counter increments; observed rests separate CC and CV and are excluded from active charging throughput. No discharge threshold is relaxed. The first unobserved fraction of each step is not integrated. Recorded counters are differenced per episode, never treated as per-cycle absolute capacity. Unusual SOH is retained when measurement checks pass.

Raw channel sheets are concatenated within each workbook, sorted by test time, and duplicate measurement tuples removed. Identical discharge timestamp/capacity signatures across files are excluded and exported. Other overlapping source intervals remain a manual-review concern. No cross-workbook fragment is joined. This loses some legitimate boundary cycles conservatively. Cycle ordinals measure observed episodes, not verified absolute cycle age. No missing interval is called rest; rest durations sum only near-zero-current blocks supported by observed samples and matching step clocks. CADEX clock units are accepted only when Duration versus Time fits a 60:1 slope within 0.1 and maximum residual below 5 seconds, with monotone Duration. Current/voltage use explicit mA/mV headers. Capacity is percentage-like and never treated as Ah. Absolute CADEX timestamps remain unknown; accepted capacity integrates measured current against the validated seconds clock.


## Final artifact validation and extreme-value investigation

{
  "validated_labels": 7793,
  "unique_batteries": 15,
  "selected_features": [
    "cc_throughput_ah",
    "throughput_bracket_3p9_4p1_ah",
    "cc_voltage_increase_v",
    "cc_voltage_slope_v_s",
    "charge_voltage_mean_v"
  ],
  "all_label_endpoints_pass": true,
  "all_available_counter_crosschecks_pass": true,
  "disjoint_splits": true,
  "forecast_horizon_indices_pass": true,
  "retained_extreme_soh_labels": 569,
  "unresolved_cadex_files": 24,
  "read_error_files": 0
}

Voltage-window throughput features use measured intervals bracketing the requested voltage endpoints, hence their explicit `throughput_bracket_` names. They include finite sample-edge contributions outside the exact thresholds. Crossing-time features interpolate the threshold crossing times. This distinction matters for acquisition robustness.

Very low measured SOH values are retained, not clipped. Their source records reach the required voltage cutoff after observed full-charge termination and pass available counter/integration checks. These checks establish delivered capacity under the recorded operating rate, not rate-independent intrinsic health. Low-capacity cycles therefore require physical review before publication; see cs2_extreme_capacity_review.csv. The pipeline.log contains earlier development-run failures as well as later runs; the final schema inventory and artifact_validation.json report the completed run.
