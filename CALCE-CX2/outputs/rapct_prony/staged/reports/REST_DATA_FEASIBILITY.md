# Raw-observation rest feasibility

```json
{
  "raw_verified_fragments_before_reaudit": 36005,
  "verified_observed_intervals": 8790,
  "uncertain_gaps": 39572,
  "events_per_battery": {
    "CX2_16": 1257,
    "CX2_33": 926,
    "CX2_34": 932,
    "CX2_35": 1019,
    "CX2_36": 1077,
    "CX2_37": 664,
    "CX2_38": 2915
  },
  "duration_quantiles_s": {
    "0.0": 1.0,
    "0.25": 90.0,
    "0.5": 90.0,
    "0.75": 125.0,
    "0.95": 125.0,
    "1.0": 286.0
  },
  "validated_pre_post_pairs": 6426,
  "consistent_condition_pairs": 6359,
  "usable_pre_cutoff_event_pairs": 5697,
  "unique_eligible_targets": 5014,
  "eligible_batteries": [
    "CX2_16",
    "CX2_33",
    "CX2_34",
    "CX2_35",
    "CX2_36",
    "CX2_37",
    "CX2_38"
  ],
  "verified_rests_over_one_hour": 0,
  "independent_long_rest_recovery_evaluation": false,
  "recovery_gate": "synthetic_memory_only; ordinary short protocol waits lack irregular long-rest excitation and independent recovery-state identification",
  "interpretation": "No claim that short rests have zero capacity effect. Labels never trigger events. Discharge-condition comparability is retrospective audit metadata only."
}
```

Verified means finite measured |I|<=0.005 A, same source without recorder resets, adjacent observed/test clock intervals <=180 seconds and agreement within two seconds. Zero-duration timestamp ties may connect observed intervals; only exact shared endpoints with the same mapped cycles are merged. Gaps, active boundary observations and overlapping/uncertain sources are never interpreted as rest. Only events ending by the target charge cutoff are available as inputs. Event count is not a count of independent recovery interventions. No hour-scale verified rests are assumed from filenames or sessions. For the irregular long-rest recovery question, experimental A6-A8 remain unavailable; synthetic memory results do not validate capacity recovery.
