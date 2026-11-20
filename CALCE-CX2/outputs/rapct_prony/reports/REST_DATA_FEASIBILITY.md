# Rest data feasibility

```json
{
  "verified_events": 36005,
  "uncertain_gaps": 12450,
  "per_battery": {
    "CX2_16": 6012,
    "CX2_33": 5050,
    "CX2_34": 5190,
    "CX2_35": 5462,
    "CX2_36": 5899,
    "CX2_37": 3814,
    "CX2_38": 4578
  },
  "duration_quantiles_s": {
    "0.0": 1.0,
    "0.25": 5.0,
    "0.5": 90.0,
    "0.75": 125.0,
    "0.95": 125.0,
    "1.0": 235.0
  },
  "validated_pairs": 23107,
  "consistent_conditions": 22824,
  "usable_pair_events": 22824,
  "unique_target_cycles": 11333,
  "usable_batteries": [
    "CX2_16",
    "CX2_33",
    "CX2_34",
    "CX2_35",
    "CX2_36",
    "CX2_37",
    "CX2_38"
  ],
  "long_verified_events_gt_1h": 0,
  "recovery_training_status": "requires_duration_diversity_and_state_history_review; ordinary_short_rest_is_not_irregular_long_rest_validation"
}
```

|I| <=0.005 A at clock-consistent consecutive observations <=180 s apart defines sampled measured rest. No cross-source gap is open-circuit evidence. Overlapping acquisition sources downgrade verification conservatively. Events are joined to the preceding discharge endpoint and following charge cutoff, never from SOH changes; pair eligibility requires consecutive validated cycles and discharge C-rate agreement within 0.02C. Counts are events, with unique target cycles reported separately. Label-derived discharge conditions are retrospective evaluation metadata only. A cutoff-valid event is not independent evidence of capacity recovery. Long-rest data support and unknown-history state initialization must be reviewed before any recovery training.

## Persistent-state feasibility gate

```csv
battery,samples,with_verified_rest,unknown_gaps,usable_state_transitions
CX2_16,1899,1899,1899,0
CX2_33,1488,1488,1488,0
CX2_34,1554,1554,1554,0
CX2_35,1630,1630,1630,0
CX2_36,1743,1743,1743,0
CX2_37,1219,1219,1219,0
CX2_38,1477,1477,1477,0

```

Every matched sample contains at least one uncertain acquisition interval since the preceding retained cutoff (or since its charge start at a discontinuity). Verified short rests exist, but zero uninterrupted state transitions survive the conservative policy. Experimental recovery training is unavailable in this execution. Sparse CV event logging is accepted for capacity integration only with counter checks; it does not independently document memory dynamics throughout the unobserved interval. A better documented gap-transition model could change this conclusion. No claim is made that every gap was physical rest, or that the dataset lacks short rests.
