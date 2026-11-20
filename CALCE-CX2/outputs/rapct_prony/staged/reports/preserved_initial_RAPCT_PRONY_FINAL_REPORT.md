# RAPCT-Prony execution report

This is an artifact-derived interim research report. It does not establish the central recovery hypothesis or publishable novelty.

## Environment and provenance

```json
{
  "python": "3.14.6",
  "packages": {
    "torch": "2.11.0+cu128",
    "numpy": "2.4.6",
    "pandas": "3.0.3",
    "scipy": "1.18.0",
    "scikit-learn": "1.9.0",
    "optuna": "5.0.0",
    "matplotlib": "3.11.0",
    "seaborn": "0.13.2",
    "tqdm": "4.68.2",
    "openpyxl": "3.1.5",
    "pytest": "9.0.3",
    "joblib": "1.5.3",
    "python-calamine": "0.9.0",
    "sqlalchemy": "2.1.6",
    "alembic": "1.20.1"
  },
  "cuda_available": true,
  "torch_cuda": "12.8",
  "gpu": "NVIDIA GeForce RTX 4050 Laptop GPU",
  "vram_bytes": 6438780928,
  "capability": [
    8,
    9
  ],
  "bf16": true
}
```

Raw hashes, historical source/config hashes, source inspection and rejection inventories are in data_audit. See experiments/rapct_prony/research/RELATED_WORK.md for the primary-source comparison and access limitations. Historical baseline scores are separate because the new full-coverage curve cohort differs.

## Prediction and preprocessing

Prediction occurs at end of current available charge, before the validated subsequent discharge. Target is discharge integral / 1.35 Ah, without clipping or smoothing. Five charging indicators use historical extraction definitions. Curves contain integrated Q, SG-derived dQ/dV and I on 176 voltage positions over 3.80–4.15 V. No extrapolation or per-curve capacity normalization. Only matched complete cases enter the three Stage-1 pathways. Scalers fit training batteries only and are stored with every run.

Retained samples: 11010. Batteries: CX2_16, CX2_33, CX2_34, CX2_35, CX2_36, CX2_37, CX2_38.

## Actual run metrics

Values below are read from completed run artifacts. RMSE/MAE pp are percentage points; fractional columns remain in metrics/all_runs.csv.

```csv
No completed training runs.
```

Pilot configurations use one inner validation battery and one outer battery, with explicit run seeds and budgets. They are cross-battery benchmarks on previously examined cells, not untouched external validation and not completed nested multi-seed evidence. No hypothesis selection should use the displayed outer scores.

## Rest feasibility

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

Sampling-confirmed zero-current intervals and uncertain acquisition gaps are separate. A verified interval alone does not establish a causal capacity-recovery event or identified transient state. No measured increase is used to trigger a branch.

## Stage gates and limitations

Stage 0: raw/source/config audit, causal contract and measured-rest inventory executed. Literature review is partial where full text is inaccessible.
Stage 1: implementations and automated tests exist; completed experiments are exactly those in the table.
Later stages require their own execution artifacts and will be recorded separately as work proceeds. Unavailable recovery performance, learned electrochemical interpretations and final nested evaluations must not be inferred from Stage-1 scores.

## Scientific conclusion

The central question remains unresolved until matched temporal/physics ablations, sufficient independent verified rest events and nested multi-seed evaluation are completed. Ordinary charging/rest data do not independently identify irreversible loss versus transient penalty. A full hybrid must outperform simpler baselines on post-rest targets without degrading ordinary-cycle generalization before any advantage can be claimed.
