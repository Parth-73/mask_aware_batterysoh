# Preprocessed, ready-to-use CALCE data: compact five-feature set

This folder is a separate export of the original validated `outputs/cx2_features_all.csv`. It does not replace raw measurements, earlier preprocessing or research outputs. It contains **11,010 real validated conventional-protocol cycles** with all five charging features present. No missing predictors were imputed and no target was clipped, smoothed or converted to percent.

## The five features and why they were selected

| Predictor | Original unit | Reason for including it |
|---|---|---|
| `cc_duration_s` | seconds | Describes observed active constant-current charging time; supplies a time-domain charging descriptor. |
| `cv_duration_s` | seconds | Describes verified constant-voltage taper duration; complements CC timing with late-charge behavior. |
| `dva_peak` | V/Ah | Maximum smoothed dV/dQ over the established 3.8-4.0 V partial-capacity interval; supplies a charging-curve slope descriptor. |
| `ic_peak_voltage` | volts | Position of the largest reliable interior incremental-capacity peak; describes where the curve peak occurs. |
| `ic_peak_width` | volts | Width of that peak at half prominence; adds peak breadth beyond its position. |

These are the **historical compact F3 feature set**, selected as a group in the saved LSTM benchmark. They are not a newly calculated ranking of the five individually most important features, and their listed order is not an importance ranking. Time descriptors and curve-shape descriptors provide complementary representations; this does not prove each adds independent predictive information.

The stored selection policy was mean validation RMSE across seeds 42, 123 and 2026, Vanilla LSTM, five-cycle windows, on a matched cohort:

| Feature group | Validation RMSE, fractional SOH | SOH percentage points |
|---|---:|---:|
| F3: these five features | 0.00546748 | 0.546748 |
| F2: nine features | 0.00613274 | 0.613274 |
| F1: three features | 0.00764933 | 0.764933 |

This supports the compact **group** under that historical protocol. It is not a universal optimality claim, an individual-feature ablation, or a performance guarantee for a new model trained on this export. The historical scores use temporal windows; these exported rows are individual cycles.

`feature_selection_reasons.csv` gives per-feature reasons and limitations. `training_feature_statistics.csv` provides training-only distributions and pooled/within-battery Spearman associations. Those associations are descriptive, can be confounded by protocol and aging, and were not used to reselect features on validation/test data. `feature_definitions.csv` preserves the existing mathematical definitions and reliability rules.

## Files to use

- `all_data.csv`: all retained cycles, original-unit predictors, fractional SOH, split and provenance.
- `train.csv`, `validation.csv`, `test.csv`: the same columns, already split by battery.
- `model_inputs/X_train.csv`, `X_validation.csv`, `X_test.csv`: exactly five predictors in the fixed feature order, in original units.
- `model_inputs/y_train.csv`, `y_validation.csv`, `y_test.csv`: aligned `soh` targets.
- `standardized/X_train.csv`, `X_validation.csv`, `X_test.csv`: ready-to-use dimensionless standardized predictors, fitted on training batteries only. The target stays in `model_inputs/y_*.csv` unchanged.
- `standardized/scaler.json`: training means/scales, fit batteries and feature order.
- `row_audit.csv`: every original candidate's inclusion/exclusion decision and original quality/feature rejection flags.
- `sample_counts.csv`, `split_definition.json`, `feature_order.json`, `manifest.json`, `verification.json`: counts, explicit contract, hashes and verification results.
- `source_evidence/`: copies of the original feature definitions, historical feature-selection evidence and historical LSTM configuration.
- `load_data.py`: a loader that returns only the five predictors, targets and separate metadata.
- `rebuild_dataset.py`: reproducible export script; it refuses to overwrite an existing `all_data.csv`.

## Frozen battery split

| Split | Batteries | Rows |
|---|---|---:|
| Training | CX2_16, CX2_33, CX2_34, CX2_35, CX2_37 | 7,790 |
| Validation | CX2_36 | 1,743 |
| Test | CX2_38 | 1,477 |

This split comes from the historical LSTM configuration, consistent with the feature-selection evidence. It is explicitly saved in this folder and does not read the currently edited original `outputs/battery_splits.json`. These previously examined cells are a benchmark, not untouched external validation.

The validation/test rows were never used to fit the exported scaler. For a different battery split or nested cross-validation fold, fit a **new scaler** on that fold's training cells from the original-unit files; do not reuse the provided standardized inputs.

## Target and information availability

`soh = validated_discharge_capacity_ah / 1.35 Ah`, expressed as a fraction: `0.8` means 80% SOH. The corresponding prediction cutoff is the end of the current available charging observations. The five predictors are charging-derived; subsequent discharge capacity is used only to establish the offline target and quality acceptance.

Battery ID, row ID, cycle ordinal, source identifiers, split and timestamps are provenance, **not predictors**. The exported cycle-start timestamp is not an end-of-charge cutoff timestamp. Same-cycle discharge measurements, target values, future cycles, post-cutoff rest, and unverified temperature/resistance must not become model inputs. Always use the explicit feature allowlist, rather than selecting every numeric CSV column.

Only conventional cycles with a validated target and five finite features are retained. Incomplete cycles, unusual protocols and missing predictors are excluded with reasons retained. Complete-case filtering can bias which aging regions remain. No additional CNN raw-curve-coverage requirement is imposed by this export.

These are cycle-level rows, not prebuilt temporal windows. Original cycle ordinals and source identifiers are preserved. For LSTM/Transformer windows, require chronological consecutive original ordinals within one battery and a trustworthy acquisition segment; never silently stitch over rejected cycles, source boundaries, overlaps or missing history. Never randomly divide windows from one battery across training and evaluation.

## Quick use

From this folder:

```python
from load_data import load_split

X_train, y_train, train_metadata = load_split('train', standardized=True)
X_val, y_val, val_metadata = load_split('validation', standardized=True)
X_test, y_test, test_metadata = load_split('test', standardized=True)

# X has exactly five columns. y remains fractional SOH.
# Select model settings using validation only, then evaluate on test.
```

Alternatively, load `model_inputs/X_*.csv` and fit your own train-only preprocessing. Do not standardize the provided standardized data a second time. The source hashes, exact raw-value CSV roundtrip, finite values, target definition, disjoint battery sets and training-only normalization are checked during export. The original source files are verified unchanged.
