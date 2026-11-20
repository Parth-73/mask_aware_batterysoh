# CALCE CX2 charging feature pipeline

This project reconstructs measured charge/discharge sequences, validates candidate SOH labels, and extracts reusable charging features. It does not implement or train prediction models. Raw files and the pre-existing `cx2_cycles.csv` are unchanged. The old CSV is not used.

## Setup and execution

Run in PowerShell from `D:\battery_soh\CALCE-CX2`:

```powershell
& D:\battery_soh\.venv\Scripts\python.exe -m pip install -r requirements.txt
& D:\battery_soh\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
& D:\battery_soh\.venv\Scripts\python.exe -m src.inspect_dataset --batteries CX2_16 --output outputs/pilot
& D:\battery_soh\.venv\Scripts\python.exe -u -m src.build_dataset --batteries CX2_16 --output outputs/pilot
& D:\battery_soh\.venv\Scripts\python.exe -u -m src.build_dataset
```

The tested direct dependency versions are pinned in `requirements-lock.txt`; use `-r requirements-lock.txt` when recreating this environment. Optional Jupyter/IPython is needed only to view the notebook.

Use `python -m src.build_dataset --reports-only` to resume downstream quality/statistics/plot generation from a complete feature checkpoint after an analysis failure. This mode does not apply changed reconstruction/extraction rules; rerun the full pipeline after such changes.

The full command inventories every supported raw file, processes conventional and unusual timestamped batteries separately by protocol, and creates all analysis outputs. Use `--raw PATH`, `--output PATH`, or `--batteries CX2_16 CX2_33 ...` to restrict a run. Always use a distinct output directory for a restricted run, to avoid overwriting full reports. The default split is frozen by battery identifier before analysis. If a selected run contains fewer than two usable training batteries, F4 remains empty.

## Inspection and reconstruction

`src/inspect_dataset.py` reads every Excel Channel sheet (and schema-compatible tables exported under generic sheet names) and records content SHA256, actual columns, experimental timestamp ranges, source clock reversals, local counter resets, schema failures, and unsupported text sessions. File modification times and filename dates never establish chronology. There are Excel, relative-time current text exports, and separate temperature logger CSVs; the inspected text schema has relative `Time`, `mV`, `mA`, `Temperature` and undocumented capacity fields, without absolute experimental timestamps. The current text exports are explicitly marked unresolved, not assigned invented dates or units. Temperature logger CSVs do contain absolute acquisition/scan timestamps and a Type T channel with explicit degrees-C units; they are summarized in `temperature_session_inventory.csv`. They cannot be synchronized with the relative-time current files without an independent experimental time anchor. No filename-based join is made. This affects text-only CX2_31/CX2_4 and additional sessions of other batteries.

Normalized Channel tables are cached as trusted local pickle files. Do not load caches from untrusted sources. Cache identity includes raw content, path, and schema version. The extracted battery cache additionally includes all source code and physical configuration. Changing extraction code or thresholds invalidates extraction caches. Failed reads remain in the inventory; a subsequent run retries them. There is a per-battery feature checkpoint. RAM use is bounded by one battery plus output tables; Excel is read once per cache version.

A measured cycle starts with one or more charge phases, may include intervening rest, and includes all discharge steps up to the next charge. Orphan discharge and terminal charge candidates are retained. Active current has magnitude >0.02 A. Local Cycle_Index is provenance rather than a grouping key. Exact duplicate exported rows are removed and counted in session events. Ambiguous overlapping sources are flagged. A clock reversal creates a separate recorder epoch. Continuous file boundaries use predecessor counters and elapsed test time only when the physical step and clock are compatible; otherwise completeness is rejected.

Long observed rests and inter-session gaps >3600 s split candidates. Recording gaps within active CC/discharge steps >180 s are excluded from integration and reject completeness. The pilot showed event-based CV logging with much longer sample intervals. A constant-voltage step at >=4.18 V, voltage spread <=0.035 V, and decreasing current may use intervals up to 3600 s. This exception is explicitly flagged and still requires counter agreement. It does not extrapolate across missing segments. Both limits are conservative operational thresholds, not claims about instrument settings.

Current capacity uses within-step trapezoids, `sum((I[k]+I[k-1])/2 * dt)/3600`. The first measured step interval uses the first current as a rectangular approximation, with the approximated duration exported. Counter capacity uses step endpoint differences against the preceding raw observation. It never uses the absolute lifetime capacity counter as cycle capacity. Resets within a step or unknown initial baselines invalidate counter verification. A reset to zero at a verified new step can be handled explicitly. Counter and current capacity must agree within `max(0.01 Ah, 5% of capacity)` for both charge and discharge.

Complete charge requires at least five measured CC samples below 4.18 V, final voltage >=4.18 V and final current <=0.06 A (tolerance around the supplied 0.05 A termination). Discharge must end in [2.6,2.75] V around the 2.7 V cutoff. Discharge capacity must be in [0.02,1.8] Ah. These are audit rules in `src/common.py`; altering them changes the dataset version through the code/config cache fingerprint. Every failed condition is listed in `rejection_reasons`. Quality acceptance establishes consistency with recorded measurements, not calibration accuracy or external ground truth.

`soh_candidate = integrated discharge Ah / 1.35`. `soh` is present only for a validated complete candidate, with `validated_discharge_capacity_ah` its numerator. SOH is a fraction, so 0.8 is 80%. No clipping to [0,1] occurs. Degraded capacity is not automatically an anomaly. Current-rate-dependent capacity from unusual protocols remains explicitly labeled; it is not pooled with conventional cycles in feature selection.

The routine/conventional protocol family uses one continuous discharge segment, median 0.30-1.10 C, peak <=1.20 C, and peak-minus-median <0.1 C. This covers the 0.4C/0.5C/1C protocols observed in the pilot/training batteries. Multisegment and 3C/high-rate protocols remain separately audited; this family is not a claim that different C-rates produce interchangeable capacity labels. Other protocols are retained as `unusual_discharge`. Observed cycle ordinals include rejected and partial candidates; they are not asserted to be the absolute laboratory lifetime cycle count. Missing sessions and unknown prehistory prevent that claim. Tests spanning weeks or self-discharge studies may become multiple partial candidates rather than invented complete cycles.

## Feature definitions and temporal availability

`outputs/feature_definitions.csv` gives each mathematical definition, units, raw measurements, phase, and missingness rule. Feature extraction uses charging measurements only, except for target/audit fields. Main definitions:

| Family | Definition |
| --- | --- |
| Partial capacity | Q(4.0 V)-Q(3.8 V), interpolated on measured constant-current charge |
| Partial voltage slope | 0.2 V / partial capacity |
| IC | Resample integrated Q on a 2 mV voltage grid over [3.80,4.15] V; Savitzky-Golay cubic smoothing, 11 points; differentiate numerically |
| IC height/area | Maximum dQ/dV and integral over that fixed voltage interval |
| IC position/width | Largest reliable interior peak, width at half prominence; missing if the global maximum is at an interval boundary or flanks are unresolved |
| Smoothing check | Peak-height relative spread across 7,11,15 point windows; >20% spread or significant negative IC invalidates IC values; variant heights are retained for audit |
| DVA | Smooth V(Q) on 151 equally spaced capacity points between the 3.8/4.0 V crossings; mean and maximum dV/dQ |
| CC/CV timing | Active charge durations below/above 4.18 V; rests are excluded; CV additionally requires constant voltage and measured current decay |
| CV decay | Negative least-squares slope of log I against active CV time; requires at least 8 samples, >=20% decay, and R2 >=0.8 |
| CV capacity | Integral I dt over verified CV samples |
| Causal history | Current feature minus first earlier finite charging reference; first observed reference is missing; never uses future cycles or SOH |
| Relaxation | V at 60 s minus initial voltage during the first measured rest after the final charge; requires 60 s coverage and |I|<0.005 A |

IC here is a reproducible numerical signature, not an assigned chemical reaction or phase transition. Sustained/large voltage reversals and variable-current CC invalidate curve features. Tiny ADC reversals are merged on a cumulative maximum voltage envelope, preserving the last Q. This avoids division by tiny measured voltage increments. Interpolation requires measured coverage; there is no extrapolation to unavailable voltage ranges. Strict interval coverage can cause substantial missingness.

The F1 columns reproduce the requested three feature types. The subsequently supplied Wang et al. PDF confirms the 3.8-4.0 V partial-charge interval (p. 5) and predictor dimensions (Table 3, p. 9). Its Gaussian IC smoothing with sigma 1 differs from the operational Savitzky-Golay definition above, and the exact IC peak-search interval is not specified. This pipeline does not claim exact numerical replication of that paper. The original feature_sets.json retains its as-run literature status; current verification is documented in [the LSTM literature comparison](outputs/lstm_baselines/LITERATURE_COMPARISON.md). F2 extends F1 with IC/DVA/charge behavior. F3 adds causal history and observed charging/rest protocol information. Unverified resistance and temperature remain NaN; recorded positive resistance is retained separately as descriptive audit evidence. No temperature is inferred from filenames or ambient conditions. Instrument metadata/calibration and a verified sensor channel are needed before activating these physical inputs.

Same-cycle measured discharge C-rate is exported as audit metadata and is deliberately excluded from all feature sets: it is unavailable at the end of charge. A truly pre-recorded scheduled protocol could be introduced as a separate future feature, with independent provenance. Actual same-cycle discharge capacity, target, battery ID, cycle ordinal and split are also excluded. Use the explicit feature-set allowlist; never train on every numeric output column.

## Splits and feature analysis

Frozen split: training CX2_16/33/34/35/36, validation CX2_37, test CX2_38, unusual protocol holdout CX2_3/8/32, unresolved text sessions CX2_4/31. Only validated conventional cycles from training batteries contribute correlations, mutual information, feature redundancy, VIF, gates, or F4. Test/validation plots are descriptive and do not influence selection. No imputer, scaler, or predictive estimator is fitted. Random seed is 42.

Pearson and Spearman are pairwise complete; mutual information is univariate, seeded and complete-case. Correlation coefficients describe associations in serially dependent data; no IID significance claims are made. Per-battery correlations and sign consistency expose pooled lifetime/protocol confounding. VIF uses complete training rows and reports rank deficiency/insufficient coverage; strong algebraic dependence between partial capacity, inverse slope, and time is expected.

Provisional F4 ranks within-battery Spearman association, direction consistency, availability and between-battery variability, then requires >=70% availability, >=2 training batteries, sign consistency >=75%, absolute mean Spearman >=0.3, and no pairwise absolute Spearman >0.9 against already selected features. It is limited to six features. Charging C-rate additionally needs a training 95th-minus-5th percentile spread of at least 0.05C; tiny instrument-scale deviations around a common 0.5C protocol are not selected as a physical operating-condition effect. Gates are fixed in code, not tuned against held-out data. Physical interpretation must be reviewed alongside the numeric ranking. **Validation-performance selection is intentionally deferred until predictive modeling is authorized. F4 is a provisional candidate, not a validated final subset.**

The same allowlist can be passed to any later FNN/CNN/LSTM/GRU/Transformer/PINN implementation. Preserve battery splits, chronological windows, and information availability. Tensor reshaping must not change columns or reveal future cycles. Later imputation/scaling must fit training data only and carry missingness masks where appropriate; do not fill absent physical measurements with invented values.

## Outputs

Required CSV/JSON files are generated under `outputs`, alongside inspection/session audit tables, per-battery summaries, selection decisions, VIF, smoothing variants and `run_metadata.json` (package versions, thresholds, raw inventory counts and code fingerprint). `REPORT.md` records measured counts and provisional F4. `figures` includes voltage-capacity, IC, DVA, IC position evolution, CC/CV duration evolution, validated SOH, completeness, missingness, capacity agreement, distributions and a training-only correlation heatmap. Representative curves are sampled across each battery's observed lifetime independently of labels. `notebooks/feature_analysis.ipynb` reads saved outputs and does not rerun selection on test data.

## Limitations

The inspected Excel channels cannot supply verified temperature or resistance trends. Separate CX2_4 temperature logs are measured and timestamped, but cannot yet be assigned to charging rows of the relative-time current exports. Text-only sessions remain unresolved without absolute experimental dates and counter-unit documentation. Cross-file continuity is conservative and unusual pulse/discharge sequences may be retained as partial candidates. CV boundaries are voltage/current rules rather than assumed Step_Index labels; small transition-time errors remain at sampling resolution. Initial-step integration and sparse CV trapezoids are approximations assessed against counters, not proof of exact coulomb counting. Additional protocol-specific reconstruction may recover rejected records after documented review. Feature missingness and quality flags must be retained in research reporting. No model training, GPU use, or predictive performance claims are included.


## LSTM baseline benchmark

See [LSTM_README.md](LSTM_README.md) for the separate supervised LSTM experiment, requested new battery split, audited windows and execution commands. Earlier feature-extraction outputs and splits remain unchanged.
