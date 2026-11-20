# Reproducible LSTM benchmark

This benchmark uses the existing audited features; it preserves the raw dataset and previous feature reports. It implements pure Vanilla, Stacked, Bidirectional and Attention LSTMs. No CNN, physical penalty, chemistry constraint or optimizer search is added.

Run in PowerShell from `D:\battery_soh\CALCE-CX2`:

```powershell
& D:\battery_soh\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
& D:\battery_soh\.venv\Scripts\python.exe -u -m src.training.train_lstm --smoke
& D:\battery_soh\.venv\Scripts\python.exe -u -m src.training.train_lstm --stage train
& D:\battery_soh\.venv\Scripts\python.exe -u -m src.training.train_lstm --stage test
& D:\battery_soh\.venv\Scripts\python.exe -u -m src.training.train_lstm --stage report
```

`--stage all` performs train, test and reporting in order. The separate commands make the validation-selection freeze reviewable before test exposure. `--output PATH` creates an isolated experiment directory. A completed run is reused; an interrupted training run restarts that seed/configuration from scratch. Existing results with changed configuration, data hashes or training code are rejected. No new training is allowed after the test-evaluation marker appears. Reporting reads saved predictions and does not evaluate models again.

CUDA-enabled PyTorch is required. The installed environment already provides it; no CPU substitution is made. NumPy, pandas, scikit-learn and Matplotlib are also required. Library/CUDA/cuDNN versions and GPU details are saved in `experiment_config.json`. Torch is intentionally not added to the older feature-only requirements file because selecting the correct CUDA build is environment-specific.

## Splits and feature definitions

The benchmark split overrides the older feature-analysis split **only within this experiment**: training CX2_16/33/34/35/37; validation CX2_36; test CX2_38. CX2_3/8 are reserved for cross-protocol studies and CX2_32 is excluded. The old split JSON is unchanged.

The benchmark's F1/F2/F3 names are task-specific; they are not the earlier pipeline's F1/F2/F3/F4 definitions. They are explicitly frozen in `src/training/config.py` and `experiment_config.json`:

- F1: IC peak height, partial charge capacity and partial voltage slope.
- F2: F1 plus IC area, peak voltage, peak width, CC duration, CV duration and DVA peak (nine features).
- F3: CC duration, CV duration, DVA peak, IC peak voltage and IC peak width (five features).

No feature is silently substituted. All three configurations use the same complete-case nine-feature cohort, so Experiment A compares information on identical target samples. `cycle_counts.csv` also reports individually usable feature records to reveal the common-cohort restriction. No imputer is used. StandardScaler fits only common eligible training-cycle inputs. The target stays an unscaled fraction: validated discharge Ah / 1.35 Ah. Prediction t estimates the eventual SOH label of cycle t from charge observations through t, rather than forecasting t+1.

## Windows and continuity

Candidates are checked against `cycle_quality.csv` and `rejected_cycles.csv`, and capacity counters/integration are rechecked. Only complete, routine/conventional cycles with reliable positive SOH labels enter windows. Every window uses original consecutive ordinals, one battery, one source sheet, no rejected or missing-feature record, no local counter reset, and nonoverlapping timestamps with inter-cycle gaps <=3600 seconds. Cross-file cycles are conservatively rejected for this benchmark even if the feature pipeline could validate them. Windows do not jump over rejected cycles. The original features are not interpolated across lifetime gaps.

Per-target decisions are saved in `window_audit.csv`; per-cycle exclusions are in `cycle_audit.csv`. `input_cycle_ids` and `input_ordinals` in prediction/window files permit exact provenance checks. Window length changes eligible target counts. The report includes the natural cohort and an identical-target intersection for validation sequence comparisons.

## Architectures and experimental selection

B1 has one 64-unit layer and zero dropout. B2 has two 64-unit layers with 0.2 inter-layer dropout. B3 has the same two layers and dropout, bidirectional only within the available window; it concatenates the last-layer final forward and reverse states. B4 has one 64-unit LSTM and additive attention, `a_t = softmax(v^T tanh(W h_t+b))`, followed by a weighted context and linear SOH head. B4 choices are explicit because the task did not prescribe its layer count or attention scoring rule. Attention weights are saved per input cycle and are descriptive, not a causal attribution.

A compares all feature sets using B1/length 5 and three seeds. Then every architecture is evaluated at lengths 5/10/20 with the validation-selected feature set, again using seeds 42/123/2026. B selects architecture at length 5; C selects sequence length for that architecture; D reports seed stability. F1/B2/length-5 is additionally run as a predictor-settings literature comparison, reusing it if already in the grid. Every configuration uses the same optimization settings from the task. The staged selection rule is fixed in advance and ranks seed-mean validation RMSE. The best individual seed is not selected.

`final_selection.json` freezes choices and checkpoint hashes before any test evaluation. Final selected and literature-comparison checkpoints are each evaluated once per seed on CX2_38. If they refer to the same checkpoint it is evaluated only once. A partial test evaluation that started without committing a result is blocked rather than silently repeated. No test metric feeds training or the frozen choices.

Metrics include MAE/RMSE/MSE on fractional and percentage-point scales, relative MAPE in percent, R-squared, timing, parameter count and peak GPU memory. Standard deviations use ddof=1 across three seeds. Model epochs are chosen by validation RMSE, with patience 20, max 200 epochs and a 1e-6 fractional improvement threshold. A scheduler halves LR after five plateau epochs to a 1e-6 minimum. Target and predictions are not clipped. GPU timing uses synchronization, synthetic warmup and a single forward pass; small timing measurements can vary.

## Research interpretation

The earlier provisional compact-feature analysis included CX2_36, now the requested validation battery. F3 is retained exactly as the user-prespecified candidate rather than reselected. Current validation is therefore not fully independent of that exploratory feature analysis. CX2_38 was excluded from the earlier selection and remains the independent test battery. The report does not present F3 as a validated optimum.

The supplied [Wang et al. paper, DOI 10.3390/batteries12040115](https://www.mdpi.com/2313-0105/12/4/115) is now verified: Table 3 (p. 9) confirms input 3, length 5, two layers, hidden size 64 and output 1; p. 5 confirms the 3.8-4.0 V partial-charge interval. The implemented F1/B2/L5 comparison matches those predictor dimensions. It remains a supervised MSE baseline, with no weight generator or physical constraints. The paper uses Gaussian smoothing (sigma 1), [0,1] feature normalization, a within-battery chronological 50/50 split and aging down to 80% SOH. Our extraction, scaling, battery-level split and lifetime coverage differ. Dropout/optimizer settings are not reported in the searchable main-text settings. See [the page-referenced comparison](outputs/lstm_baselines/LITERATURE_COMPARISON.md) and literature_verification.json. No direct published-performance superiority claim is valid. The frozen as-run experiment metadata, checkpoints and all results remain unchanged.

Deterministic algorithms, CUDA seeds, cuDNN settings and cuBLAS workspace are configured. See [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html); cross-version/hardware bitwise reproducibility is not guaranteed. Overlapping windows are serially dependent, and only one validation/test battery is used. Seed SD is not population uncertainty.

## Outputs

`outputs/lstm_baselines` contains the requested results, validation/test tables, per-window predictions, epoch histories, experiment configuration, checkpoints, 300-DPI PNG and vector PDF figures, attention weights, seed summaries, audits and `BASELINE_REPORT.md`. The notebook `notebooks/lstm_baseline_analysis.ipynb` reads these saved outputs without retraining or reopening test evaluation.
