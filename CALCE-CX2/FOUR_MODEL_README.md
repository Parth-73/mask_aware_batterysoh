# Four-model SOH benchmark

Run from this `CALCE-CX2` directory in PowerShell. Existing raw data, feature reports, historical splits, LSTM checkpoints and experiments are never overwritten. All scientific changes live in `src/benchmark` and new output directories.

```powershell
$benchmarkPython = 'C:\Users\parth\anaconda3\python.exe'
& $benchmarkPython -B -m unittest discover -s tests -v
& $benchmarkPython -u -B -m src.benchmark.run --smoke --output outputs/soh_four_model_benchmarks_smoke_final
& $benchmarkPython -u -B -m src.benchmark.run --epochs 200 --suite all --historical --output outputs/soh_four_model_benchmarks_full
```

The default raw hash inventory location is `../../data`. Override with `--raw PATH`. Training consumes the repository's frozen validated feature/quality tables, not pickled raw caches. A sparse checkout may omit raw files and historical binary checkpoints without deleting them from Git.

The delivered reduced experiment uses `--epochs 10 --suite all --historical --output outputs/soh_four_model_benchmarks`. The full-budget command above intentionally uses a different directory. Outputs explicitly say `partial_reduced_budget`. Never interpret smoke scores as model performance. The full all-suite with historical reference has 264 training runs: 8 partitions × 3 seeds × 11 configurations. Seven-fold cross-battery only has 231. The eleven configurations are four primary models, two zero-penalty PINNs, three/nine-feature vanilla LSTM alternatives, and three matched-target sequence lengths. Primary windows use the common nine-feature complete-case cohort, so feature ablations compare input information rather than missingness populations.

Restart the **identical command with unchanged source/data/environment** to reuse completed runs and resume interrupted ones. Code/data/config/version changes require a new output directory. Best checkpoints are under `checkpoints`; complete training resume states under `runs/*/resume.pt` include optimizer, scheduler, best model, history, Python/NumPy/Torch/CUDA RNG states. Checkpoint choice is strictly inner-validation MSE; no outer score enters the training loop. A fixed single coefficient pair (0.1,0.1) is the prespecified physics setting, plus (0,0) ablations, so no coefficient search uses outer data.

Regenerate reports from completed experiments without retraining:

```powershell
& $benchmarkPython -B -m src.benchmark.run --report-only --output outputs/soh_four_model_benchmarks
& $benchmarkPython -B validate_four_model_artifacts.py outputs/soh_four_model_benchmarks
```

The report generator reads only completed `result.json` records; missing runs are counted explicitly. Run the independent validator after report generation to verify every prediction/window, all six primary metric formulas, cohort matching, checkpoint hashes and scaler provenance. It also attaches explicit source configuration/fold/seed lists to aggregate tables and appends measured ablation/failure-case interpretation to the report. Primary tables macro-average seed means equally across held-out batteries; seed sample standard deviations are reported separately. Every individual prediction is identified by architecture, configuration, fold, seed, feature set, original cycle ID and input window IDs. Validation recommendations average inner-validation RMSE, never outer-test scores. Parameter counts and synchronized GPU timing are measured, not estimated.

The PINN names denote a **soft latent-aging regularizer**, not a validated physical differential equation. Its state is not guaranteed globally monotone or identifiable from these labels. No rest duration is inferred from recording gaps, no temperature/resistance feature is activated, and no Neural Prony architecture is implemented. Read `FOUR_MODEL_IMPLEMENTATION_PLAN.md` for the audit and `reports/BASELINE_COMPARISON_REPORT.md` for measured limitations and execution status.
