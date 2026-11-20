# LSTM benchmark plan
1. Audit existing features/quality/labels and freeze the requested new battery split separately from prior outputs.
2. Build common complete-case cohorts for F1/F2/F3; verify consecutive ordinals, single acquisition source, timestamp continuity, and no rejected/missing records in windows. Record every exclusion. Fit feature standardization on training batteries only; preserve fractional SOH targets.
3. Implement four pure LSTM architectures and a predictor-settings literature comparison. Verify dataset boundaries, target exclusion, scaling isolation, bidirectional final-state use and attention normalization with tests.
4. Run a small deterministic CUDA smoke test. Run feature comparison (B1, length 5, all three seeds), then all four architectures at lengths 5/10/20 using the validation-selected feature set and three seeds. Include F1/two-layer/64-hidden/length-5 comparison.
5. Freeze model/feature/sequence choices using validation RMSE only. Evaluate selected checkpoints once per seed on held-out CX2_38, without further tuning.
6. Export checkpoints, fit metadata, complete window audits, metrics/seed summaries, predictions, attention, 300 DPI plots and a report/notebook. Preserve existing feature outputs and raw files.
