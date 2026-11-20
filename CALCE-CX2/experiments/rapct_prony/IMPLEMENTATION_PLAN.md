# RAPCT-Prony implementation plan

Prediction cutoff: last available charging observation of the target cycle. Target: subsequent validated discharge integral / 1.35 Ah, without smoothing or clipping.

1. Audit: hash actual raw files and historical configurations, inspect source and historical cohorts, record installed packages/GPU, inventory measured zero-current intervals separately from gaps. Review accessible primary literature and mark inaccessible details explicitly.
2. Data gate: use unchanged reconstruction and IC/DVA extraction. Rebuild isolated channel caches from raw Excel. Verify reconstructed cycle IDs/targets against saved quality records. Cache voltage-aligned Q, dQ/dV and I only with full measured coverage. Preserve rejection reasons and end-of-charge cutoff.
3. CNN gate: test leakage, interpolation, scaling and model shapes. Train five-feature MLP, curve CNN and fusion on matched complete cases. Save immutable run folders, train-only scalers, checkpoints, predictions, losses and metrics. Initial runs are pilots, not nested final results.
4. Transformer gate: only after completed CNN gate, add causal five-cycle transformer; reject ordinal/session/clock discontinuities. Compare frozen and trainable CNN on matched target cycles.
5. Degradation gate: only after transformer execution, add nonnegative latent aging recurrence with unknown initial state. Use independent rate smoothness rather than a recurrence residual against itself. Compare zero/nonzero regularization.
6. Recovery gate: quantify verified rest support and matched discharge conditions on independent batteries. No event inferred from capacity rebound. If support is insufficient, experimental recovery training remains unavailable; synthetic memory tests remain permissible.
7. Memory gates: implement one-mode, then multi-mode positive exponential memory with explicit state carry/reset and bounded capacity penalty. Derive experimental timescales from training rests only. Synthetic timescales have no electrochemical interpretation.
8. Full hybrid gate: fine-tune only after preceding stages pass, retain all intermediate results and test frozen/unfrozen encoders.
9. Optimization gate: SQLite Optuna TPE/MedianPruner, sequential component searches, inner battery folds inside every outer fold. Train selected settings with seeds 42, 123, 2026. Outer results never select settings.
10. Report: distinguish completed experiments, synthetic validation, untested code and data/compute blockers. Produce figures and matched comparison tables only from real artifacts.

All generated files belong under outputs/rapct_prony. No historical checkpoint or preprocessing output is overwritten. Runs must fail on collisions. Do not manufacture empty result tables implying completed experiments. Execution status is recorded in reports/RAPCT_PRONY_FINAL_REPORT.md.
