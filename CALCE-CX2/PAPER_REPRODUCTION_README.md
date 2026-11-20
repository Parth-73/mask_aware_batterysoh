# Paper-style pure-LSTM reproduction

This is a separate experiment under `outputs/paper_lstm_reproduction`. It preserves the completed unseen-battery benchmark. It follows the reported Wang et al. 2026 predictor methodology for CX2_34, CX2_36, CX2_37 and CX2_38; CS2 and Tongji datasets are not available locally.

Reported settings: IC peak plus charge capacity and voltage slope over 3.8-4.0 V; Gaussian IC filtering with sigma 1; [0,1] feature normalization; aging down to 80% SOH; first-half training and latter-half testing within each battery; three inputs, sequence length 5, two LSTM layers, hidden 64 and one output. No physical branch, PINN loss, CNN or GWO is implemented.

The paper omits details necessary for exact numerical reproduction. `experiment_config.json` records each assumption before tests. We use a uniform 2 mV voltage grid, an established 3.8-4.15 V primary IC peak interval, and a full observed CC interval as a sensitivity alternative. Gaussian differentiation/smoothing is recomputed from existing normalized raw-row caches. Existing validated labels and cycle IDs are reused; Excel is not downloaded or read again and cycles/labels are not rebuilt.

The chronological aging prefix stops at the first validated routine SOH below 0.8. No recovery cycles later in life reenter the dataset. SOH is discharge capacity divided by nominal 1.35 Ah and is neither clipped nor smoothed. Invalid or missing charge features are rejected, never filled.

The paper describes trend-based abnormal-point correction without its algorithm. Our explicit rule fits a robust quadratic trend and residual thresholds using training features only. Only an observed point with BOTH a large training-trend residual and an abrupt measured adjacent jump is corrected. Partial slope is then recalculated from corrected partial capacity. All observed/raw/corrected values, fitted coefficients and exclusions are retained in `datasets`.

MinMaxScaler fits the permitted training/fitting rows only. Test values can fall outside [0,1]; no test fitting or clipping is performed. Windows preserve original ordinal adjacency, single-source continuity and recorded time continuity. They may use earlier training-half feature observations for a later-half target, never future cycles or SOH inputs. The paper does not specify these continuity rules or its exact normalization-fitting scope.

Two **pre-test** training-budget assumptions are reported rather than selecting a budget from test error:

1. Internal-validation budget: the last 20% of the first half chooses epochs and an LR schedule. A fresh model with refitted training-only preprocessing is then trained on the complete first half for that budget. Adam lr=0.001, no weight decay/dropout, MSE, batch 64, max 200 epochs and patience 20 are documented unreported choices. The two IC-domain variants and three seeds give 24 final predictors.
2. Fixed-200 budget: an independently frozen, constant-LR Adam fit uses the complete training half for exactly 200 epochs, without validation-based stopping. This tests the paper's unreported epoch policy after some internal validations selected only 7-27 epochs. It was registered before either experiment's test evaluations. It adds 12 final predictors for the primary IC interval, under the `fixed_epoch` subdirectory.

All models are frozen before single-pass evaluation. Published Table 4 values are contextual references; there is no search to match them, selection by test score, or claim of superiority over the paper. Three seeds (42/123/2026) measure optimization variation. The reported pure-LSTM baseline architecture itself is not separately specified in the paper, so Table 3's SOH-predictor settings are the closest available description.

## Commands

Run from `D:\battery_soh\CALCE-CX2`:

```powershell
& D:\battery_soh\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
& D:\battery_soh\.venv\Scripts\python.exe -m src.paper_reproduction.run --stage prepare
& D:\battery_soh\.venv\Scripts\python.exe -m src.paper_reproduction.run --smoke
& D:\battery_soh\.venv\Scripts\python.exe -m src.paper_reproduction.run --stage train
& D:\battery_soh\.venv\Scripts\python.exe -m src.paper_reproduction_fixed_epochs --stage train
& D:\battery_soh\.venv\Scripts\python.exe -m src.paper_reproduction.run --stage test
& D:\battery_soh\.venv\Scripts\python.exe -m src.paper_reproduction_fixed_epochs --stage test
& D:\battery_soh\.venv\Scripts\python.exe -m src.paper_reproduction.run --stage report
```

Preparation is for the initial run only; rebuilding data after freezing is blocked. Completed models/evaluations are cached. Use a fresh `--output` (and matching `--parent` for fixed epochs) for changed configurations. Do not add training or tune choices after reading test metrics.

`REPRODUCTION_REPORT.md`, `published_comparison.csv`, `seed_summary.csv`, per-seed test metrics, predictions, epoch histories, split/correction/window audits, checkpoints and 300-DPI PNG/vector PDF plots are saved in the new output directory. The internal-validation and fixed-epoch policies remain separately identifiable. `notebooks/paper_lstm_reproduction.ipynb` reads saved artifacts without retraining.
