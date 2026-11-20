# Mask-aware battery state-of-health experiments

On held-out CALCE cell CX2_38, training with masks improved five-cycle-ahead forecasting in all three conditions shown below. The acquisition-consistency model (AIDO) achieved the lowest error when constant-current charging features were unavailable. The masked transformer achieved the lowest error with clean inputs and with a block of history missing.

| Input condition | Clean transformer | Masked transformer | AIDO |
|---|---:|---:|---:|
| Clean | 2.44 | **1.33** | 1.44 |
| Block of history missing | 4.36 | **1.43** | 1.60 |
| Constant-current features missing | 8.64 | 2.08 | **1.97** |

Values are test RMSE in SOH percentage points, averaged over seeds 42, 123 and 2026. The three seeds measure training variation on one test cell. The [full AIDO results](pre%20processed%20ready%20to%20use%20data/aido_soh_results/fixed_full/summary.csv) also cover horizons of one and three cycles, additional masks, MAE and interval coverage. [Per-seed paired comparisons](pre%20processed%20ready%20to%20use%20data/aido_soh_results/fixed_full/paired_improvements.csv) show how consistently the models improved.

The separate [multi-model benchmark](pre%20processed%20ready%20to%20use%20data/final_seven_model_results/fixed_full/summary.csv) ranked the GWO-tuned MLP first on clean five-cycle test forecasts at 1.32 percentage points RMSE. Compare models within each run's protocol. On the [CS2 feature study](Battery_SOH_Diverse/CS2/cs2_feature_research/FEATURE_ENGINEERING_REPORT.md), the five selected features did not beat charging and persistence baselines on untouched test cells. The [CS2 protocol](Battery_SOH_Diverse/CS2/cs2_feature_research/EXPERIMENTAL_PROTOCOL.md) describes how the study formed held-out splits and forecast pairs.

## Evidence by experiment

| Experiment | Saved evidence |
|---|---|
| CX2 acquisition consistency and provenance-aware forecasting | [AIDO notebook](AIDO_SOH_Acquisition_Consistency_Research.ipynb), [AIDO runs](pre%20processed%20ready%20to%20use%20data/aido_soh_results), [PARDON smoke run](pre%20processed%20ready%20to%20use%20data/pardon_soh_results/smoke) |
| CX2 data preparation and quality | [Five-feature export and split definition](pre%20processed%20ready%20to%20use%20data/README.md), [raw-feature study](calce_raw_feature_research), [rest-event audit](train_rest_analysis), [CX2 extraction report](CALCE-CX2/outputs/REPORT.md) |
| CNN-LSTM and physics-informed baselines | [Original features](cnn_lstm_original_features_results), [paper-informed comparison](cnn_lstm_paper_informed_results/README.md), [LSTM-PINN](notebook_lstm_pinn_results), [five-model runs](soh_five_model_results) |
| Forecasting and robustness | [Simple baselines and ablations](soh_research_gap_benchmark), [latent forecasting](latenttsf_soh_results), [cross-cell ablations](latent_lobo_results), [Neural Prony comparisons](neural_prony_five_model_results) |
| Architecture and optimization studies | [FlowState, GWO, seven-cell and transformer-PINN studies](pre%20processed%20ready%20to%20use%20data), [CX2 baseline and staged RAPCT-Prony reports](CALCE-CX2/outputs) |
| CS2 missing observations and acquisition robustness | [Study guide](Battery_SOH_Diverse/README_CS2_RESEARCH.md), [quality audit](Battery_SOH_Diverse/CS2/cs2_feature_research/DATA_QUALITY_AUDIT.md), [results and plots](Battery_SOH_Diverse/CS2/cs2_feature_research) |

This archive contains saved experiment outputs. Directory names distinguish full runs, pilots and smoke tests. Raw workbooks, binary model weights, extraction caches and local Python environments are outside the archive. Some original manifests record paths on the machine where training ran.
