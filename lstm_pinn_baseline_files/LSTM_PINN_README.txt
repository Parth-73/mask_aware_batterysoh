CALCE CX2 baseline: standard LSTM vs physics-guided LSTM

1. Extract the supplied ZIP, retaining train.csv, validation.csv, test.csv.
2. Install dependencies: pip install torch numpy pandas
3. Run (PowerShell):
   python train_lstm_pinn.py --data-dir "C:\\path\\to\\pre processed ready to use data" --epochs 80 --seq-len 12

Outputs in lstm_pinn_results/: metrics.csv, training histories, prediction CSVs,
checkpoints, config.json and training-only feature scaler.

Interpretation:
- SOH is fractional. RMSE 0.01 = 1 percentage point SOH.
- Uses the frozen existing battery split; test battery is not an untouched external dataset.
- 'PINN' is a soft non-increasing SOH constraint, not an electrochemical PDE.
- Real reversible capacity recovery can violate monotonicity; vary physics_weight,
  including 0, 0.05, 0.1, 0.5, 1.0, and report this as a limitation.
- No test-set early stopping or scaling; validation selects the checkpoint.
- The script computes metrics after training; no fabricated benchmark scores.
- For publishable comparisons, run 3-5 seeds, report mean/std, and compare to
  simple non-sequence models and existing published methods on matched splits.
