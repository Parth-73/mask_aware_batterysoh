# Paper-informed CNN-LSTM experiments

Open `../CALCE_CNN_LSTM_Paper_Informed.ipynb` for the full implementation and executed results.

## Observed result

Across all three seeds, charging validation RMSE improved from 0.808 to 0.643 percentage points, but matched test RMSE worsened from 1.011 to 1.590 percentage points. The existing original-features notebook remains the better observed charging baseline. The separate completed-discharge diagnostic had mean test RMSE 1.751 percentage points on 574 windows and is not directly comparable to the 662-window charging result. No seed is omitted. Six saved checkpoints, all notebook code cells and 18 split/seed metric rows were verified.

These experiments did not reproduce the paper's lower errors. Matching the original CS2 data and feature extraction is necessary before attributing that difference to architecture alone. Additional tuning would need a predefined validation protocol and a fresh final evaluation set, since the current test results have already been inspected.

The provided `sustainability-17-04014-v2.pdf` was read locally using QtPdf; its extracted text is `../sustainability_paper_text.txt`. This work uses no external paper summary.

## Findings from the paper

- Pages 6 and 14: nominal 1.1 Ah CS2 cells; train CS2_34/36 and test CS2_33/35, with a discussion of aging to 80% SOH. Our available validated cells are CX2 with capacity divided by 1.35 Ah and the original exported cell-disjoint split. We do not relabel CX2 as CS2 or change the target denominator to force numerical agreement.
- Pages 9–10: selected H1–H3 use the completed discharge. H1 is time at minimum discharge voltage; H2 is average discharge time; H3 is average discharge voltage. Extraction equations and time-origin conventions are not specified.
- Page 12: three convolution stages, dimensions written 3×3, 5×4 and 9×32, LSTM hidden size 256, dropout, fully connected and regression layers. Figure 10 shows two LSTM layers and pooling but does not fully specify implementation geometry; softmax in the figure conflicts with the regression description.
- Pages 8–9: K-means is used for outliers. The exact rule converting clusters to rejected records is absent. We retain validated targets and do not delete true recoveries or smooth SOH to lower error.
- Tables 4–5 report RMSE 0.339 and 0.239 percentage points for CS2_33/35. These are not reference errors for our CX2_38 windows.

## Experiments

`charging/` uses the original five charging features, rows and existing battery split. Its 662 test target windows exactly match the prior notebook. `matched_charging_comparison.csv` compares each seed on identical IDs and targets.

`discharge_diagnostic/` uses explicit approximations to H1–H3 from the raw local channel caches. These features become available after the same-cycle discharge. Discharge duration can almost determine capacity under a fixed current. This experiment is retrospective estimation with a later prediction cutoff, not a better charging-only estimator or future forecast.

Both use 12-cycle windows, consecutive original ordinals, unchanged acquisition source and increasing timestamps. Neither crosses missing-feature rows. Scalers are fitted on training observations only. Targets remain unchanged fractional SOH. Model output is expressed through training-only target mean and standard deviation. A linear current-feature skip accompanies the serial CNN/two-layer LSTM; this skip is an engineering addition, not a reported paper component.

Fixed choices before evaluation: Conv1d kernels 3/5/9 with channels 3/4/32, same padding, ReLU and same-length max pooling; two LSTM layers with 256 hidden units and 0.1 dropout; linear regression output; AdamW at 0.001 and weight decay 0.0001; gradient norm limit 1; batch size 128; maximum 150 epochs; patience 30; ReduceLROnPlateau factor 0.5, patience 7 and minimum learning rate 0.00001. Three seeds: 42, 123, 2026. Validation MSE selects checkpoints. All seeds are reported; no test-based seed selection is performed.

The paper does not report several of these settings. This is a documented adaptation, not an exact reproduction or a claim to match the published scores. Previously inspected validation/test data also limits independent generalization claims.

## Re-running

The project base Anaconda environment has pandas 3 and pyarrow needed to read the cached raw pickles: run `extract_paper_discharge_features.py` there if re-extraction is needed. The generated CSV is already present. Train using the notebook's existing ISLES Research kernel, with numpy, pandas, scikit-learn, matplotlib and PyTorch. `create_paper_informed_notebook.py` regenerates the unexecuted notebook; `execute_paper_informed_notebook.py` executes and saves it, preserving partial outputs on failure. Regeneration replaces the notebook's execution outputs.
