# Verification against the supplied AW-PINN paper

[Source PDF](../../../batteries-12-00115.pdf): Wang et al., Batteries 2026, 12(4), 115; DOI 10.3390/batteries12040115. Page references below are the 1-based printed PDF pages. The PDF has 21 pages. Its SHA256 is recorded in literature_verification.json.

**The F1/B2/length-5 comparison matches the predictor dimensions in Table 3 (p. 9): input 3, sequence 5, layers 2, hidden size 64, output 1.** This verifies the SOH predictor branch only, not the complete parallel predictor/weight-generator AW-PINN model.

| Aspect | Paper | Implemented benchmark | Assessment / source |
| --- | --- | --- | --- |
| Predictor input dimension | 3 | 3 (F1) | matches; p. 9, Table 3 |
| Predictor sequence length | 5 | 5 | matches; p. 9, Table 3 |
| Predictor LSTM layers | 2 | 2 stacked unidirectional LSTM layers (B2) | matches; p. 9, Table 3 |
| Predictor hidden size | 64 | 64 | matches; p. 9, Table 3 |
| Predictor output dimension | 1 | 1 linear SOH output | matches; p. 9, Table 3 |
| Feature types | IC peak; charged capacity; voltage slope | ic_peak_height, partial_charge_ah, partial_dv_dq | matches feature types not exact numerical extraction; p. 5,8, Section 2.2 and Section 3.1 |
| Partial-charge voltage interval | [V_cutoff - 0.4 V, V_cutoff - 0.2 V]; 3.8-4.0 V at 4.2 V cutoff | 3.8-4.0 V | matches; p. 4,5, Table 1; Equations 3-4 |
| IC smoothing | Gaussian filter, sigma=1 | Cubic Savitzky-Golay, 11 samples on a 2 mV voltage grid; 7/11/15 sensitivity checks | differs; p. 6, Section 2.2 |
| IC peak search domain | Exact peak-search interval and resampling resolution not specified in the main text | 3.80-4.15 V | paper detail not specified; p. 4,5,6, Section 2.2 |
| Outlier handling | Remove obvious outliers; correct abnormal points using trend fitting/deviation analysis | Keep all candidates in audit tables; reject unreliable inputs/windows; no across-cycle trend filling | differs; p. 6, Section 2.2 |
| Feature normalization | Map feature values to [0,1]; exact fitting partition not stated | StandardScaler fitted on training batteries only | differs; p. 6, Section 2.2 |
| SOH denominator | Current capacity divided by nominal capacity; CX2 nominal=1.35 Ah | Validated discharge current integral / 1.35 Ah; counters cross-checked | matches denominator capacity validation not reproduced; p. 4, Equation 1; Table 1 |
| Aging domain | Focus on aging from initial state to 80% SOH | No 80% cutoff; retained CX2_38 targets cover approximately 47.2%-99.7% SOH | differs; p. 4, Section 2.1 |
| Evaluation split | Within each battery: first 50% training, latter 50% testing | Battery-level split; validation CX2_36, held-out test CX2_38 | differs; p. 12, Section 4.1 |
| CX2 discharge protocol summary | Table 1 lists 0.5C | Observed validated routine records include approximately 0.4C, 0.5C and 1C; protocol is audited per cycle | protocol equivalence not established; p. 4, Table 1 |
| Physical branch and loss | Separate weight-generator LSTM and monotonicity loss; GWO tuning of global physical weight | Supervised predictor only, MSE; excluded by task | intentionally excluded; p. 8,9,10,11, Section 3; Equations 12-16 |
| Optimizer and regularization | Optimizer, learning rate, weight decay, dropout, batch size, epochs and early stopping not specified in searchable main-text settings/Table 3 | AdamW, lr=0.0005, weight decay=0.001, dropout=0.2, batch=64, max=200, patience=20 | benchmark choices not claimed to match paper; p. 8,9,12, Reported predictor/settings and experimental description |
| Metric reporting | Equation 26 has MAPE multiplied by 100; Table 4 uses unlabeled small decimal MAPE values. Equation 25 defines MSE while result tables report MAE. | Separate MAE/MSE/RMSE with explicit fractional and percentage-point units; MAPE reported as percent | published units require care no automatic rescaling; p. 12,13,17, Equations 24-26; Tables 4-6 |

## Implications for interpreting the completed experiment

The 3.8-4.0 V partial-capacity and voltage-slope interval is confirmed: the paper defines the interval relative to the 4.2 V charging cutoff (p. 5, Equations 3-4). The IC peak is the same feature type, but exact numeric equivalence is not established because the smoothing, peak-search interval and outlier handling differ or are unspecified.

The paper normalizes features to [0,1] (p. 6). Our StandardScaler was deliberately fitted on training batteries only. The paper does not state which observations fit its normalizer or trend correction; no claim of leakage in the paper is inferred from that omission. Dropout and optimization settings cannot be reconstructed from unreported values; the benchmark choices remain explicitly documented.

The paper focuses on aging down to 80% SOH (p. 4) and trains/tests chronologically within each battery using a 50/50 split (p. 12). Our independent held-out battery design and retained test domain of approximately 47.2%-99.7% SOH answer a different question. Our window rejection, complete-case filtering and session continuity also alter the evaluated cohort. Direct error rankings against the publication are therefore not valid.

Table 1 lists a CX2 discharge rate of 0.5C. The actual records in our selected batteries include about 1C for CX2_34/36/37/38, 0.5C for CX2_33/35, and 0.4C for CX2_16. This is a protocol-equivalence limitation, not evidence that the paper used the same raw steps or that our measurements should be replaced by the table summary. Per-battery measured medians are saved in literature_verification.json.

Metric labels also need care: Equation 26 includes the factor 100 for MAPE, while the result tables show small decimal values without an explicit percent label. Equation 25 defines MSE, whereas the tables report MAE. We retain separately labeled MAE, MSE, RMSE and percent MAPE and do not guess a conversion or assert that the publication is numerically wrong.

The Table 3 predictor settings are not an independently specified architecture for every pure-LSTM result in Table 4. The implemented comparator should be called a supervised LSTM matching the reported AW-PINN predictor dimensions, rather than an exact reproduction of the published pure-LSTM benchmark.

## Frozen results and scope

This verification was performed after the original test results were committed. No model, feature set, scaler, data split, checkpoint, epoch, prediction or test metric was changed. The as-run experiment_config.json and final_selection.json remain immutable; their literature metadata records what was known when training began. Updated provenance is in literature_verification.json. No models were retrained and no inference was repeated.

The uploaded PDF does not require a dimensional architecture correction to the existing F1/B2/length-5 comparator. Matching the paper's entire preprocessing and within-battery evaluation would be a separate experiment, not a post-hoc revision of the completed unseen-battery benchmark.
