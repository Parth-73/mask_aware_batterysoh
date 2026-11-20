Input: pre processed ready to use data/train.csv (training rows only).
Source end times: benchmark-repository/CALCE-CX2/outputs/cx2_features_all.csv.
Method: sort within battery by original cycle ordinal. Compare SOH of consecutive retained rows.
Candidate long rest: at least 6 hours between prior cycle END and current cycle START, with consecutive original cycle ordinals.
Material change: absolute SOH change at least 0.2 percentage points (an exploratory threshold, not a calibrated noise limit).
Fluctuation flag: next consecutive training cycle reverses direction by at least 0.2 percentage points.
Missing original cycles are exported separately: elapsed time across them cannot be interpreted as rest.
Gaps are unobserved intervals, not verified zero-current rest. Source changes, protocol, temperature and measurement effects can explain changes. Causation is not established.
SOH is discharge capacity / 1.35 Ah. All changes are percentage points, not relative percent.
