# Implementation plan
1. Inventory every raw file and Channel sheet; cache normalized tables with content hashes, experimental timestamp bounds, and explicit read/schema failures.
2. Order sessions using Date_Time; reconstruct charge/rest/discharge sequences across files, independently of local Cycle_Index. Preserve resets, overlaps, gaps, partial cycles and protocol changes as quality flags.
3. Validate charge and discharge counters against signed current integration. Gate SOH on complete endpoint coverage and agreement; separate unusual discharge protocols.
4. Extract documented charging-only IC/DVA/CC/CV features, sensitivity estimates and causal historical deltas; unavailable physical measurements remain missing.
5. Pilot CX2_16, run numerical and reconstruction unit tests, then scale to available timestamped batteries.
6. Freeze battery-level splits before analysis. Rank and assess redundancy on training batteries only. F4 is provisional until future validation-performance evaluation, because model fitting is outside this task.
7. Export auditable CSV/JSON reports, plots, a notebook and execution documentation. No predictive models or neural networks are trained.
