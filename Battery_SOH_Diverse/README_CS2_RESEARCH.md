# CALCE CS2 research notebook

Open `CALCE_CS2_Feature_Research.ipynb` in VS Code and select `.venv/Scripts/python.exe`.

Run all notebook cells, or execute the complete resumable pipeline from this folder:

```powershell
.\.venv\Scripts\python.exe run_cs2_research.py
```

Research outputs, measured results, plots, reports, splits and checkpoints are in `CS2/cs2_feature_research`. `output_index.csv` lists file paths and hashes. The original preprocessing notebook and outputs are preserved.

The five-feature set is provisional: it did not beat charging/persistence baselines on the untouched test cells. SOH is measured delivered capacity at the recorded discharge rate; operating protocols are not normalized. Read `FEATURE_ENGINEERING_REPORT.md`, `DATA_QUALITY_AUDIT.md` and `EXPERIMENTAL_PROTOCOL.md` before using the data. Unresolved CADEX clocks and unknown workbook boundaries stay flagged. Extreme measurements are retained for physical review.

Missingness masks account for phase dependencies. Raw charging prefixes/suffixes and historical-vector dropout are separate, bounded robustness experiments. None establishes physical acquisition invariance. Voltage-window throughput uses observed intervals bracketing the thresholds, reflected in its `throughput_bracket_` name.

The notebook loads completed model results by default. Set `RECOMPUTE_ANALYSIS` and `RECOMPUTE_ROBUSTNESS` to True after changing the analysis. Extraction checkpoints depend on source metadata and configuration; increment the extraction configuration version after changing reconstruction semantics. Dependencies are pinned in `requirements-cs2-research.txt`.
