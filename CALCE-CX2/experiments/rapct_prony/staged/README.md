# RAPCT-Prony sequential continuation

The original prototype and all historical baselines remain preserved. This implementation extends them under `experiments/rapct_prony/staged`, with separate artifacts under `outputs/rapct_prony/staged`. Read CONTINUATION_PLAN.md, ARCHITECTURE.md and research/RELATED_WORK_VERIFIED.md in the parent experiment directory. The central recovery claim remains unsupported by the available irregular-rest data; synthetic success is not experimental capacity-recovery evidence.

Run from CALCE-CX2 in PowerShell with the existing interpreter:

```powershell
$rapctPython = 'C:\Users\parth\anaconda3\python.exe'
& $rapctPython -B -m experiments.rapct_prony.staged.prepare
& $rapctPython -B -m experiments.rapct_prony.staged.diagnostics
& $rapctPython -B -m pytest tests experiments/rapct_prony/test_stages.py experiments/rapct_prony/staged -q
& $rapctPython -u -B -m experiments.rapct_prony.staged.stage1_run
& $rapctPython -u -B -m experiments.rapct_prony.staged.stage2_run
& $rapctPython -u -B -m experiments.rapct_prony.staged.stage3_run
& $rapctPython -B -m experiments.rapct_prony.staged.single_memory_run
& $rapctPython -B -m experiments.rapct_prony.staged.memory_run
& $rapctPython -u -B -m experiments.rapct_prony.staged.finalize
& $rapctPython -B -m experiments.rapct_prony.staged.validate
```

Run preparation once; it verifies real raw hashes, preserves the original source/config/report snapshot, validates cached physical-unit curves and rechecks raw measured rest observations. The diagnostic stage independently reconstructs one interior raw-file cycle per battery. Changed raw/cache/source inputs require a new study. Do not load pickle or PyTorch artifacts from untrusted sources.

The delivered resource-limited protocol uses outer CX2_38, two inner validation cells CX2_16/CX2_33, two Optuna trials per component, pilot search epochs 3/3/8 and all final seeds 42/123/2026. Stage 1 and 2 caps are 15 epochs; Stage 3 cap is 35, with warm-up epochs 10/30 and stopping patience 20. The physics search explicitly uses a shortened 2/5 warm-up. Full 150/200-epoch, larger-trial, seven-outer-fold results are not claimed. All stage choices and hashes are frozen before the first new outer prediction. Old four-model scores have the same primary target IDs, but different information and budgets; they stay historical.

Model configuration dictionaries expose batch_size, accumulation and amp. The general trainer supports sample-weighted gradient accumulation and optional BF16 autocast; state-space/physics calculations stay FP32. Data are small cached tensors with zero DataLoader workers, avoiding unbounded Windows worker/VRAM use. Stateful physics training uses ordered segment chunks of configurable size; it carries detached states and reinitializes unknown histories at boundaries. It never shuffles individual temporal differences or resets every minibatch. Rate positivity is parameterized, not counted as a zero-valued extra penalty.

Completed identical fits are reused. Interrupted fits restore optimizer, scheduler, best model, history and Python/NumPy/CPU/CUDA RNG states from resume.pt. SQLite studies have a single-writer OS lock and requeue abandoned trials; already completed candidate fits are reused. The stateful resume test verifies identical weights with dropout. No further fitting is permitted for an outer battery after its frozen-selection marker. Generate reports and validate saved predictions without retraining:

```powershell
& $rapctPython -B -m experiments.rapct_prony.staged.report
& $rapctPython -B -m experiments.rapct_prony.staged.validate
```

For a fresh study root, select it **before importing any stage module** with the launcher; existing roots and checkpoints are not overwritten:

```powershell
& $rapctPython -B -m experiments.rapct_prony.staged.launch --output outputs/rapct_prony/new_study prepare
& $rapctPython -u -B -m experiments.rapct_prony.staged.launch --output outputs/rapct_prony/new_study stage1_run --outer CX2_16 --epochs 150 --trials 10 --search-epochs 20
```

Pass the same output root for subsequent stages. Each stage runner supports --outer and its own epoch/search budget. Outer folds are independent; run the ordered stages for each of the seven cells to extend cross-battery coverage. Broad Transformer-search helpers expose dimensions/heads/layers/FFN/dropout/length/lr/decay/batch; the pilot intentionally searches only encoder strategy first. Physics search supports explicit zero and log-scaled positive weights. Extended budgets need a consistent preregistered study plan and an updated completion report; specifying a larger epoch cap alone does not establish a complete scientific study.

Each selected model has best.pt, complete resume.pt, configuration/source/data hashes, train-only scalers, physical normalization parameters, validation/outer predictions, embeddings or attention, losses, resource measurements and standalone.pt. The standalone bundle contains parent specifications/weights and loads without the original parent files:

```python
from experiments.rapct_prony.staged.bundles import load_bundle
model, config = load_bundle('path/to/standalone.pt', device='cpu')
model.eval()
```

Final reports are artifact-derived. They distinguish every experimental fit from the 21 primary final models, three single-seed matched-length controls, short-wait conditional errors and synthetic memory fits. Seed SD is not a battery-population confidence interval. Timing includes parent-lineage training cost separately from incremental fitting and includes frozen-CNN encoding in outer device inference. Physical audits exclude unknown segment resets and record whether selected checkpoints actually reached active prior weights. Temperature, resistance, SOC, unobserved rest and electrochemical parameters are not invented.
