# RAPCT-Prony isolated experiments

Run commands from CALCE-CX2 with the existing compatible interpreter:

```powershell
$python = 'C:\Users\parth\anaconda3\python.exe'
& $python -m experiments.rapct_prony.audit audit
& $python -m experiments.rapct_prony.audit prepare
& $python -m experiments.rapct_prony.finalize_audit
& $python -m experiments.rapct_prony.events
& $python -m pytest tests/test_pipeline.py experiments/rapct_prony/test_stages.py -q
& $python -m experiments.rapct_prony.train --pathway all --epochs 30 --label reduced_budget_pilot
& $python -m experiments.rapct_prony.report
```

The audit defaults to C:/Projects/Battery_Soh/data relative to this checkout; --raw can override it. Preparation uses python-calamine for newly read Excel tables, reuses compatible isolated channel caches and checks every accepted reconstructed target against the saved historical validated table. Parser changes cannot silently change labels. Channel caches are trusted local pickle files, not portable interchange formats.

Optional dependencies were installed into vendor/ without modifying the base environment. The package adds that directory to its import path. Recreate it with `python -m pip install --target experiments/rapct_prony/vendor --no-deps optuna python-calamine sqlalchemy alembic colorlog Mako greenlet`; complete dependency versions are recorded in outputs/rapct_prony/configs/environment.json. Required base packages remain those in the project environment. vendor/ is ignored by version control.

Every completed training run owns an immutable config-derived directory under outputs/rapct_prony/checkpoints. Identical configurations fail on collisions to protect results; use a distinct label for a scientifically distinct rerun. Each run includes config, explicit battery splits, train-only scaler objects and physical-unit normalization parameters, best checkpoint, validation/outer predictions and embeddings, epoch losses, and measured metrics/resources. Outputs with no completed experiment are not fabricated.

The initial fixed protocol trains on CX2_16/33/34/35/37, validates on CX2_36, and evaluates on CX2_38. All three Stage-1 pathways use identical complete-case curve samples. This differs from historical cohorts and does not constitute final nested evaluation. An outer-battery score cannot select downstream architecture or loss weights. For final scientific conclusions use the nested protocol in IMPLEMENTATION_PLAN.md and all three specified seeds.

Read outputs/rapct_prony/reports/RAPCT_PRONY_FINAL_REPORT.md for actual execution status. ARCHITECTURE.md distinguishes Stage-1 code from later specifications. research/RELATED_WORK.md documents related methods and verification gaps. Synthetic tests cannot establish experimental recovery prediction.

New last.pt checkpoints include optimizer, scaler, Torch/CUDA RNG and DataLoader generator states. Resume such a run with `python -m experiments.rapct_prony.train --resume outputs/rapct_prony/checkpoints/RUN_DIRECTORY`. A synthetic equivalence test compares interrupted and uninterrupted updates. Older checkpoints lacking the loader RNG require an explicit `--initialize CHECKPOINT --label weights_continuation` run with a fresh optimizer; this is recorded separately.

Additional executed-stage commands (use a new label for a new immutable run):

```powershell
# Source checkpoint is the completed Stage-1 fusion artifact.
& $python -m experiments.rapct_prony.train --stage transformer --pathway fusion --encoder-checkpoint outputs/rapct_prony/checkpoints/fusion_1779a8d18fc6/best.pt --epochs 20 --frozen --label temporal_frozen_reproduction
& $python -m experiments.rapct_prony.train --stage transformer --pathway fusion --encoder-checkpoint outputs/rapct_prony/checkpoints/fusion_1779a8d18fc6/best.pt --epochs 20 --label temporal_joint_reproduction
& $python -m experiments.rapct_prony.train_physics
& $python -m experiments.rapct_prony.synthetic_memory
& $python -m experiments.rapct_prony.tune --outer CX2_38 --inner CX2_36 CX2_37 --trials 10 --epochs 5 --study-label cnn_pilot10
& $python -m experiments.rapct_prony.finalize_results
& $python -m experiments.rapct_prony.report
& $python -m experiments.rapct_prony.figures
```

`train_physics` chooses the completed Transformer with minimum inner validation RMSE for the fixed CX2_38/CX2_36 protocol, then runs zero/nonzero regularization over full segments. Existing identical output directories fail rather than overwrite. Synthetic memory likewise preserves existing fitting runs. Every synthetic artifact is labeled and kept out of experimental all_runs.csv.

`load.load_model(checkpoint, device='cpu')` restores a trained Stage-1, Transformer or degradation branch without reloading its original pretraining initialization file. Restore the stored train-only scalers before inference. A degradation branch consumes frozen Transformer representations and a valid chronological segment, with its source encoder recorded in config.json. Hex configuration identifiers must be parsed as strings; metrics.read_metrics enforces this.

The tuning command can enumerate all seven outer batteries and all remaining inner folds if --outer/--inner are omitted. The actual executed 10-trial study uses one outer/two inner batteries and five inner epochs. Refits use all six non-outer cells for a fixed inner-selected epoch count and all three seeds. SQLite and sampler state are resumable, exact numeric configurations are reused, and outer scores never choose parameters. The ten-trial pilot has only three fully completed trials and does not establish TPE convergence or reliable parameter importance. Other component Optuna searches are not implemented; full nested evaluation remains incomplete.

The repository's original outputs/battery_splits.json changed after the initial snapshot and contains CX2_{35. It was not written/restored by this experiment. New experiment splits are explicit and independent. See data_audit/historical_preservation_check.json and the frozen snapshot.
