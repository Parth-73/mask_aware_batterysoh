"""PowerShell: python -m src.benchmark.run --epochs 200 --suite all.

Use a new --output for any changed code/data/configuration. Completed identical
runs are reused; interrupted runs restore optimizer, scheduler and RNG states.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
from pathlib import Path
import gc
import torch
from .data import BATTERIES, FEATURES, load_frame, windows, fold_data, partitions
from .models import ARCHITECTURES
from .runtime import provenance, atomic_csv
from .engine import fit
from ..training.config import ROOT, json_write


def matrix(suite, historical=False):
    rows = [(a, 'five', 5, .1 if a.endswith('_pinn') else 0.,
             .1 if a.endswith('_pinn') else 0., False, 'primary') for a in ARCHITECTURES]
    if suite == 'all':
        rows += [(a, 'five', 5, 0., 0., False, 'physics_ablation') for a in ('lstm_pinn', 'cnn_pinn')]
        rows += [('lstm', f, 5, 0., 0., False, 'feature_ablation') for f in ('three', 'nine')]
        rows += [('lstm', 'five', n, 0., 0., True, 'sequence_ablation') for n in (5, 10, 20)]
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT/'outputs/soh_four_model_benchmarks')
    p.add_argument('--raw', type=Path, default=ROOT.parents[1]/'data')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--suite', choices=['primary', 'all'], default='all')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--historical', action='store_true')
    p.add_argument('--report-only', action='store_true')
    args = p.parse_args()
    if args.epochs < 1:
        p.error('epochs must be positive')
    if args.report_only:
        from .report import generate
        generate(args.output)
        return
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seeds = [42] if args.smoke else [42, 123, 2026]
    folds = BATTERIES[:1] if args.smoke else BATTERIES + (['historical'] if args.historical else [])
    suite = 'primary' if args.smoke else args.suite
    epochs = 2 if args.smoke else args.epochs
    config = {'epochs': epochs, 'max_epochs_full_protocol': 200, 'smoke': args.smoke,
              'status': 'smoke' if args.smoke else ('full_budget' if epochs==200 else 'partial_reduced_budget'),
              'suite': suite, 'folds': folds, 'seeds': seeds, 'matrix': matrix(suite),
              'checkpoint_criterion': 'minimum inner-validation MSE; patience 20',
              'coefficient_space': [{'physics': .1, 'regularization': .1}],
              'coefficient_policy': 'fixed singleton prespecified; zero-zero ablation; no outer-fold tuning',
              'optimizer': {'name': 'AdamW', 'lr': .0005, 'weight_decay': .001, 'batch_size': 64,
                            'gradient_clip': 1., 'scheduler': 'ReduceLROnPlateau', 'factor': .5, 'patience': 5, 'min_lr': 1e-6},
              'device': device, 'aggregation': 'equal-weight battery macro metrics; seed sample SD separately',
              'recommendation': 'mean inner validation RMSE across folds and seeds; primary five-feature length-5 only'}
    manifest_hash = provenance(args.output, config, args.raw)
    frame = load_frame(args.output/'data_audit')
    json_write(args.output/'features/definitions.json', FEATURES)
    prepared = {}
    expected = []
    for a, feature_set, length, pw, rw, matched, experiment in matrix(suite):
        key = (length, matched)
        if key not in prepared:
            whole, audit = windows(frame, length, matched)
            atomic_csv(audit, args.output/'data_audit'/f'window_audit_L{length}_matched{matched}.csv')
            atomic_csv(whole.metadata, args.output/'splits'/f'windows_L{length}_matched{matched}.csv')
            prepared[key] = whole
        for fold in folds:
            json_write(args.output/'splits'/f'{fold}.json', partitions(fold))
            for seed in seeds:
                expected.append(dict(architecture=a, feature_set=feature_set, length=length, fold=fold,
                                     seed=seed, physics_weight=pw, reg_weight=rw, matched_targets=matched,
                                     experiment=experiment, epochs=epochs, device=device, status=config['status']))
    json_write(args.output/'planned_runs.json', expected)
    # Train according to immutable matrix. Outer results are never consumed by this loop.
    for run in expected:
        whole = prepared[(run['length'], run['matched_targets'])]
        data, scaler, info = fold_data(frame, whole, run['fold'], run['feature_set'])
        if args.smoke:
            for role, (x, y, meta, previous, valid) in data.items():
                n = min(len(y), 128)
                data[role] = (x[:n], y[:n], meta.iloc[:n].copy(), previous[:n], valid[:n])
        fit(args.output, run, data, scaler, info, manifest_hash)
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
    from .report import generate
    generate(args.output)


if __name__ == '__main__':
    main()
