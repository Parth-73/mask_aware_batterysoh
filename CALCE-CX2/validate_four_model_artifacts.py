"""Independent post-run integrity validation; run after the benchmark completes."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def validate(output):
    rows = pd.read_csv(output/'metrics/all_runs.csv')
    assert not rows.configuration_id.duplicated().any(), 'Duplicate completed run'
    expected = json.loads((output/'planned_runs.json').read_text())
    assert len(rows) == len(expected), 'Incomplete requested matrix'
    count, paired = 0, {}
    for r in rows.itertuples():
        with open(r.checkpoint_path, 'rb') as f:
            assert hashlib.file_digest(f, 'sha256').hexdigest() == r.checkpoint_sha256
        preprocessing = json.loads((output/'scalers'/f'{r.configuration_id}.json').read_text())
        split = preprocessing['partitions']
        assert len(set(sum(split.values(), []))) == 7
        assert len(sum(split.values(), [])) == 7
        assert set(preprocessing['fit_batteries']) == set(split['train'])
        assert all(c.split(':')[0] in split['train'] for c in preprocessing['fit_cycles'])
        for role in ['train', 'validation', 'test']:
            p = pd.read_csv(output/'predictions'/f'{r.configuration_id}_{role}.csv')
            assert set(p.battery) <= set(split[role])
            assert not p.cycle_id.duplicated().any()
            assert np.isfinite(p[['actual_soh', 'predicted_soh']]).all().all()
            assert np.allclose(p.signed_error, p.predicted_soh-p.actual_soh, atol=1e-7)
            assert np.allclose(p.squared_error, p.signed_error**2, atol=1e-7)
            assert not p.rest_verified.any()
            assert p.rest_duration_s.isna().all()
            for q in p.itertuples():
                ids = json.loads(q.input_cycle_ids)
                ordinals = json.loads(q.input_ordinals)
                assert len(ids) == r.length
                assert ids[-1] == q.cycle_id
                assert np.all(np.diff(ordinals)==1)
                assert ordinals[-1] == q.lifetime_cycle
                assert all(c.split(':')[0] == q.battery for c in ids)
            key = (r.fold, role, bool(r.matched_targets))
            ids = set(p.cycle_id)
            if key in paired:
                assert paired[key] == ids, 'Architectures/features/seeds do not have matched targets'
            else:
                paired[key] = ids
            count += len(p)
            if role == 'test':
                assert abs(np.sqrt(np.mean(p.signed_error**2))-r.rmse_fraction)<1e-7
                assert abs(np.mean(p.absolute_error)*100-r.mae_percentage_points)<1e-5
                error = p.predicted_soh.to_numpy()-p.actual_soh.to_numpy()
                actual = p.actual_soh.to_numpy()
                predicted = p.predicted_soh.to_numpy()
                derived = {'mae_fraction': np.mean(abs(error)), 'mse_fraction_squared': np.mean(error**2),
                    'mape_percent': 100*np.mean(abs(error)/np.maximum(abs(actual), 1e-8)),
                    'smape_percent': 200*np.mean(abs(error)/np.maximum(abs(actual)+abs(predicted), 1e-8)),
                    'r2': 1-np.sum(error**2)/np.sum((actual-actual.mean())**2),
                    'bias_fraction': error.mean(), 'max_absolute_error_fraction': max(abs(error)),
                    'median_absolute_error_fraction': np.median(abs(error)),
                    'p95_absolute_error_fraction': np.quantile(abs(error), .95)}
                assert len(p) == r.n_samples
                for metric, value in derived.items():
                    assert np.isclose(value, getattr(r, metric), atol=2e-6, rtol=1e-5), f'{metric} inconsistent: {r.configuration_id}'
        h = pd.read_csv(output/'training_logs'/f'{r.configuration_id}.csv')
        assert np.isfinite(h[['train_loss', 'data_loss', 'physics_loss', 'regularization_loss', 'validation_mse']]).all().all()
        assert r.best_epoch == h.loc[h.validation_mse.idxmin(), 'epoch']
    result = {'passed': True, 'completed_runs': len(rows), 'planned_runs': len(expected),
              'validated_prediction_rows': count, 'matched_cohort_groups': len(paired),
              'checks': ['checkpoint SHA256', 'training-only scaling and disjoint batteries',
                         'matched cohorts across architectures/features/seeds/sequence ablations',
                         'original consecutive window IDs and causal final target', 'finite saved losses',
                         'validation-only best epoch', 'all six primary and secondary metric formulas, units and prediction algebra',
                         'rest remains unverified and missing']}
    # Carry explicit dataset, fold, seed and configuration provenance into aggregate rows.
    cross = rows[rows.fold.ne('historical')]
    for path in (output/'metrics').glob('*.csv'):
        table = pd.read_csv(path)
        if 'dataset' not in table:
            table['dataset'] = 'CALCE-CX2 frozen validated exports'
        if 'configuration_id' not in table and len(table):
            source = cross
            if path.stem in ('final_comparison', 'validation_recommendation'):
                source = cross[cross.experiment.eq('primary')]
            if path.stem == 'feature_ablation':
                source = cross[cross.architecture.eq('lstm') & cross.experiment.isin(['primary', 'feature_ablation'])]
            for index, row in table.iterrows():
                match = source
                for key in ['architecture', 'feature_set', 'length', 'fold', 'experiment',
                            'physics_weight', 'reg_weight', 'matched_targets']:
                    if key in table:
                        match = match[match[key].eq(row[key])]
                assert len(match), f'Aggregate has no source experiments: {path.name}'
                table.loc[index, 'source_configuration_ids'] = json.dumps(sorted(match.configuration_id))
                table.loc[index, 'source_folds'] = json.dumps(sorted(match.fold.unique()))
                table.loc[index, 'source_seeds'] = json.dumps(sorted(int(s) for s in match.seed.unique()))
                table.loc[index, 'aggregation'] = 'seed mean within battery, equal battery macro; seed SD in seed_stability.csv'
        tmp = path.with_suffix('.tmp')
        table.to_csv(tmp, index=False)
        tmp.replace(path)
    # Describe measured ablations without changing or selecting any experiment.
    physics = cross[cross.architecture.str.endswith('_pinn')].groupby(['architecture', 'physics_weight']).rmse_percentage_points.mean().unstack()
    feature = cross[cross.architecture.eq('lstm') & cross.experiment.isin(['primary', 'feature_ablation'])].groupby('feature_set').rmse_percentage_points.mean()
    sequence = cross[cross.experiment.eq('sequence_ablation')].groupby('length').rmse_percentage_points.mean()
    text = ['\n\n## Measured ablation interpretation (reduced-budget descriptive results)',
            '\nOuter metrics below describe the frozen design and never feed tuning. Positive penalty-minus-zero deltas mean the penalty increased error. These small differences do not establish population significance or converged superiority.']
    deltas = {}
    for architecture, row in physics.iterrows():
        delta = float(row[.1]-row[0.])
        deltas[architecture] = delta
        text.append(f'\n- {architecture}: regularized RMSE {row[.1]:.6f} versus zero-weight {row[0.]:.6f} SOH percentage points; delta {delta:+.6f}.')
    text.append('\nVanilla LSTM feature ablation (common length-5 targets), RMSE in SOH percentage points:\n\n'+feature.to_string())
    text.append('\nVanilla LSTM matched length-20 target cohort, RMSE in SOH percentage points:\n\n'+sequence.to_string())
    text.append('\nError bars in aggregate comparison figures are descriptive sample SD over fold/seed runs, not confidence intervals. Use per-fold seed_stability.csv for seed-only variability and cross_validation_summary.csv for variability of battery seed means.')
    worst = cross[cross.experiment.eq('primary')].groupby(['architecture', 'fold']).rmse_percentage_points.mean().reset_index()
    worst = worst.loc[worst.groupby('architecture').rmse_percentage_points.idxmax()]
    text.append('\nWorst held-out battery by seed-mean RMSE for each primary architecture (descriptive failure cases):\n\n'+worst.to_string(index=False))
    text.append('\nThese batteries require scrutiny for protocol/lifetime distribution shift. Neither future feature choices nor additional tuning have been made from these outer outcomes. The ten-epoch budget leaves optimization and generalization effects partly confounded.')
    report = output/'reports/BASELINE_COMPARISON_REPORT.md'
    content = report.read_text(encoding='utf-8').split('\n\n## Measured ablation interpretation')[0]
    report.write_text(content+'\n'.join(text), encoding='utf-8')
    (output/'reports/ablation_interpretation.json').write_text(json.dumps({'penalty_minus_zero_rmse_percentage_points': deltas,
        'feature_ablation_rmse_percentage_points': feature.to_dict(),
        'matched_sequence_rmse_percentage_points': {str(k): v for k, v in sequence.items()}}, indent=2))
    supporting = [Path(__file__), Path(__file__).parent/'tests/test_four_model_benchmark.py',
                  Path(__file__).parent/'FOUR_MODEL_README.md', Path(__file__).parent/'FOUR_MODEL_IMPLEMENTATION_PLAN.md']
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in supporting}
    (output/'data_audit/supporting_code_hashes.json').write_text(json.dumps(hashes, indent=2))
    (output/'data_audit/artifact_validation.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    validate(parser.parse_args().output)
