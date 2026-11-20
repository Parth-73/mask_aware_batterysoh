import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from src.benchmark.models import BenchmarkModel, ARCHITECTURES, physics_terms
from src.benchmark.data import BATTERIES, FEATURES, UNITS, partitions, fold_data
from src.benchmark.engine import metrics
from src.benchmark.runtime import seed_all, atomic_torch
from src.training.dataset import WindowSet


class FourModelTests(unittest.TestCase):
    def test_vectorized_windows_match_audited_reference(self):
        from test_lstm_baselines import frame_for
        from src.training.dataset import build_windows
        from src.benchmark.data import audited_windows
        a = frame_for(n=90)
        a.loc[8, 'cycle_exclusion'] = 'missing_features'
        a.loc[22:, 'lifetime_cycle'] += 1
        a.loc[34:, 'sources'] = 'second_source'
        a.loc[44, 'local_counter_resets'] = 1
        a.loc[52, 'sources'] = 'a;b'
        a.loc[63:, 'start'] += pd.Timedelta(days=2)
        a.loc[63:, 'end'] += pd.Timedelta(days=2)
        a.loc[71, 'start'] -= pd.Timedelta(hours=2)
        frame = pd.concat([a, frame_for('CX2_33', 30)], ignore_index=True)
        for length in (5, 10, 20):
            ref, audit = build_windows(frame, length, FEATURES['nine'], np.zeros(9), np.ones(9))
            fast, fast_audit = audited_windows(frame, length)
            np.testing.assert_array_equal(ref.x, fast.x)
            np.testing.assert_array_equal(ref.y, fast.y)
            pd.testing.assert_frame_equal(ref.metadata, fast.metadata)
            pd.testing.assert_frame_equal(audit, fast_audit)

    def test_shapes_finite_gradients_and_reproducibility(self):
        torch.set_num_threads(2)
        for architecture in ARCHITECTURES:
            seed_all(42)
            m = BenchmarkModel(architecture)
            x = torch.randn(4, 5, 5)
            p, d, c = m(x)
            self.assertEqual(p.shape, (4,))
            loss = p.square().mean()
            if m.physics:
                a, b = physics_terms(d, c, d+.1, torch.ones(4, dtype=torch.bool))
                loss += a+b
                self.assertTrue((d>=0).all())
                self.assertTrue((c.abs()<=.030001).all())
            loss.backward()
            for parameter in m.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
            seed_all(42)
            n = BenchmarkModel(architecture)
            for a, b in zip(m.parameters(), n.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_shared_physics_definition(self):
        d = torch.tensor([.2, .1], requires_grad=True)
        c = torch.tensor([.01, -.01], requires_grad=True)
        p, r = physics_terms(d, c, torch.tensor([.1, .2]), torch.tensor([True, True]))
        self.assertAlmostEqual(p.item(), .005)
        self.assertAlmostEqual(r.item(), .0001)
        (p+r).backward()
        self.assertTrue(torch.isfinite(d.grad).all())
        p, _ = physics_terms(d, c, d, torch.tensor([False, False]))
        self.assertEqual(p.item(), 0)

    def test_partitions(self):
        for fold in BATTERIES+['historical']:
            s = partitions(fold)
            self.assertEqual(set(sum(s.values(), [])), set(BATTERIES))
            self.assertEqual(len(sum(s.values(), [])), 7)
            self.assertFalse(set(s['train']) & set(s['test']))
        self.assertEqual(partitions('CX2_38')['validation'], ['CX2_16'])
        self.assertEqual(partitions('historical')['validation'], ['CX2_36'])

    def test_features_and_units_no_leakage(self):
        self.assertEqual(FEATURES['five'], ['cc_duration_s', 'cv_duration_s', 'dva_peak', 'ic_peak_voltage', 'ic_peak_width'])
        for features in FEATURES.values():
            for feature in features:
                self.assertNotIn('soh', feature)
                self.assertNotIn('discharge', feature)
                self.assertNotIn(feature, ['battery', 'lifetime_cycle', 'split'])
                self.assertIn(feature, UNITS)

    def test_train_scaler_and_unverified_rest(self):
        rows = []
        for j, battery in enumerate(BATTERIES):
            rows.append(dict(battery=battery, cycle_exclusion='', cycle_id=battery,
                             **{f: float(j+1) for f in FEATURES['nine']}))
        frame = pd.DataFrame(rows)
        meta = pd.DataFrame({'battery': BATTERIES, 'cycle_id': BATTERIES,
                             'lifetime_cycle': [5]*7, 'source': BATTERIES})
        w = WindowSet(np.ones((7, 5, 9), np.float32), np.ones(7, np.float32), meta)
        a, scaler, info = fold_data(frame, w, 'CX2_38', 'five')
        frame.loc[frame.battery.eq('CX2_38'), FEATURES['five']] = 999.
        _, other, _ = fold_data(frame, w, 'CX2_38', 'five')
        np.testing.assert_array_equal(scaler.mean_, other.mean_)
        self.assertNotIn('CX2_38', info['fit_batteries'])
        self.assertFalse(a['train'][4].any())

    def test_metric_units(self):
        m = metrics([.8, 1.], [.81, 1.01])
        self.assertAlmostEqual(m['mae_fraction'], .01)
        self.assertAlmostEqual(m['rmse_percentage_points'], 1.)
        self.assertAlmostEqual(m['mse_fraction_squared'], .0001)
        self.assertAlmostEqual(m['mape_percent'], 1.125)
        self.assertTrue(np.isfinite(metrics([0.], [0.])['smape_percent']))

    def test_checkpoint_roundtrip(self):
        model = BenchmarkModel('lstm_pinn')
        x = torch.randn(2, 5, 5)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'checkpoint.pt'
            atomic_torch(path, model.state_dict())
            restored = BenchmarkModel('lstm_pinn')
            restored.load_state_dict(torch.load(path, weights_only=True))
            torch.testing.assert_close(model(x)[0], restored(x)[0])

    def test_interrupted_resume_and_outer_label_isolation(self):
        from unittest.mock import patch
        from sklearn.preprocessing import StandardScaler
        from src.benchmark.engine import fit
        from src.benchmark.runtime import atomic_torch as real_save
        rng = np.random.default_rng(9)
        x = rng.normal(size=(8, 5, 5)).astype(np.float32)
        y = np.linspace(.7, .9, 8).astype(np.float32)
        meta = pd.DataFrame({'battery': ['synthetic']*8, 'cycle_id': [f'x{i}' for i in range(8)],
                             'lifetime_cycle': range(5, 13), 'input_cycle_ids': ['[]']*8})
        previous = np.maximum(np.arange(8)-1, 0)
        valid = np.arange(8)>0
        data = {s: (x, y.copy(), meta, previous, valid) for s in ('train', 'validation', 'test')}
        scaler = StandardScaler().fit(x.reshape(-1, 5))
        info = {'feature_order': FEATURES['five']}
        run = dict(architecture='transformer', feature_set='five', length=5, fold='synthetic',
                   seed=42, physics_weight=0., reg_weight=0., epochs=3, device='cpu',
                   matched_targets=False, experiment='test', status='test')
        with tempfile.TemporaryDirectory() as folder:
            first, second = Path(folder)/'a', Path(folder)/'b'
            for output in (first, second):
                for sub in ('runs', 'checkpoints', 'scalers', 'training_logs', 'predictions'):
                    (output/sub).mkdir(parents=True)
            a = fit(first, run, data, scaler, info, 'test_manifest')
            def interrupt_after_save(path, value):
                real_save(path, value)
                if Path(path).name == 'resume.pt':
                    raise KeyboardInterrupt('simulated process interruption')
            with patch('src.benchmark.engine.atomic_torch', side_effect=interrupt_after_save):
                with self.assertRaises(KeyboardInterrupt):
                    fit(second, run, data, scaler, info, 'test_manifest')
            # An arbitrary outer-label change must not alter resumed training/checkpoint selection.
            data['test'] = (x, y+5, meta, previous, valid)
            b = fit(second, run, data, scaler, info, 'test_manifest')
            aa = torch.load(a['checkpoint_path'], weights_only=False)['state_dict']
            bb = torch.load(b['checkpoint_path'], weights_only=False)['state_dict']
            for k in aa:
                torch.testing.assert_close(aa[k], bb[k], rtol=0, atol=0)
            self.assertEqual(a['best_epoch'], b['best_epoch'])
            self.assertEqual(a['validation_rmse_fraction'], b['validation_rmse_fraction'])


if __name__ == '__main__':
    unittest.main()
