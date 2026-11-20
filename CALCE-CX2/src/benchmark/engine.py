"""Shared training budget, validation checkpointing, exact-state resume, frozen evaluation."""
import json
import random
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from .models import BenchmarkModel, physics_terms
from .runtime import seed_all, atomic_torch, atomic_csv, identity
from ..training.config import json_write, file_sha


def metrics(y, p):
    from sklearn.metrics import r2_score
    y, p = np.asarray(y, float), np.asarray(p, float)
    if not len(y) or y.shape != p.shape or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('Invalid metric inputs')
    e = p-y
    mae, mse = np.mean(abs(e)), np.mean(e**2)
    return {'n_samples': len(y), 'mae_fraction': mae, 'mse_fraction_squared': mse,
            'rmse_fraction': np.sqrt(mse), 'mae_percentage_points': mae*100,
            'rmse_percentage_points': np.sqrt(mse)*100,
            'mape_percent': 100*np.mean(abs(e)/np.maximum(abs(y), 1e-8)),
            'smape_percent': 200*np.mean(abs(e)/np.maximum(abs(y)+abs(p), 1e-8)),
            'r2': r2_score(y, p) if len(y)>1 else np.nan,
            'max_absolute_error_fraction': max(abs(e)), 'bias_fraction': np.mean(e),
            'median_absolute_error_fraction': np.median(abs(e)), 'p95_absolute_error_fraction': np.quantile(abs(e), .95)}


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def infer(model, x):
    model.eval()
    with torch.inference_mode():
        return torch.cat([model(b)[0] for b in x.split(1024)]).float()


def fit(output, run, data, scaler, preprocessing, manifest_hash):
    output = Path(output)
    description = {**run, 'manifest_hash': manifest_hash, 'preprocessing': preprocessing}
    ident = (f"{run['architecture']}_{run['feature_set']}_L{run['length']}_{run['fold']}_"
             f"seed{run['seed']}_p{run['physics_weight']}_r{run['reg_weight']}_" + identity(description)[:12])
    directory = output/'runs'/ident
    directory.mkdir(parents=True, exist_ok=True)
    completed = directory/'result.json'
    if completed.exists():
        result = json.loads(completed.read_text())
        checkpoint = output/'checkpoints'/f'{ident}.pt'
        if file_sha(checkpoint) != result['checkpoint_sha256']:
            raise RuntimeError('Completed checkpoint checksum changed')
        return result
    json_write(directory/'configuration.json', description)
    seed_all(run['seed'])
    device = torch.device(run['device'])
    model = BenchmarkModel(run['architecture'], len(preprocessing['feature_order'])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0005, weight_decay=.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=5, min_lr=1e-6)
    tensors = {role: (torch.tensor(d[0], device=device), torch.tensor(d[1], device=device),
                       torch.tensor(d[3], device=device), torch.tensor(d[4], device=device)) for role, d in data.items() if role != 'test'}
    x, y, previous, pairs = tensors['train']
    vx, vy, _, _ = tensors['validation']
    best, best_epoch, stale, start_epoch, duration = float('inf'), 0, 0, 1, 0.
    best_state, history = None, []
    resume = directory/'resume.pt'
    if resume.exists():
        saved = torch.load(resume, map_location='cpu', weights_only=False)
        if saved['description'] != description:
            raise RuntimeError('Resume identity mismatch')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        best, best_epoch, stale = saved['best'], saved['best_epoch'], saved['stale']
        best_state, history = saved['best_state'], saved['history']
        start_epoch, duration = saved['epoch']+1, saved['duration']
        random.setstate(saved['python_rng'])
        np.random.set_state(saved['numpy_rng'])
        torch.set_rng_state(saved['torch_rng'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    sync(device)
    start = time.perf_counter()
    for epoch in range(start_epoch, run['epochs']+1):
        if stale >= 20:
            break
        model.train()
        totals = np.zeros(4)
        permutation = torch.randperm(len(y), device=device)
        for idx in permutation.split(64):
            optimizer.zero_grad(set_to_none=True)
            pred, d, c = model(x[idx])
            data_loss = (pred.float()-y[idx]).square().mean()
            phys = reg = data_loss*0
            if model.physics:
                # Same model, strictly earlier observed charge window. Gradients flow to both states.
                pd_ = model(x[previous[idx]])[1]
                phys, reg = physics_terms(d, c, pd_, pairs[idx])
            loss = data_loss + run['physics_weight']*phys + run['reg_weight']*reg
            if not torch.isfinite(loss):
                raise FloatingPointError(ident)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            totals += np.array([loss.item(), data_loss.item(), phys.item(), reg.item()])*len(idx)
        vp = infer(model, vx)
        vmse = (vp-vy).square().mean().item()
        scheduler.step(vmse)
        if vmse < best-1e-10:
            best, best_epoch, stale = vmse, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        totals /= len(y)
        history.append({'epoch': epoch, 'train_loss': totals[0], 'data_loss': totals[1],
                        'physics_loss': totals[2], 'regularization_loss': totals[3], 'validation_mse': vmse,
                        'learning_rate': optimizer.param_groups[0]['lr'], 'elapsed_seconds': duration+time.perf_counter()-start})
        sync(device)
        state = {'description': description, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                 'scheduler': scheduler.state_dict(), 'best': best, 'best_epoch': best_epoch, 'stale': stale,
                 'best_state': best_state, 'history': history, 'epoch': epoch,
                 'duration': duration+time.perf_counter()-start,
                 'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
                 'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all() if device.type=='cuda' else []}
        atomic_torch(resume, state)
    sync(device)
    duration += time.perf_counter()-start
    model.load_state_dict(best_state)
    checkpoint = output/'checkpoints'/f'{ident}.pt'
    atomic_torch(checkpoint, {'state_dict': best_state, 'description': description, 'best_epoch': best_epoch})
    tmp = output/'scalers'/f'{ident}.tmp'
    joblib.dump(scaler, tmp)
    tmp.replace(output/'scalers'/f'{ident}.joblib')
    json_write(output/'scalers'/f'{ident}.json', preprocessing)
    atomic_csv(pd.DataFrame(history), output/'training_logs'/f'{ident}.csv')
    base = {**run, 'configuration_id': ident, 'dataset': 'CALCE-CX2 frozen validated exports',
            'parameter_count': sum(p.numel() for p in model.parameters()), 'best_epoch': best_epoch,
            'epochs_trained': len(history), 'training_seconds': duration, 'train_samples': len(y),
            'validation_samples': len(vy), 'validation_rmse_fraction': np.sqrt(best),
            'peak_gpu_memory_mb': torch.cuda.max_memory_allocated(device)/1024**2 if device.type=='cuda' else 0.,
            'checkpoint_path': str(checkpoint), 'checkpoint_sha256': file_sha(checkpoint)}
    # Commit the validation-selected checkpoint and selection evidence BEFORE outer inference.
    json_write(directory/'frozen_selection.json', base)
    region_rows = []
    for role, (a, b, meta, _, _) in data.items():
        xx = torch.tensor(a, device=device)
        # Synthetic warmup avoids making an additional held-out prediction pass.
        model.eval()
        with torch.inference_mode():
            model(torch.zeros((2, run['length'], a.shape[-1]), device=device))
        sync(device)
        begin = time.perf_counter()
        pp = infer(model, xx)
        sync(device)
        elapsed = time.perf_counter()-begin
        p = pp.cpu().numpy()
        table = meta.copy()
        for k, v in {**run, 'configuration_id': ident, 'evaluation_split': role}.items():
            table[k] = v
        table['actual_soh'] = b
        table['predicted_soh'] = p
        table['signed_error'] = p-b
        table['absolute_error'] = abs(p-b)
        table['squared_error'] = (p-b)**2
        table['rest_duration_s'] = np.nan
        table['rest_verified'] = False
        atomic_csv(table, output/'predictions'/f'{ident}_{role}.csv')
        if role == 'test':
            base.update(metrics(b, p), inference_seconds=elapsed, inference_ms_per_sample=1000*elapsed/len(b),
                        held_out_battery=meta.battery.iloc[0], evaluation_split='outer_test' if run['fold']!='historical' else 'historical_test')
            for region, mask in [('above_90', b>.9), ('80_to_90', (b>=.8)&(b<=.9)), ('below_80', b<.8)]:
                region_rows.append({**run, 'configuration_id': ident, 'held_out_battery': meta.battery.iloc[0],
                                    'region': region, **(metrics(b[mask], p[mask]) if mask.any() else {'n_samples': 0})})
    atomic_csv(pd.DataFrame(region_rows), directory/'regions.csv')
    json_write(completed, base)
    print(f"FINISHED {ident}: outer RMSE={base['rmse_fraction']:.5f}; val={np.sqrt(best):.5f}; {duration:.1f}s", flush=True)
    return base
