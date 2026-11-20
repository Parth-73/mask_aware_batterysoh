"""Measured runtime provenance and atomic, resumable experiment identity."""
import hashlib
import importlib.metadata as md
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch
from ..training.config import ROOT, file_sha, json_write

PACKAGES = ['numpy', 'pandas', 'scipy', 'scikit-learn', 'torch', 'matplotlib',
            'seaborn', 'tqdm', 'openpyxl', 'joblib', 'pytest']


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def atomic_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.csv.tmp')
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def provenance(output, config, raw):
    import pandas as pd
    output = Path(output)
    for name in ['data_audit', 'features', 'splits', 'scalers', 'checkpoints', 'training_logs',
                 'metrics', 'predictions', 'figures', 'reports', 'runs']:
        (output/name).mkdir(parents=True, exist_ok=True)
    versions = {}
    for p in PACKAGES:
        try:
            versions[p] = md.version(p)
        except md.PackageNotFoundError:
            versions[p] = None
    env = {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
           'packages': versions, 'cublas_workspace': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
           'precision': 'float32; AMP disabled', 'threads': torch.get_num_threads()}
    hardware = {'cuda_available': torch.cuda.is_available(), 'torch_cuda_runtime': torch.version.cuda,
                'cudnn': torch.backends.cudnn.version(), 'cpu': platform.processor()}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        hardware.update(gpu=props.name, vram_bytes=props.total_memory,
                        compute_capability=list(torch.cuda.get_device_capability()),
                        bf16_supported=torch.cuda.is_bf16_supported())
    try:
        hardware['nvidia_smi'] = subprocess.check_output(['nvidia-smi'], text=True)
    except (OSError, subprocess.CalledProcessError):
        hardware['nvidia_smi'] = None
    code = {str(p.relative_to(ROOT)): file_sha(p) for p in sorted((ROOT/'src').rglob('*.py'))}
    data = {name: file_sha(ROOT/'outputs'/name) for name in
            ['cx2_features_all.csv', 'cycle_quality.csv', 'rejected_cycles.csv',
             'feature_definitions.csv', 'feature_sets.json', 'inspection_manifest.csv']}
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    frozen = {'config': config, 'code_hashes': code, 'data_hashes': data, 'versions': versions,
              'python': sys.version, 'git_commit': commit, 'hardware_device': hardware.get('gpu', 'CPU')}
    manifest = output/'experiment_manifest.json'
    if manifest.exists():
        if json.loads(manifest.read_text()) != frozen:
            raise RuntimeError('Configuration, source, data or runtime changed: preserve artifacts and choose a new --output.')
    else:
        json_write(manifest, frozen)
    json_write(output/'environment.json', env)
    json_write(output/'hardware_info.json', hardware)
    (output/'requirements_benchmark.txt').write_text('\n'.join(f'{p}=={v}' for p, v in versions.items() if v)+'\n')
    raw_manifest = output/'data_audit/local_raw_hashes.csv'
    if not raw_manifest.exists():
        rows = []
        for p in sorted(Path(raw).rglob('*')):
            if p.is_file() and p.suffix.lower() in {'.xlsx', '.xls', '.txt', '.csv'}:
                rows.append({'path': str(p.relative_to(raw)), 'bytes': p.stat().st_size, 'sha256': file_sha(p)})
        atomic_csv(pd.DataFrame(rows, columns=['path', 'bytes', 'sha256']), raw_manifest)
    local = pd.read_csv(raw_manifest)
    historical = pd.read_csv(ROOT/'outputs/inspection_manifest.csv')
    reference = set(historical.sha256.dropna())
    verified = set(local.sha256) & reference
    json_write(output/'reproducibility_manifest.json', {**frozen, 'seeds': [42, 123, 2026],
        'local_raw_files': len(local), 'local_raw_hash_manifest_sha256': file_sha(raw_manifest),
        'matching_unique_raw_hashes': len(verified), 'historical_unique_raw_hashes': len(reference),
        'preprocessing': 'Reused frozen repository exports; no new raw reconstruction claimed.',
        'raw_coverage_warning': 'Local raw set may differ; model dataset identity is the exported table hashes.',
        'assumptions': 'See FOUR_MODEL_IMPLEMENTATION_PLAN.md and reports/BASELINE_COMPARISON_REPORT.md',
        'determinism': 'Deterministic algorithms and fixed RNG states; cross-platform bitwise identity not guaranteed.'})
    return identity(frozen)
