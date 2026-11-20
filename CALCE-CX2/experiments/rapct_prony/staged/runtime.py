"""Deterministic FP32 default, atomic writes and stage-specific immutable identities."""
import hashlib
import importlib
import json
import random
from pathlib import Path
import numpy as np
import torch
from src.common import digest
from src.training.config import json_write
from .prepare import OUT, SOURCE


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False


def identity(config):
    return hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()


def atomic_torch(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');torch.save(value,tmp);tmp.replace(path)


def atomic_csv(path, frame):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');frame.to_csv(tmp,index=False);tmp.replace(path)


def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if torch.cuda.is_available():torch.cuda.set_rng_state_all(state['cuda'])


def synchronize(device):
    if device.type=='cuda':torch.cuda.synchronize(device)


def definition(config):
    paths=[Path(__file__),Path(__file__).with_name('prepare.py'),Path(__file__).with_name('dataset.py'),Path(__file__).with_name('engine.py'),
           Path(__file__).with_name(config['module']+'.py'),SOURCE/'models.py',SOURCE/'data.py']
    if config['module']=='stage3':
        paths.extend([Path(__file__).with_name('stage2.py'),Path(__file__).with_name('physics_engine.py')])
    return {'parameters':config,'source_hashes':{str(p.relative_to(SOURCE)):digest(p) for p in paths},
            'data_manifest':json.loads((OUT/'processed/manifest.json').read_text()),
            'environment':json.loads((OUT/'configs/environment.json').read_text()),
            'window_indices_sha256':digest(OUT/'processed'/f"indices_L{config.get('length',5)}.npy"),
            'parent_sha256':digest(config['parent']) if config.get('parent') else None}


def build(config):
    module=importlib.import_module('.'+config['module'],package=__package__)
    return module.build(config)
