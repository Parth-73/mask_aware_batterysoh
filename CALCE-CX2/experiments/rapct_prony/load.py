"""Load trained stage weights without requiring the original pretraining initialization file."""
from pathlib import Path
import json
import torch
from .models import Stage1
from .temporal import Stage2
from .physics import DegradationBranch


def load_model(checkpoint,device='cpu'):
    path=Path(checkpoint);saved=torch.load(path,map_location=device,weights_only=True)
    cfg=saved.get('config') or json.loads((path.parent/'config.json').read_text())
    stage=cfg.get('stage','cycle')
    if stage=='cycle':model=Stage1(cfg['pathway'],cfg['channels'],cfg['embedding'])
    elif stage=='transformer':model=Stage2(cfg['pathway'],cfg['channels'],cfg['embedding'],cfg['frozen'],
        d_model=cfg.get('transformer_dimension',64),heads=cfg.get('transformer_heads',4),layers=cfg.get('transformer_layers',2),
        ffn=cfg.get('transformer_ffn',128),dropout=cfg.get('transformer_dropout',.1))
    elif stage=='physics':model=DegradationBranch(saved['model']['initial.weight'].shape[1])
    else:raise ValueError(f'unsupported_trained_stage: {stage}')
    model.load_state_dict(saved['model']);return model.to(device).eval(),cfg
