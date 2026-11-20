"""Single-file independently loadable stage checkpoints, including parent specifications."""
import tempfile
from pathlib import Path
import torch
from .runtime import atomic_torch,build


def package(checkpoint):
    checkpoint=Path(checkpoint)
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    parent=payload['config'].get('parent')
    return {'checkpoint':payload,'parent':package(parent) if parent else None}


def save_bundle(checkpoint,destination):
    atomic_torch(destination,{'format':'rapct-independent-v1','payload':package(checkpoint)})


def load_bundle(path,device='cpu'):
    bundle=torch.load(path,map_location='cpu',weights_only=False)
    if bundle['format']!='rapct-independent-v1':raise ValueError('Unsupported checkpoint bundle')
    original=bundle['payload']['checkpoint']['config'].copy()
    with tempfile.TemporaryDirectory(prefix='rapct_checkpoint_') as temporary:
        def materialize(node,depth=0):
            payload=dict(node['checkpoint']);config=dict(payload['config'])
            if node['parent'] is not None:config['parent']=str(materialize(node['parent'],depth+1))
            payload['config']=config
            target=Path(temporary)/f'parent_{depth}.pt';torch.save(payload,target)
            return target
        checkpoint=materialize(bundle['payload'])
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        model=build(payload['config']);model.load_state_dict(payload['state_dict']);model.to(device)
    return model,original
