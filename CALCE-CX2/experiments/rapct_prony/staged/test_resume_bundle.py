import numpy as np
import pandas as pd
import pytest
import torch
from experiments.rapct_prony.staged.bundles import save_bundle,load_bundle
from experiments.rapct_prony.staged.stage1 import Stage1Model
from experiments.rapct_prony.staged.stage2 import Stage2Model
from experiments.rapct_prony.staged.runtime import atomic_torch,seed_all


def test_independent_bundle_needs_no_original_parent_file(tmp_path):
    seed_all(42)
    a=Stage1Model('fusion',16,64)
    parent=tmp_path/'parent.pt';atomic_torch(parent,{'state_dict':a.state_dict(),'config':{'module':'stage1','pathway':'fusion','channels':16,'embedding':64}})
    cfg={'module':'stage2','purpose':'synthetic','parent':str(parent),'freeze_cnn':True}
    b=Stage2Model(cfg).eval();checkpoint=tmp_path/'child.pt'
    atomic_torch(checkpoint,{'state_dict':b.state_dict(),'config':cfg})
    bundle=tmp_path/'independent.pt';save_bundle(checkpoint,bundle)
    # Rename only this test's known temporary files, preserving their bytes.
    parent.rename(tmp_path/'original_parent_preserved.pt');checkpoint.rename(tmp_path/'original_child_preserved.pt')
    restored,_=load_bundle(bundle);restored.eval()
    f=torch.randn(2,5,5);c=torch.randn(2,5,3,176)
    torch.testing.assert_close(b(f,c)['prediction'],restored(f,c)['prediction'],rtol=0,atol=0)


def test_stateful_training_resume_exact_with_dropout(tmp_path,monkeypatch):
    import experiments.rapct_prony.staged.physics_engine as engine
    from experiments.rapct_prony.staged.stage3 import Stage3Model
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    rng=np.random.default_rng(4)
    f=rng.normal(size=(12,5)).astype(np.float32);c=rng.normal(size=(12,3,176)).astype(np.float32)
    indices=np.stack([np.arange(j,j+5) for j in range(8)])
    meta=pd.DataFrame({'battery':['training']*4+['validation']*4,'cycle_id':[f'synthetic:{j}' for j in range(8)],
        'lifetime_cycle':list(range(1,5))*2,'segment_id':[1]*4+[2]*4,'prediction_cutoff':['2020-01-01']*8,
        'soh':np.linspace(.7,.9,8),'input_cycle_ids':['[]']*8})
    data={'features':f,'curves':c,'indices':indices,'metadata':meta,
        'groups':{'train':np.arange(4),'validation':np.arange(4,8)},'scalers':{},'preprocessing':{}}
    monkeypatch.setattr(engine,'prepare',lambda cfg:data)
    monkeypatch.setattr(engine,'definition',lambda cfg:{'synthetic_resume_test':True})
    monkeypatch.setattr(engine,'build',lambda cfg:Stage3Model({'purpose':'synthetic'}))
    cfg={'module':'stage3','stage':'A4','outer':'synthetic','validation':'validation','seed':42,
         'purpose':'synthetic','epochs':3,'batch_size':2,'physics_weight':.01,'smooth_weight':.001,'warmup':[0,1]}
    first=tmp_path/'uninterrupted';monkeypatch.setattr(engine,'OUT',first)
    a,_=engine.fit(cfg)
    second=tmp_path/'resumed';monkeypatch.setattr(engine,'OUT',second)
    save=engine.atomic_torch
    def interrupt(path,value):
        save(path,value)
        if path.name=='resume.pt':raise KeyboardInterrupt('simulated interruption')
    monkeypatch.setattr(engine,'atomic_torch',interrupt)
    with pytest.raises(KeyboardInterrupt):engine.fit(cfg)
    monkeypatch.setattr(engine,'atomic_torch',save)
    b,_=engine.fit(cfg)
    aa=torch.load(a/'best.pt',weights_only=False)['state_dict'];bb=torch.load(b/'best.pt',weights_only=False)['state_dict']
    for key in aa:torch.testing.assert_close(aa[key],bb[key],rtol=0,atol=0)
