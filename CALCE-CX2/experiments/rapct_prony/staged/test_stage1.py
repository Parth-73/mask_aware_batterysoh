import io
import numpy as np
import pandas as pd
import pytest
import torch
from experiments.rapct_prony.staged.stage1 import Stage1Model
from experiments.rapct_prony.staged.prepare import windows,validate_cached_data
from experiments.rapct_prony.staged.dataset import partitions,inner_batteries,prepare
from experiments.rapct_prony.staged.runtime import seed_all


@pytest.mark.parametrize('pathway',['mlp','cnn','fusion'])
def test_stage1_dimensions_gradients_checkpoint_and_cutoff(pathway):
    seed_all(42);m=Stage1Model(pathway).eval()
    features=torch.randn(2,5,5);curves=torch.randn(2,5,3,176)
    y=m(features,curves)['prediction'];assert y.shape==(2,) and torch.isfinite(y).all()
    f,c=features.clone(),curves.clone();f[:,:-1]=999;c[:,:-1]=-999
    torch.testing.assert_close(y,m(f,c)['prediction'],rtol=0,atol=0)
    y.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    stream=io.BytesIO();torch.save(m.state_dict(),stream);stream.seek(0)
    copy=Stage1Model(pathway).eval();copy.load_state_dict(torch.load(stream,weights_only=True))
    torch.testing.assert_close(y,copy(features,curves)['prediction'],rtol=0,atol=0)


def test_raw_curve_kernel_axis():
    model=Stage1Model('cnn');seen=[]
    handle=model.network.cnn.branches[0][0].register_forward_pre_hook(lambda m,args:seen.append(tuple(args[0].shape)))
    model(torch.randn(2,5,5),torch.randn(2,5,3,176));handle.remove()
    assert seen==[(2,3,176)]


def test_partitions_and_train_only_scaling():
    assert inner_batteries('CX2_38')==['CX2_16','CX2_33']
    config={'outer':'CX2_38','validation':'CX2_16','length':5}
    data=prepare(config);split=partitions('CX2_38','CX2_16')
    assert len(set(sum(split.values(),[])))==7
    assert not set(split['train']) & set(split['outer'])
    assert all(c.split(':')[0] in split['train'] for c in data['preprocessing']['fit_cycle_ids'])
    assert data['indices'].shape==(8395,5)
    for role,ids in data['groups'].items():assert set(data['metadata'].iloc[ids].battery)<=set(split[role])


def test_cached_data_label_and_feature_consistency():
    f,c=validate_cached_data();assert c.shape==(11010,3,176)
    assert np.allclose(f.soh,pd.read_csv('outputs/rapct_prony/processed/samples.csv').soh)


def test_windows_never_stitch_missing_or_source_cycles():
    f=pd.DataFrame({'battery':['b']*9,'cycle_id':[f'b:{i}' for i in range(9)],'lifetime_cycle':np.arange(1,10),
        'sources':['s']*9,'start':pd.date_range('2020',periods=9,freq='h'),
        'end':pd.date_range('2020',periods=9,freq='h')+pd.Timedelta(minutes=59),'local_counter_resets':[0]*9})
    idx,_=windows(f,5);assert idx.shape==(5,5)
    f.loc[4:,'lifetime_cycle']+=1
    idx,_=windows(f,5);assert idx.shape==(1,5)
    f.loc[7:,'sources']='new'
    idx,_=windows(f,5);assert not len(idx)
