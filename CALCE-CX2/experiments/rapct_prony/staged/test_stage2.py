import io
import torch
import pytest
from experiments.rapct_prony.staged.stage2 import Stage2Model
from experiments.rapct_prony.staged.runtime import seed_all


def model():
    seed_all(42)
    return Stage2Model({'purpose':'synthetic','channels':16,'freeze_cnn':False,'d_model':64,'layers':2}).eval()


def test_causal_temporal_mask_and_padding():
    m=model();x=torch.randn(2,5,64)
    a,_=m.encode_history(x);x[:,3:]=999;b,_=m.encode_history(x)
    torch.testing.assert_close(a[:,:3],b[:,:3],rtol=0,atol=0)
    padding=torch.tensor([[True,False,False,False,True],[False,False,True,False,False]])
    out=m.forward_embeddings(x,padding)
    assert torch.isfinite(out['prediction']).all()
    changed=x.clone();changed[padding]=-9999
    torch.testing.assert_close(out['prediction'],m.forward_embeddings(changed,padding)['prediction'],rtol=0,atol=0)
    weights=out['attention'];assert weights.shape==(2,2,4,5,5)
    assert torch.all(weights.masked_select(torch.triu(torch.ones(5,5,dtype=torch.bool),1)[None,None,None])==0)
    with pytest.raises(ValueError):m.forward_embeddings(x,torch.ones(2,5,dtype=torch.bool))


def test_raw_encoder_transformer_gradient_and_checkpoint():
    m=model();f=torch.randn(2,5,5);c=torch.randn(2,5,3,176)
    result=m(f,c);assert result['prediction'].shape==(2,)
    result['prediction'].square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    stream=io.BytesIO();torch.save(m.state_dict(),stream);stream.seek(0)
    other=model();other.load_state_dict(torch.load(stream,weights_only=True))
    torch.testing.assert_close(result['prediction'],other(f,c)['prediction'],rtol=0,atol=0)


def test_frozen_encoder_has_no_gradients():
    m=Stage2Model({'purpose':'synthetic','channels':16,'freeze_cnn':True}).train()
    out=m(torch.randn(2,5,5),torch.randn(2,5,3,176))
    out['prediction'].square().mean().backward()
    assert all(p.grad is None for p in m.cycle_encoder.parameters())
    assert not m.cycle_encoder.training
