import io
import torch
from experiments.rapct_prony.staged.stage3 import Stage3Model,activation
from experiments.rapct_prony.staged.runtime import seed_all


def model():
    seed_all(42);return Stage3Model({'purpose':'synthetic'}).eval()


def test_positive_rates_persistent_state_and_unknown_initialization():
    m=model();x=torch.randn(7,5,64)
    whole=m.forward_embeddings(x)
    first=m.forward_embeddings(x[:3]);second=m.forward_embeddings(x[3:],first['state'])
    torch.testing.assert_close(whole['degradation'],torch.cat([first['degradation'],second['degradation']]),rtol=1e-5,atol=1e-6)
    torch.testing.assert_close(whole['prediction'],torch.cat([first['prediction'],second['prediction']]),rtol=1e-5,atol=1e-6)
    assert (whole['rate']>=0).all() and (whole['degradation'].diff()>=0).all()
    assert whole['degradation'][0]>0
    reset=m.forward_embeddings(x[3:])
    torch.testing.assert_close(reset['degradation'][0],reset['state_proposal'][0])
    assert (whole['observation_residual'].abs()<=.030001).all()


def test_independent_losses_have_gradients_and_are_not_tautologies():
    m=model();out=m.forward_embeddings(torch.randn(6,5,64))
    assert out['proxy_consistency']>0
    loss=(out['prediction']-.8).square().mean()+.01*out['physics_loss']+.001*out['smoothness_loss']
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters() if p.requires_grad)


def test_no_future_context_leakage():
    m=model();x=torch.randn(7,5,64)
    a=m.forward_embeddings(x)['prediction'];x[4:]=999
    b=m.forward_embeddings(x)['prediction']
    torch.testing.assert_close(a[:4],b[:4],rtol=1e-5,atol=1e-6)


def test_warmup_and_checkpoint():
    assert [activation(e,{'warmup':[10,30]}) for e in [1,10,20,30,35]]==[0.,0.,.5,1.,1.]
    m=model();x=torch.randn(5,5,64);a=m.forward_embeddings(x)['prediction']
    stream=io.BytesIO();torch.save(m.state_dict(),stream);stream.seek(0)
    n=model();n.load_state_dict(torch.load(stream,weights_only=True))
    torch.testing.assert_close(a,n.forward_embeddings(x)['prediction'],rtol=0,atol=0)
