import io
import torch
import pytest
from experiments.rapct_prony.staged.prony import NeuralPronyMemory
from experiments.rapct_prony.staged.hybrid import RecoveryAwareHybrid
from experiments.rapct_prony.staged.runtime import seed_all


@pytest.mark.parametrize('modes',[1,2,4,8])
@pytest.mark.parametrize('learnable',[False,True])
def test_positive_bounded_modes_exact_rest_and_persistence(modes,learnable):
    seed_all(42);m=NeuralPronyMemory(modes,learnable=learnable)
    initial=torch.full((2,modes),.03);context=torch.zeros(2,4,64)
    dt=torch.full((2,4),30.);rest=torch.ones(2,4,dtype=torch.bool)
    all_=m(context,dt,rest,initial_state=initial)
    first=m(context[:,:2],dt[:,:2],rest[:,:2],initial_state=initial)
    second=m(context[:,2:],dt[:,2:],rest[:,2:],initial_state=first['final_state'])
    torch.testing.assert_close(all_['final_state'],second['final_state'])
    expected=initial*torch.exp(-120./m.timescales())
    torch.testing.assert_close(all_['final_state'],expected,atol=1e-7,rtol=1e-5)
    assert (m.timescales()>0).all() and (m.timescales()<1200).all()
    assert (m.amplitudes()>0).all() and torch.isclose(m.amplitudes().sum(),torch.tensor(1.))
    assert (all_['recovery_contribution']>=-1e-7).all()
    assert (all_['penalty']>=0).all() and (all_['penalty']<=.05).all()


def test_active_memory_unknown_gap_and_stability_gradients():
    m=NeuralPronyMemory(4);context=torch.zeros(1,64);initial=torch.full((1,4),.03)
    active=m.step(context,[10.],[False],state=initial,stress_input=torch.zeros(1,4))
    assert active['state'].sum()>0 and active['recovery_contribution']==0
    unknown=m.step(context,[float('nan')],[False],state=initial,valid_interval=[False])
    torch.testing.assert_close(unknown['state'],m.initial_state(context))
    assert unknown['reset'].item()
    with pytest.raises(ValueError):m.step(context,[10.],[True],state=initial,valid_interval=[False])
    huge=m.step(context,[1e12],[True],state=initial)
    assert torch.isfinite(huge['state']).all()
    active['penalty'].sum().backward()
    assert torch.isfinite(m.tau_raw.grad).all() and torch.isfinite(m.amplitude_raw.grad).all()


def test_full_hybrid_shape_causality_and_checkpoint():
    seed_all(42);m=RecoveryAwareHybrid().eval()
    f=torch.randn(4,5,5);c=torch.randn(4,5,3,176);dt=torch.tensor([0.,90.,30.,125.]);rest=torch.tensor([False,True,False,True])
    a=m(f,c,dt,rest)
    assert a['prediction'].shape==(4,) and a['memory_states'].shape==(4,4)
    assert (a['degradation'].diff()>=0).all() and torch.isfinite(a['prediction']).all()
    f2=f.clone();f2[3:]=999
    torch.testing.assert_close(a['prediction'][:3],m(f2,c,dt,rest)['prediction'][:3],rtol=1e-5,atol=1e-6)
    first=m(f[:2],c[:2],dt[:2],rest[:2]);second=m(f[2:],c[2:],dt[2:],rest[2:],state=first['state'])
    torch.testing.assert_close(a['prediction'],torch.cat([first['prediction'],second['prediction']]),rtol=1e-5,atol=1e-6)
    stream=io.BytesIO();torch.save(m.state_dict(),stream);stream.seek(0)
    n=RecoveryAwareHybrid().eval();n.load_state_dict(torch.load(stream,weights_only=True))
    torch.testing.assert_close(a['prediction'],n(f,c,dt,rest)['prediction'],rtol=0,atol=0)
    a['prediction'].square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    with pytest.raises(RuntimeError):RecoveryAwareHybrid(experimental=True)
