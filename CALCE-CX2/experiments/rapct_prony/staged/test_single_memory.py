import math
import torch
import pytest
from experiments.rapct_prony.staged.single_memory import SingleRelaxation


def test_known_synthetic_exponential_rest_and_rebound():
    model=SingleRelaxation(tau_initial=80.,learnable=False)
    out=model.step(torch.zeros(2,64),[80.,160.],[True,True],state=torch.tensor([[.03],[.03]]))
    expected=torch.tensor([.03/math.e,.03/math.e**2])
    torch.testing.assert_close(out['penalty'],expected)
    torch.testing.assert_close(out['recovery_contribution'],.03-expected)
    assert (out['timescales_s']>0).all()


def test_state_persists_during_active_cycling_and_resets_unknown_gap():
    model=SingleRelaxation()
    context=torch.zeros(1,64);state=torch.tensor([[.03]])
    first=model.step(context,[30.],[False],state=state,stress_input=[0.])
    second=model.step(context,[30.],[False],state=first['state'],stress_input=[0.])
    whole=model.step(context,[60.],[False],state=state,stress_input=[0.])
    torch.testing.assert_close(second['state'],whole['state'])
    assert first['penalty']>0 and first['recovery_contribution']==0
    reset=model.step(context,[float('nan')],[False],state=state,valid_interval=[False])
    torch.testing.assert_close(reset['state'],model.initial_state(context))
    assert reset['reset'].item()
    with pytest.raises(ValueError):model.step(context,[-1.],[True],state=state)
    with pytest.raises(ValueError):model.step(context,[30.],[True],state=state,valid_interval=[False])


def test_state_float32_and_finite_timescale_gradients():
    model=SingleRelaxation()
    with torch.autocast('cpu',dtype=torch.bfloat16):
        output=model.step(torch.zeros(1,64),[1e9],[True],state=torch.tensor([[.03]]))
    assert output['state'].dtype==torch.float32 and torch.isfinite(output['state']).all()
    out=model.step(torch.zeros(1,64),[20.],[True],state=torch.tensor([[.03]]))
    out['penalty'].sum().backward()
    assert model.tau_raw.grad is not None and torch.isfinite(model.tau_raw.grad)
