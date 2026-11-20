"""Stage 4 single-exponential memory; experimental use remains gated by event support."""
import math
import torch
from torch import nn


class SingleRelaxation(nn.Module):
    def __init__(self,tau_min_s=10.,tau_max_s=1000.,learnable=True,penalty_bound=.1):
        super().__init__()
        if not 0<tau_min_s<tau_max_s:raise ValueError('positive_timescale_bounds_required')
        self.log_min=math.log(tau_min_s);self.log_max=math.log(tau_max_s)
        if penalty_bound<=0:raise ValueError('positive_penalty_bound_required')
        self.penalty_bound=penalty_bound
        self.tau_raw=nn.Parameter(torch.tensor(0.),requires_grad=learnable)

    @property
    def timescales(self):
        return torch.exp(self.log_min+(self.log_max-self.log_min)*torch.sigmoid(self.tau_raw.float())).reshape(1)

    def step(self,state,stress,rest_s,verified_rest,unknown=None,unknown_initial=None,active=None):
        state,stress=state.float(),stress.float()
        if not all(torch.isfinite(x).all() for x in [state,stress,rest_s]):raise ValueError('nonfinite_memory_input')
        if (stress<0).any() or (state<0).any() or (state>self.penalty_bound+1e-6).any() or (rest_s<0).any():raise ValueError('invalid_memory_input')
        elapsed=torch.where(verified_rest,rest_s.float(),torch.zeros_like(rest_s,dtype=torch.float32))
        relaxed=state*torch.exp(-elapsed[:,None]/self.timescales)
        # Smooth bounded stress accumulation; rest itself is exactly exponential.
        evolved=relaxed+(self.penalty_bound-relaxed)*(-torch.expm1(-stress/self.penalty_bound))
        if active is not None:evolved=torch.where(active[:,None],evolved,relaxed)
        if unknown is not None and unknown.any():
            if unknown_initial is None:raise ValueError('unknown_gap_requires_explicit_state_initialization')
            if not torch.isfinite(unknown_initial).all() or (unknown_initial<0).any() or (unknown_initial>self.penalty_bound).any():raise ValueError('invalid_unknown_state')
            evolved=torch.where(unknown[:,None],unknown_initial.float(),evolved)
        return evolved

    def penalty(self,state):return state[:,0]


class NeuralProny(SingleRelaxation):
    """Stage 5: positive normalized amplitudes and fixed or bounded learnable timescales.

    A battery adaptation of the reference NPSSL's exponential states, without its
    oscillator equation or its implicit per-window zero initialization.
    """
    def __init__(self,modes=4,tau_min_s=10.,tau_max_s=1000.,learnable=True,penalty_bound=.1):
        super().__init__(tau_min_s,tau_max_s,learnable,penalty_bound)
        if modes not in [1,2,4,8]:raise ValueError('unsupported_mode_count')
        self.modes=modes
        p=torch.linspace(.001,.999,modes) if modes>1 else torch.tensor([.5])
        self.tau_raw=nn.Parameter(torch.logit(p),requires_grad=learnable)
        self.amplitude_raw=nn.Parameter(torch.full((modes,),-1.))

    @property
    def timescales(self):
        return torch.exp(self.log_min+(self.log_max-self.log_min)*torch.sigmoid(self.tau_raw.float()))

    @property
    def weights(self):
        positive=torch.nn.functional.softplus(self.amplitude_raw.float())
        return positive/positive.sum().clamp_min(1e-12)

    def penalty(self,state):return (state.float()*self.weights).sum(dim=-1)

    def rollout(self,stress,rest_s,verified_rest,initial_state,unknown=None,unknown_initial=None):
        state=initial_state.float();trajectory=[]
        for t in range(stress.shape[1]):
            state=self.step(state,stress[:,t],rest_s[:,t],verified_rest[:,t],
                None if unknown is None else unknown[:,t],None if unknown_initial is None else unknown_initial[:,t])
            trajectory.append(state)
        return torch.stack(trajectory,dim=1),state
