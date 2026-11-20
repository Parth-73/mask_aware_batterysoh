"""Persistent bounded Neural Prony adaptation; see pinned NPSSL_REFERENCE.txt.

Original NPSSL: fixed exponential rates, softplus amplitudes, Euler input and zero
window initialization. Here: explicit carry/reset, bounded log timescales, normalized
positive amplitudes and a stable exact zero-order-hold forcing update. No oscillator PDE.
"""
import math
import torch
from torch import nn


class NeuralPronyMemory(nn.Module):
    def __init__(self,modes=4,context_dim=64,tau_bounds=(5.,1200.),learnable=True,maximum_penalty=.05):
        super().__init__()
        if modes not in [1,2,4,8]:raise ValueError('Supported mode-count ablations are 1/2/4/8')
        low,high=map(float,tau_bounds)
        if not 0<low<high:raise ValueError('Timescales require positive declared bounds')
        self.modes=modes;self.maximum_penalty=float(maximum_penalty)
        self.register_buffer('log_bounds',torch.tensor([math.log(low),math.log(high)]))
        fractions=torch.linspace(.01,.99,modes) if modes>1 else torch.tensor([.5])
        initial=torch.logit(fractions)
        if learnable:self.tau_raw=nn.Parameter(initial)
        else:self.register_buffer('tau_raw',initial)
        self.amplitude_raw=nn.Parameter(torch.zeros(modes))
        self.stress=nn.Linear(context_dim,modes);self.initializer=nn.Linear(context_dim,modes)

    def timescales(self):
        return torch.exp(self.log_bounds[0]+(self.log_bounds[1]-self.log_bounds[0])*torch.sigmoid(self.tau_raw.float()))

    def amplitudes(self):return torch.softmax(self.amplitude_raw.float(),dim=0)

    def initial_state(self,context):
        return self.maximum_penalty*torch.sigmoid(self.initializer(context.float())).float()

    def step(self,context,elapsed_s,verified_rest,state=None,valid_interval=None,stress_input=None):
        with torch.autocast(device_type=context.device.type,enabled=False):
            context=context.float();batch=len(context)
            dt=torch.as_tensor(elapsed_s,device=context.device,dtype=torch.float32).reshape(batch,1)
            rest=torch.as_tensor(verified_rest,device=context.device,dtype=torch.bool).reshape(batch,1)
            valid=torch.ones_like(rest) if valid_interval is None else torch.as_tensor(valid_interval,device=context.device,dtype=torch.bool).reshape(batch,1)
            if (rest & ~valid).any():raise ValueError('Unverified intervals cannot activate a rest transition')
            if not torch.isfinite(dt[valid]).all() or (dt[valid]<0).any():raise ValueError('Invalid verified elapsed time')
            initial=self.initial_state(context)
            old=initial if state is None else state.float()
            if old.shape!=(batch,self.modes) or not torch.isfinite(old).all() or (old<0).any() or (old>self.maximum_penalty+1e-7).any():
                raise ValueError('Invalid carried memory state')
            forcing=(self.maximum_penalty*torch.sigmoid(self.stress(context))) if stress_input is None else torch.as_tensor(stress_input,device=context.device,dtype=torch.float32).reshape(batch,self.modes)
            if not torch.isfinite(forcing).all() or (forcing<0).any() or (forcing>self.maximum_penalty).any():raise ValueError('Invalid bounded forcing')
            forcing=torch.where(rest,torch.zeros_like(forcing),forcing)
            safe_dt=torch.where(valid,dt,torch.zeros_like(dt))
            exponent=-safe_dt/self.timescales()[None]
            updated=torch.exp(exponent)*old+(-torch.expm1(exponent))*forcing
            updated=torch.where(valid,updated,initial)
            weights=self.amplitudes()
            before=(old*weights).sum(-1);penalty=(updated*weights).sum(-1)
            recovery=torch.where(rest[:,0]&valid[:,0],before-penalty,torch.zeros_like(penalty))
            return {'state':updated,'penalty':penalty,'recovery_contribution':recovery,
                    'timescales_s':self.timescales(),'amplitudes':weights,'reset':~valid[:,0]}

    def forward(self,context,elapsed_s,verified_rest,initial_state=None,valid_intervals=None,stress_input=None):
        if context.ndim!=3:raise ValueError('Context must have [batch,time,features] dimensions')
        state=initial_state;states=[];penalties=[];recoveries=[]
        for j in range(context.shape[1]):
            output=self.step(context[:,j],elapsed_s[:,j],verified_rest[:,j],state,
                None if valid_intervals is None else valid_intervals[:,j],
                None if stress_input is None else stress_input[:,j])
            state=output['state'];states.append(state);penalties.append(output['penalty']);recoveries.append(output['recovery_contribution'])
        return {'states':torch.stack(states,1),'penalty':torch.stack(penalties,1),
                'recovery_contribution':torch.stack(recoveries,1),'final_state':state,
                'timescales_s':self.timescales(),'amplitudes':self.amplitudes()}
