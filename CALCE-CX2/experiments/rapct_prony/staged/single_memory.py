"""One-mode effective transient penalty. Synthetic validation, not a battery law."""
import math
import torch
from torch import nn


class SingleRelaxation(nn.Module):
    def __init__(self,context_dim=64,tau_bounds=(5.,1000.),tau_initial=50.,learnable=True,maximum_penalty=.05):
        super().__init__()
        low,high=map(float,tau_bounds)
        if not 0<low<tau_initial<high:raise ValueError('Positive initial tau must lie inside declared bounds')
        self.register_buffer('log_bounds',torch.tensor([math.log(low),math.log(high)]))
        fraction=(math.log(tau_initial)-math.log(low))/(math.log(high)-math.log(low))
        raw=torch.tensor(math.log(fraction/(1-fraction)))
        if learnable:self.tau_raw=nn.Parameter(raw)
        else:self.register_buffer('tau_raw',raw)
        self.maximum_penalty=float(maximum_penalty)
        self.stress=nn.Linear(context_dim,1)
        self.initializer=nn.Linear(context_dim,1)

    def timescales(self):
        return torch.exp(self.log_bounds[0]+(self.log_bounds[1]-self.log_bounds[0])*torch.sigmoid(self.tau_raw.float()))

    def initial_state(self,context):
        return self.maximum_penalty*torch.sigmoid(self.initializer(context.float())).float()

    def step(self,context,elapsed_s,verified_rest,state=None,valid_interval=None,stress_input=None):
        with torch.autocast(device_type=context.device.type,enabled=False):
            context=context.float();batch=len(context)
            dt=torch.as_tensor(elapsed_s,device=context.device,dtype=torch.float32).reshape(batch,1)
            rest=torch.as_tensor(verified_rest,device=context.device,dtype=torch.bool).reshape(batch,1)
            valid=torch.ones_like(rest) if valid_interval is None else torch.as_tensor(valid_interval,device=context.device,dtype=torch.bool).reshape(batch,1)
            if (rest & ~valid).any():raise ValueError('A verified rest requires a verified time interval')
            if not torch.isfinite(dt[valid]).all() or (dt[valid]<0).any():raise ValueError('Invalid observed elapsed time')
            initial=self.initial_state(context)
            old=initial if state is None else state.float()
            if old.shape!=(batch,1) or not torch.isfinite(old).all() or (old<0).any() or (old>self.maximum_penalty+1e-7).any():
                raise ValueError('Invalid carried penalty state')
            forcing=(self.maximum_penalty*torch.sigmoid(self.stress(context))) if stress_input is None else torch.as_tensor(stress_input,device=context.device,dtype=torch.float32).reshape(batch,1)
            if not torch.isfinite(forcing).all() or (forcing<0).any() or (forcing>self.maximum_penalty).any():raise ValueError('Invalid bounded stress input')
            forcing=torch.where(rest,torch.zeros_like(forcing),forcing)
            safe_dt=torch.where(valid,dt,torch.zeros_like(dt))
            decay=torch.exp(-safe_dt/self.timescales())
            updated=decay*old+(-torch.expm1(-safe_dt/self.timescales()))*forcing
            updated=torch.where(valid,updated,initial)
            recovery=torch.where(rest & valid,old-updated,torch.zeros_like(old))
            return {'state':updated,'penalty':updated[:,0],'recovery_contribution':recovery[:,0],
                    'reset':~valid[:,0],'estimated_initial_state':state is None,'timescales_s':self.timescales().reshape(1)}
