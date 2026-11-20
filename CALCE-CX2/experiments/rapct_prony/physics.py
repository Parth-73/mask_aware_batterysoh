"""Stage 3: constrained cycle-count aging with independent mathematical priors."""
import torch
from torch import nn
from torch.nn import functional as F


class DegradationBranch(nn.Module):
    def __init__(self,input_dim=64):
        super().__init__()
        self.initial=nn.Linear(input_dim,1)
        self.rate=nn.Linear(input_dim,1)
        self.correction=nn.Linear(input_dim,1)
        self.reference_raw=nn.Parameter(torch.tensor(0.))
        nn.init.constant_(self.initial.bias,-1.)
        nn.init.constant_(self.rate.bias,-1.)

    def forward(self,h,carried=None):
        """One whole chronological segment. Carry (last D,last rate) across known adjacent chunks."""
        with torch.autocast(device_type=h.device.type,enabled=False):
            h=h.float()
            rate=.001*F.softplus(self.rate(h).squeeze(-1))
            if carried is None:initial=.2*F.softplus(self.initial(h[:,0]).squeeze(-1))
            else:initial=carried[0].float()+carried[1].float()
            increments=torch.cat([torch.zeros_like(rate[:,:1]),rate[:,:-1]],dim=1)
            degradation=initial[:,None]+torch.cumsum(increments,dim=1)
            correction=.03*torch.tanh(self.correction(h).squeeze(-1))
            reference=1.+.2*torch.tanh(self.reference_raw)
            prediction=reference-degradation+correction
            return dict(prediction=prediction,degradation=degradation,rate=rate,correction=correction,reference=reference,
                carried=(degradation[:,-1],rate[:,-1]))


def loss_terms(output,target,mask,epoch,physics_weight=.01,smooth_weight=.001):
    supervised=(output['prediction'][mask]-target.float()[mask]).square().mean()
    correction=output['correction'][mask].square().mean()
    pairs=mask[:,1:]&mask[:,:-1]
    rate_change=(output['rate'][:,1:]-output['rate'][:,:-1])/.001
    smooth=rate_change[pairs].square().mean() if pairs.any() else rate_change.sum()*0.
    weight=max(0.,min(1.,(epoch-10)/20.))
    total=supervised+weight*(physics_weight*correction+smooth_weight*smooth)
    # No identically-zero monotonicity loss or recurrence-vs-itself residual.
    return total,dict(data=supervised,correction_prior=correction,rate_smoothness=smooth,warmup=weight)
