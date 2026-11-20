"""Physics-guided latent recurrence; no measured-SOH monotonicity or invented PDE."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .stage2 import Stage2Model


class Stage3Model(nn.Module):
    frozen_encoder=True
    temporal=True

    def __init__(self,config):
        super().__init__()
        if config.get('parent'):
            checkpoint=torch.load(config['parent'],map_location='cpu',weights_only=False)
            self.encoder=Stage2Model(checkpoint['config'])
            self.encoder.load_state_dict(checkpoint['state_dict'])
        elif config.get('purpose')=='synthetic':
            self.encoder=Stage2Model({'purpose':'synthetic','channels':16,'freeze_cnn':True})
        else:raise ValueError('Stage 3 requires a completed Stage-2 parent')
        self.encoder.head=nn.Identity()
        self.encoder.frozen_encoder=True;self.encoder.cycle_encoder.requires_grad_(False)
        width=self.encoder.width
        self.state_proposal=nn.Linear(width,1)
        self.rate_head=nn.Sequential(nn.Linear(width,32),nn.GELU(),nn.Linear(32,1))
        self.observation_head=nn.Linear(width,1)
        self.log_reference=nn.Parameter(torch.tensor(math.log(1.05)))
        nn.init.constant_(self.state_proposal.bias,-2.5)
        nn.init.constant_(self.rate_head[-1].bias,-8.)

    def encode_cycles(self,features,curves):return self.encoder.encode_cycles(features,curves)

    def forward_embeddings(self,embeddings,state=None):
        # Outer dimension is an ORDERED contiguous target segment/chunk, not shuffled samples.
        history,_=self.encoder.encode_history(embeddings)
        h=history[:,-1].float()
        proposal=F.softplus(self.state_proposal(h).squeeze(-1)).float()
        rate=F.softplus(self.rate_head(h).squeeze(-1)).float()
        first=proposal[0] if state is None else state['degradation'].float()+state['rate'].float()
        prior_increments=torch.cat([rate.new_zeros(1),rate[:-1]])
        degradation=first+torch.cumsum(prior_increments,0)
        correction=.03*torch.tanh(self.observation_head(h).squeeze(-1).float())
        reference=self.log_reference.float().exp()
        prediction=reference-degradation+correction
        # Independent encoder-state proposal versus accumulated state. The initialization
        # equality at the first element is excluded, so there is no tautological residual.
        offset=1 if state is None else 0
        consistency=(proposal[offset:]-degradation[offset:]).square().mean() if len(proposal)>offset else proposal.sum()*0
        previous=torch.cat([rate[:1] if state is None else state['rate'].float().reshape(1),rate[:-1]])
        variation=(rate[offset:]-previous[offset:]).square().mean() if len(rate)>offset else rate.sum()*0
        observation_penalty=correction.square().mean()
        return {'prediction':prediction,'degradation':degradation,'rate':rate,'state_proposal':proposal,
                'observation_residual':correction,'embedding':h,
                'reference':reference.expand_as(prediction),
                'physics_loss':consistency+.1*observation_penalty,'proxy_consistency':consistency,
                'smoothness_loss':variation,'observation_penalty':observation_penalty,
                'state':{'degradation':degradation[-1],'rate':rate[-1]}}


def build(config):return Stage3Model(config)


def activation(epoch,config):
    start,end=config.get('warmup',[10,30])
    return min(1.,max(0.,(epoch-start)/max(1,end-start)))
