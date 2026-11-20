"""Full architecture interface; CALCE recovery training is explicitly feasibility-gated."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .stage2 import Stage2Model
from .prony import NeuralPronyMemory


class RecoveryAwareHybrid(nn.Module):
    def __init__(self,modes=4,learnable=True,tau_bounds=(5.,1200.),experimental=False):
        super().__init__()
        if experimental:
            raise RuntimeError('Experimental recovery gate is closed: verify independent rest excitation and state identification first')
        self.encoder=Stage2Model({'purpose':'synthetic','channels':16,'freeze_cnn':False})
        self.encoder.head=nn.Identity()
        self.initial_loss=nn.Linear(64,1);self.rate_head=nn.Linear(64,1)
        nn.init.constant_(self.initial_loss.bias,-2.5);nn.init.constant_(self.rate_head.bias,-8.)
        self.log_reference=nn.Parameter(torch.tensor(math.log(1.05)))
        self.memory=NeuralPronyMemory(modes,tau_bounds=tau_bounds,learnable=learnable)

    def forward_embeddings(self,embeddings,elapsed_s,verified_rest,state=None,valid_intervals=None):
        history,attention=self.encoder.encode_history(embeddings);context=history[:,-1].float()
        proposal=F.softplus(self.initial_loss(context).squeeze(-1));rate=F.softplus(self.rate_head(context).squeeze(-1))
        valid=torch.ones(len(context),dtype=torch.bool,device=context.device) if valid_intervals is None else valid_intervals
        d=[];z=[];p=[];recovery=[];resets=[]
        previous=state
        for j,h in enumerate(context):
            if previous is None or not valid[j]:
                if verified_rest[j] and not valid[j]:raise ValueError('Unknown gap cannot be declared verified rest')
                loss=proposal[j];memory=self.memory.initial_state(h[None]);penalty=(memory*self.memory.amplitudes()).sum(-1)[0]
                recovered=loss*0;reset=True
            else:
                loss=previous['degradation']+previous['rate']
                event=self.memory.step(h[None],elapsed_s[j:j+1],verified_rest[j:j+1],previous['memory'],valid_interval=valid[j:j+1])
                memory=event['state'];penalty=event['penalty'][0];recovered=event['recovery_contribution'][0];reset=False
            d.append(loss);z.append(memory[0]);p.append(penalty);recovery.append(recovered);resets.append(reset)
            previous={'degradation':loss,'rate':rate[j],'memory':memory}
        degradation=torch.stack(d);penalty=torch.stack(p)
        return {'prediction':self.log_reference.exp()-degradation-penalty,'degradation':degradation,
                'transient_penalty':penalty,'recovery_contribution':torch.stack(recovery),
                'memory_states':torch.stack(z),'timescales_s':self.memory.timescales(),
                'amplitudes':self.memory.amplitudes(),'state':previous,'reset':torch.tensor(resets,device=context.device)}

    def forward(self,features,curves,elapsed_s,verified_rest,state=None,valid_intervals=None):
        t,l,_=features.shape
        embedded=self.encoder.encode_cycles(features.reshape(t*l,5),curves.reshape(t*l,3,176)).reshape(t,l,64)
        return self.forward_embeddings(embedded,elapsed_s,verified_rest,state,valid_intervals)
