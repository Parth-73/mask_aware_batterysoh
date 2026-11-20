"""Stage 7 composition, synthetic-tested only; CALCE recovery training is gated off."""
import torch
from torch import nn
from torch.nn import functional as F
from .temporal import Stage2
from .physics import DegradationBranch
from .memory import NeuralProny


class RAPCTProny(nn.Module):
    def __init__(self,modes=4,tau_min_s=10.,tau_max_s=1000.,learnable=True):
        super().__init__()
        self.encoder=Stage2()
        self.physics=DegradationBranch()
        self.physics.correction.requires_grad_(False) # Full fusion uses S_ref-D-P, with no signed correction.
        self.memory=NeuralProny(modes,tau_min_s,tau_max_s,learnable)
        self.stress=nn.Linear(64,modes);self.initial_memory=nn.Linear(64,modes)

    def forward(self,features,curves,rest_s,verified_rest,unknown_gap,carried_aging=None,carried_memory=None):
        """One valid chronological segment. Reset both carries at an invalid segment boundary.

        unknown_gap initializes current memory from causal charging context; it never
        treats an unobserved interval as rest or as zero aging. All memory math is float32.
        """
        h=self.encoder.encode_sequence(features,curves)
        aging=self.physics(h,carried_aging)
        initials=self.memory.penalty_bound*torch.sigmoid(self.initial_memory(h).float())
        stress=.001*F.softplus(self.stress(h).float())
        state=initials[:,0] if carried_memory is None else carried_memory
        trajectories=[];penalties=[];recoveries=[];availability=[]
        for t in range(h.shape[1]):
            elapsed=torch.where(verified_rest[:,t],rest_s[:,t].float(),torch.zeros_like(rest_s[:,t],dtype=torch.float32))
            relaxed=state.float()*torch.exp(-elapsed[:,None]/self.memory.timescales)
            available=verified_rest[:,t]&~unknown_gap[:,t]
            recoveries.append(torch.where(available,self.memory.penalty(state)-self.memory.penalty(relaxed),torch.zeros_like(elapsed)))
            state=self.memory.step(state,stress[:,t],rest_s[:,t],verified_rest[:,t],unknown_gap[:,t],initials[:,t])
            trajectories.append(state);penalties.append(self.memory.penalty(state));availability.append(available)
        penalty=torch.stack(penalties,dim=1)
        return dict(prediction=aging['reference']-aging['degradation']-penalty,
            degradation=aging['degradation'],transient_penalty=penalty,
            modeled_recovery_contribution=torch.stack(recoveries,dim=1),recovery_available=torch.stack(availability,dim=1),
            timescales_s=self.memory.timescales,memory_states=torch.stack(trajectories,dim=1),
            carried_memory=state,carried_aging=aging['carried'],reference_factor=aging['reference'])
