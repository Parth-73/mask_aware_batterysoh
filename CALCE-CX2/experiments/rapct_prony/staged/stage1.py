"""Stage 1 independently loadable models reuse the preserved, tested curve CNN."""
from torch import nn
from ..models import Stage1


class Stage1Model(nn.Module):
    temporal=False

    def __init__(self, pathway='fusion',channels=32,embedding=64):
        super().__init__()
        self.network=Stage1(pathway,channels,embedding)

    def forward(self, features, curves):
        # The comparison's target population is temporal, but Stage 1 uses current charge only.
        f,c=features[:,-1],curves[:,-1]
        z=self.network.encode(f,c)
        return {'prediction':self.network.head(z).squeeze(-1),'embedding':z}


def build(config):
    return Stage1Model(config['pathway'],config.get('channels',32),config.get('embedding',64))
