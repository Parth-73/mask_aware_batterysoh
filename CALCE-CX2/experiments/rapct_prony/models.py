"""Stage 1: separate physical voltage-axis CNN and engineered-feature MLP."""
import torch
from torch import nn


class CurveEncoder(nn.Module):
    def __init__(self, channels=32, embedding=64):
        super().__init__()
        self.branches=nn.ModuleList([nn.Sequential(nn.Conv1d(3,channels,k,padding=k//2),
            nn.GroupNorm(4,channels),nn.GELU(),nn.Conv1d(channels,channels,k,padding=k//2),nn.GELU()) for k in (3,5,7)])
        self.pool=nn.AdaptiveAvgPool1d(1)
        self.projection=nn.Linear(3*channels,embedding)

    def forward(self,x):
        return self.projection(self.pool(torch.cat([b(x) for b in self.branches],dim=1)).squeeze(-1))


class Stage1(nn.Module):
    def __init__(self,pathway='fusion',channels=32,embedding=64):
        super().__init__()
        if pathway not in ('mlp','cnn','fusion'):raise ValueError(pathway)
        self.pathway=pathway
        self.cnn=CurveEncoder(channels,embedding) if pathway!='mlp' else None
        self.mlp=nn.Sequential(nn.Linear(5,32),nn.GELU(),nn.Linear(32,32),nn.GELU()) if pathway!='cnn' else None
        width={'mlp':32,'cnn':embedding,'fusion':embedding+32}[pathway]
        self.fusion=nn.Linear(width,64)
        self.head=nn.Sequential(nn.GELU(),nn.Linear(64,32),nn.GELU(),nn.Linear(32,1))

    def encode(self,features,curves):
        z=[]
        if self.cnn is not None:z.append(self.cnn(curves))
        if self.mlp is not None:z.append(self.mlp(features))
        return self.fusion(torch.cat(z,dim=-1))

    def forward(self,features,curves):
        return self.head(self.encode(features,curves)).squeeze(-1)
