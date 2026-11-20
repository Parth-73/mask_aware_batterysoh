"""Stage 2 causal cycle-order Transformer, added after Stage-1 execution."""
import math
import torch
from torch import nn
from .models import Stage1


class AttentionBlock(nn.Module):
    def __init__(self,d_model=64,heads=4,ffn=128,dropout=.1):
        super().__init__()
        self.attention=nn.MultiheadAttention(d_model,heads,dropout=dropout,batch_first=True)
        self.norm1=nn.LayerNorm(d_model);self.norm2=nn.LayerNorm(d_model)
        self.drop=nn.Dropout(dropout)
        self.ffn=nn.Sequential(nn.Linear(d_model,ffn),nn.GELU(),nn.Dropout(dropout),nn.Linear(ffn,d_model))

    def forward(self,x,mask,padding=None):
        update,weights=self.attention(x,x,x,attn_mask=mask,key_padding_mask=padding,need_weights=not self.training,average_attn_weights=False)
        x=self.norm1(x+self.drop(update));x=self.norm2(x+self.drop(self.ffn(x)))
        return x,weights


class Stage2(nn.Module):
    def __init__(self,pathway='fusion',channels=32,embedding=64,frozen=False,checkpoint='',d_model=64,heads=4,layers=2,ffn=128,dropout=.1):
        super().__init__()
        self.cycle=Stage1(pathway,channels,embedding)
        if checkpoint:self.cycle.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['model'])
        self.cycle.head=nn.Identity()
        if frozen:
            for p in self.cycle.parameters():p.requires_grad_(False)
        self.project=nn.Linear(64,d_model)
        self.blocks=nn.ModuleList([AttentionBlock(d_model,heads,ffn,dropout) for _ in range(layers)])
        self.head=nn.Sequential(nn.Linear(d_model,32),nn.GELU(),nn.Linear(32,1))
        position=torch.arange(4096).float()[:,None]
        div=torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.)/d_model))
        encoding=torch.zeros(4096,d_model);encoding[:,0::2]=torch.sin(position*div);encoding[:,1::2]=torch.cos(position*div)
        self.register_buffer('position',encoding)
        self.last_attention=None

    def encode_sequence(self,features,curves,padding_mask=None):
        b,t,_=features.shape
        z=self.cycle.encode(features.reshape(b*t,5),curves.reshape(b*t,*curves.shape[-2:])).reshape(b,t,64)
        z=self.project(z)+self.position[:t]
        mask=torch.triu(torch.ones(t,t,dtype=torch.bool,device=z.device),diagonal=1)
        attention=[]
        for block in self.blocks:
            z,w=block(z,mask,padding_mask)
            if w is not None:attention.append(w.detach())
        self.last_attention=torch.stack(attention).mean(dim=1) if attention else None
        return z

    def encode(self,features,curves,padding_mask=None):
        z=self.encode_sequence(features,curves,padding_mask)
        if padding_mask is None:return z[:,-1]
        last=(~padding_mask).sum(dim=1)-1
        if (last<0).any():raise ValueError('all_padding_sequence')
        return z[torch.arange(len(z),device=z.device),last]

    def forward(self,features,curves,padding_mask=None):
        return self.head(self.encode(features,curves,padding_mask)).squeeze(-1)
