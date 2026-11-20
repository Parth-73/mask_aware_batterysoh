"""Stage 2: causal, masked Transformer over independently encoded charging curves."""
import math
from pathlib import Path
import torch
from torch import nn
from ..models import Stage1


class CausalBlock(nn.Module):
    def __init__(self,width=64,heads=4,ffn=128,dropout=.1):
        super().__init__()
        self.attention=nn.MultiheadAttention(width,heads,dropout=dropout,batch_first=True)
        self.norm1=nn.LayerNorm(width);self.norm2=nn.LayerNorm(width)
        self.ff=nn.Sequential(nn.Linear(width,ffn),nn.GELU(),nn.Dropout(dropout),nn.Linear(ffn,width))
        self.dropout=nn.Dropout(dropout)

    def forward(self,x,padding=None):
        length=x.shape[1]
        causal=torch.triu(torch.ones((length,length),device=x.device,dtype=torch.bool),diagonal=1)
        attended,weights=self.attention(x,x,x,attn_mask=causal,key_padding_mask=padding,
                                       need_weights=not self.training,average_attn_weights=False)
        if padding is not None:attended=attended.masked_fill(padding[...,None],0.)
        x=self.norm1(x+self.dropout(attended));x=self.norm2(x+self.dropout(self.ff(x)))
        if padding is not None:
            x=x.masked_fill(padding[...,None],0.)
            if weights is not None:weights=weights.masked_fill(padding[:,None,:,None],0.)
        return x,weights


class Stage2Model(nn.Module):
    temporal=True

    def __init__(self,config):
        super().__init__()
        parent=config.get('parent')
        if parent:
            checkpoint=torch.load(parent,map_location='cpu',weights_only=False)
            old=checkpoint['config']
            self.cycle_encoder=Stage1('fusion',old.get('channels',32),old.get('embedding',64))
            self.cycle_encoder.load_state_dict({k.removeprefix('network.'):v for k,v in checkpoint['state_dict'].items()})
        elif config.get('purpose')=='synthetic':
            self.cycle_encoder=Stage1('fusion',config.get('channels',16),64)
        else:raise ValueError('Stage 2 requires a completed Stage-1 parent checkpoint')
        self.cycle_encoder.head=nn.Identity()
        self.frozen_encoder=config.get('freeze_cnn',True)
        if self.frozen_encoder:
            self.cycle_encoder.requires_grad_(False)
        width=config.get('d_model',64);heads=config.get('heads',4)
        if width%heads:raise ValueError('d_model must be divisible by heads')
        self.width=width
        self.projection=nn.Linear(64,width)
        self.blocks=nn.ModuleList([CausalBlock(width,heads,config.get('ffn',128),config.get('dropout',.1)) for _ in range(config.get('layers',2))])
        self.head=nn.Sequential(nn.Linear(width,32),nn.GELU(),nn.Linear(32,1))
        position=torch.arange(256).float()[:,None]
        divisor=torch.exp(torch.arange(0,width,2).float()*(-math.log(10000.)/width))
        pe=torch.zeros(256,width);pe[:,0::2]=torch.sin(position*divisor);pe[:,1::2]=torch.cos(position*divisor)
        self.register_buffer('position',pe)

    def train(self,mode=True):
        super().train(mode)
        if self.frozen_encoder:self.cycle_encoder.eval()
        return self

    def encode_cycles(self,features,curves):
        return self.cycle_encoder.encode(features,curves)

    def encode_history(self,embeddings,padding=None):
        if padding is not None and padding.all(1).any():raise ValueError('Entirely unavailable history')
        x=self.projection(embeddings)+self.position[:embeddings.shape[1]]
        if padding is not None:x=x.masked_fill(padding[...,None],0.)
        attentions=[]
        for block in self.blocks:
            x,a=block(x,padding)
            if a is not None:attentions.append(a)
        return x,torch.stack(attentions,dim=1) if attentions else None

    def forward_embeddings(self,embeddings,padding=None):
        x,attention=self.encode_history(embeddings,padding)
        if padding is None:last=x[:,-1]
        else:
            index=torch.arange(x.shape[1],device=x.device).expand(x.shape[:2]).masked_fill(padding,-1).max(1).values
            last=x[torch.arange(len(x),device=x.device),index]
        result={'prediction':self.head(last).squeeze(-1),'embedding':last}
        if attention is not None:result['attention']=attention
        return result

    def forward(self,features,curves,padding=None):
        b,l,_=features.shape
        encoded=self.encode_cycles(features.reshape(b*l,5),curves.reshape(b*l,3,176)).reshape(b,l,64)
        return self.forward_embeddings(encoded,padding)


def build(config):return Stage2Model(config)
