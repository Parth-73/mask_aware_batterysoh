"""Pure PyTorch LSTM baselines; only past/current observations enter each window."""
import torch
from torch import nn

class LSTMBaseline(nn.Module):
    def __init__(self,input_size,architecture='B1',hidden_size=64):
        super().__init__()
        if architecture not in ['B1','B2','B3','B4']:raise ValueError(f'Unknown architecture {architecture}')
        self.architecture=architecture;self.hidden_size=hidden_size
        layers=2 if architecture in ['B2','B3'] else 1
        self.bidirectional=architecture=='B3'
        self.lstm=nn.LSTM(input_size,hidden_size,num_layers=layers,dropout=.2 if layers>1 else 0.,bidirectional=self.bidirectional,batch_first=True)
        width=hidden_size*(2 if self.bidirectional else 1)
        self.attention=nn.Sequential(nn.Linear(width,hidden_size),nn.Tanh(),nn.Linear(hidden_size,1,bias=False)) if architecture=='B4' else None
        self.head=nn.Linear(width,1)

    def forward(self,x,return_attention=False):
        output,(h,c)=self.lstm(x)
        weights=None
        if self.attention is not None:
            weights=torch.softmax(self.attention(output).squeeze(-1),dim=1)
            context=torch.sum(output*weights.unsqueeze(-1),dim=1)
        elif self.bidirectional:
            # h[-2] and h[-1] are the final forward and reverse states of the last layer.
            # output[:,-1] would omit most of the reverse history.
            context=torch.cat([h[-2],h[-1]],dim=-1)
        else:context=h[-1]
        prediction=self.head(context).squeeze(-1)
        return (prediction,weights) if return_attention else prediction

def parameter_count(model):return sum(p.numel() for p in model.parameters() if p.requires_grad)
