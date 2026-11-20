"""Table-3 predictor dimensions; dropout is an explicitly unreported setting."""
from torch import nn
class PaperLSTM(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.lstm=nn.LSTM(3,config.hidden_size,num_layers=config.layers,dropout=config.dropout,batch_first=True)
        self.head=nn.Linear(config.hidden_size,1)
    def forward(self,x,return_attention=False):
        _,(hidden,cell)=self.lstm(x);prediction=self.head(hidden[-1]).squeeze(-1)
        return (prediction,None) if return_attention else prediction
