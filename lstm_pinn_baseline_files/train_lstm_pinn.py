"""CALCE CX2 LSTM vs physics-guided LSTM baseline.
Usage: python train_lstm_pinn.py --data-dir '/path/to/pre processed ready to use data' --epochs 80
"""
import argparse, json, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

FEATURES=['cc_duration_s','cv_duration_s','dva_peak','ic_peak_voltage','ic_peak_width']

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def read_split(path):
    df=pd.read_csv(path,parse_dates=['cycle_start'])
    needed=['battery','cycle_ordinal','cycle_start','soh']+FEATURES
    assert set(needed).issubset(df.columns), f'Missing columns in {path}'
    assert df[needed].notna().all().all(),f'NaNs in {path}'
    assert not df.duplicated(['battery','cycle_ordinal']).any(),f'Duplicate cycles in {path}'
    return df.sort_values(['battery','cycle_ordinal']).reset_index(drop=True)

class Windows(Dataset):
    def __init__(self, df, mean, std, seq_len):
        self.X=[];self.y=[];self.meta=[]
        for bat,g in df.groupby('battery',sort=False):
            g=g.sort_values('cycle_ordinal').reset_index(drop=True)
            x=(g[FEATURES].to_numpy(dtype=np.float32)-mean)/std
            y=g.soh.to_numpy(dtype=np.float32)
            ords=g.cycle_ordinal.to_numpy()
            for end in range(len(g)):
                # Keep only strictly consecutive retained cycles; no implicit missing-cycle history.
                start=end
                while start>0 and end-start+1<seq_len and ords[start]-ords[start-1]==1:
                    start-=1
                window=x[start:end+1]
                pad=np.zeros((seq_len,len(FEATURES)),dtype=np.float32)
                pad[-len(window):]=window
                self.X.append(pad);self.y.append(y[end]);self.meta.append((bat,int(ords[end]),str(g.cycle_start.iloc[end])))
        self.X=torch.tensor(np.stack(self.X));self.y=torch.tensor(self.y,dtype=torch.float32)
    def __len__(self):return len(self.y)
    def __getitem__(self,i):return self.X[i],self.y[i]

class LSTMSOH(nn.Module):
    def __init__(self,hidden=64,layers=1,dropout=0.1):
        super().__init__()
        self.lstm=nn.LSTM(len(FEATURES),hidden,layers,batch_first=True,dropout=dropout if layers>1 else 0)
        self.head=nn.Sequential(nn.Linear(hidden,32),nn.ReLU(),nn.Dropout(dropout),nn.Linear(32,1))
    def forward(self,x):
        _,(h,_) = self.lstm(x)
        return self.head(h[-1]).squeeze(-1)

def metrics(y,p):
    y=np.asarray(y);p=np.asarray(p);e=p-y
    return dict(rmse=float(np.sqrt(np.mean(e**2))),mae=float(np.mean(np.abs(e))),
                mape_pct=float(np.mean(np.abs(e)/np.maximum(np.abs(y),1e-8))*100),
                r2=float(1-np.sum(e**2)/max(np.sum((y-y.mean())**2),1e-12)),
                n=int(len(y)))

def predict(model,ds,device,batch_size):
    model.eval();pred=[]
    with torch.no_grad():
        for x,_ in DataLoader(ds,batch_size=batch_size,shuffle=False):
            pred.extend(model(x.to(device)).cpu().numpy().tolist())
    return np.asarray(pred)

def train_one(name,train_ds,val_ds,args,device,output):
    seed_all(args.seed)
    model=LSTMSOH(args.hidden,args.layers,args.dropout).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    loader=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True)
    best=float('inf');best_epoch=0;bad=0;history=[];t0=time.time()
    for epoch in range(1,args.epochs+1):
        model.train();losses=[];fit_losses=[];phys_losses=[]
        for x,y in loader:
            x=x.to(device);y=y.to(device);opt.zero_grad()
            pred=model(x);fit=((pred-y)**2).mean()
            # Physics-guided prior: predicted SOH should not increase when the
            # last two observations are genuinely consecutive and present.
            # This is a SOFT prior, not a governing battery differential equation.
            # Apply to a pair of separate current-cycle predictions below.
            phys=torch.zeros((),device=device)
            if name=='lstm_pinn':
                # Compare predictions at consecutive observed cycles only.
                # Windows with no previous valid cycle have zero-padded first positions.
                prev=torch.cat([torch.zeros_like(x[:,:1,:]),x[:,:-1,:]],dim=1)
                has_prev=(x[:,-2,:].abs().sum(dim=1)>1e-8)
                if has_prev.any():
                    prev_pred=model(prev[has_prev])
                    phys=torch.relu(pred[has_prev]-prev_pred).square().mean()
            loss=fit+(args.physics_weight*phys if name=='lstm_pinn' else 0.)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
            losses.append(loss.item());fit_losses.append(fit.item());phys_losses.append(phys.item())
        vp=predict(model,val_ds,device,args.batch_size)
        vm=metrics(val_ds.y.numpy(),vp)
        history.append(dict(epoch=epoch,train_loss=float(np.mean(losses)),train_mse=float(np.mean(fit_losses)),physics_loss=float(np.mean(phys_losses)),validation_rmse=vm['rmse']))
        if vm['rmse']<best-1e-6:
            best=vm['rmse'];best_epoch=epoch;bad=0
            torch.save({'state_dict':model.state_dict(),'features':FEATURES,'args':vars(args)},output/f'{name}.pt')
        else:bad+=1
        if epoch==1 or epoch%10==0:print(f'{name} epoch {epoch:03d} | val RMSE {vm["rmse"]:.5f} | best {best:.5f}',flush=True)
        if bad>=args.patience:break
    pd.DataFrame(history).to_csv(output/f'{name}_history.csv',index=False)
    checkpoint=torch.load(output/f'{name}.pt',map_location=device,weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    return model,dict(best_epoch=best_epoch,training_seconds=round(time.time()-t0,2))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--output',type=Path,default=Path('lstm_pinn_results'))
    p.add_argument('--epochs',type=int,default=80);p.add_argument('--patience',type=int,default=15)
    p.add_argument('--batch-size',type=int,default=128);p.add_argument('--seq-len',type=int,default=12)
    p.add_argument('--hidden',type=int,default=64);p.add_argument('--layers',type=int,default=1)
    p.add_argument('--dropout',type=float,default=0.1);p.add_argument('--lr',type=float,default=0.001)
    p.add_argument('--weight-decay',type=float,default=1e-4);p.add_argument('--physics-weight',type=float,default=0.5)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--device',default='auto')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    seed_all(args.seed)
    device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device)
    splits={s:read_split(args.data_dir/f'{s}.csv') for s in ['train','validation','test']}
    assert not (set(splits['train'].battery)&set(splits['validation'].battery)|set(splits['train'].battery)&set(splits['test'].battery)|set(splits['validation'].battery)&set(splits['test'].battery)), 'Battery leakage across splits'
    mean=splits['train'][FEATURES].mean().to_numpy(dtype=np.float32)
    std=splits['train'][FEATURES].std(ddof=0).to_numpy(dtype=np.float32);std=np.maximum(std,1e-8)
    (args.output/'scaler.json').write_text(json.dumps({'features':FEATURES,'mean':mean.tolist(),'std':std.tolist()},indent=2))
    ds={s:Windows(df,mean,std,args.seq_len) for s,df in splits.items()}
    results=[]
    for name in ['lstm','lstm_pinn']:
        model,info=train_one(name,ds['train'],ds['validation'],args,device,args.output)
        for split in ['train','validation','test']:
            pred=predict(model,ds[split],device,args.batch_size)
            y=ds[split].y.numpy()
            row={'model':name,'split':split,**metrics(y,pred),**info}
            results.append(row)
            out=pd.DataFrame(ds[split].meta,columns=['battery','cycle_ordinal','cycle_start'])
            out['soh_true']=y;out['soh_pred']=pred;out['error']=pred-y
            out.to_csv(args.output/f'{name}_{split}_predictions.csv',index=False)
    pd.DataFrame(results).to_csv(args.output/'metrics.csv',index=False)
    (args.output/'config.json').write_text(json.dumps({**vars(args),'data_dir':str(args.data_dir),'output':str(args.output),'device':str(device)},indent=2))
    print('\n',pd.DataFrame(results).to_string(index=False));print('Saved:',args.output.resolve())
if __name__=='__main__':main()
