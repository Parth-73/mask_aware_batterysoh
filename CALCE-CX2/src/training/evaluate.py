"""Explicit metric units and GPU-synchronized single-pass inference timing."""
import time
import numpy as np
import torch
from sklearn.metrics import r2_score


def metrics(actual,predicted):
    y=np.asarray(actual,dtype=float).reshape(-1);p=np.asarray(predicted,dtype=float).reshape(-1)
    if len(y)!=len(p) or not len(y) or not (np.isfinite(y)&np.isfinite(p)).all():raise ValueError('Invalid predictions/targets')
    e=p-y;mae=float(np.mean(abs(e)));mse=float(np.mean(e**2));rmse=float(np.sqrt(mse))
    nonzero=abs(y)>1e-8
    return dict(n_samples=len(y),mae_fraction=mae,rmse_fraction=rmse,mse_fraction_squared=mse,
                mae_percentage_points=mae*100,rmse_percentage_points=rmse*100,mse_percentage_points_squared=mse*10000,
                mape_percent=float(np.mean(abs(e[nonzero]/y[nonzero]))*100) if nonzero.any() else np.nan,
                mape_excluded_zero_targets=int((~nonzero).sum()),r2=float(r2_score(y,p)) if len(y)>1 else np.nan)


def synchronize(device):
    if torch.device(device).type=='cuda':torch.cuda.synchronize(device)


def predict(model,x,device,batch_size=1024,timed=False):
    model.eval();predictions=[];attentions=[]
    synchronize(device);start=time.perf_counter()
    with torch.inference_mode():
        for j in range(0,len(x),batch_size):
            p,w=model(x[j:j+batch_size],return_attention=True)
            predictions.append(p)
            if w is not None:attentions.append(w)
    synchronize(device);elapsed=time.perf_counter()-start
    # Report CUDA forward wall time separately from device-to-host transfer/metric calculation.
    p=torch.cat(predictions).detach().cpu().numpy()
    w=torch.cat(attentions).detach().cpu().numpy() if attentions else None
    return p,w,elapsed
