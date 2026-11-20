"""Fractional SOH metrics; percentage points are explicit separate columns."""
import numpy as np
from sklearn.metrics import r2_score


def metrics(y,p):
    y,p=np.asarray(y,float),np.asarray(p,float)
    if y.shape!=p.shape or not len(y) or not np.isfinite([y,p]).all():raise ValueError('invalid_metrics_input')
    e=p-y;a=np.abs(e)
    mae=a.mean();rmse=np.sqrt(np.mean(e**2))
    return dict(mae=mae,mse=np.mean(e**2),rmse=rmse,mae_pp=100*mae,rmse_pp=100*rmse,
        mape_pct=100*np.mean(a/np.maximum(np.abs(y),1e-8)),smape_pct=200*np.mean(a/np.maximum(np.abs(y)+np.abs(p),1e-8)),
        r2=r2_score(y,p) if len(y)>1 else float('nan'),median_absolute_error=np.median(a),maximum_absolute_error=a.max(),
        p95_absolute_error=np.quantile(a,.95),prediction_bias=e.mean(),n=len(y))


def read_metrics(path):
    """Hex IDs can resemble overflowing scientific notation; preserve lexical identifiers."""
    import pandas as pd
    return pd.read_csv(path,dtype={"config_id":"string"})
