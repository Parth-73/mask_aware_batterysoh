"""Use only the explicit five predictors; preserve fractional targets and metadata."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
FEATURES=json.loads((ROOT/'feature_order.json').read_text())['features']


def load_split(split='train',standardized=False):
    if split not in ['train','validation','test']:raise ValueError('Use train, validation or test.')
    table=pd.read_csv(ROOT/f'{split}.csv',float_precision='round_trip')
    values=pd.read_csv(ROOT/'standardized'/f'X_{split}.csv',float_precision='round_trip') if standardized else table[FEATURES]
    if list(values.columns)!=FEATURES or len(values)!=len(table):raise ValueError('Predictor order/row alignment mismatch.')
    x=values.to_numpy(dtype=np.float64);y=table.soh.to_numpy(dtype=np.float64)
    if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Nonfinite model input or target.')
    metadata=table.drop(columns=FEATURES+['soh'])
    return x,y,metadata


if __name__=='__main__':
    for name in ['train','validation','test']:
        x,y,_=load_split(name,standardized=True)
        print(f'{name}: X={x.shape}, y={y.shape}; SOH remains a fraction')
