"""Shared matched targets, battery partitions and training-only physical-unit scaling."""
from functools import lru_cache
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from .prepare import OUT, PARENT, FEATURES, BATTERIES


def inner_batteries(outer, count=2):
    i=BATTERIES.index(outer)
    return [BATTERIES[(i+j)%len(BATTERIES)] for j in range(1,count+1)]


def partitions(outer, validation):
    if outer==validation or outer not in BATTERIES or validation not in BATTERIES:
        raise ValueError('Invalid or overlapping battery partition')
    return {'train':[b for b in BATTERIES if b not in [outer,validation]],
            'validation':[validation],'outer':[outer]}


@lru_cache(maxsize=6)
def load(length=5, matched20=False):
    f=pd.read_csv(OUT/'processed/samples.csv')
    curves=np.load(PARENT/'processed/curves.npy',mmap_mode='r')
    idx=np.load(OUT/'processed'/f'indices_L{length}.npy')
    if matched20:
        target20=np.load(OUT/'processed/indices_L20.npy')[:,-1]
        idx=idx[np.isin(idx[:,-1],target20)]
    meta=f.iloc[idx[:,-1]].reset_index(drop=True).copy()
    meta['input_cycle_ids']=[json.dumps(f.cycle_id.iloc[w].tolist()) for w in idx]
    # State resets are derived entirely from causal provenance, never from SOH changes.
    continuation=(meta.battery.eq(meta.battery.shift()) & meta.sources.eq(meta.sources.shift()) &
                  meta.lifetime_cycle.diff().eq(1))
    meta['segment_id']=(~continuation).cumsum().to_numpy()
    return f,curves,idx,meta


def prepare(config):
    f,raw,index,meta=load(config.get('length',5),config.get('matched20',False))
    split=partitions(config['outer'],config['validation'])
    fit=f.battery.isin(split['train']).to_numpy()
    fs=StandardScaler().fit(f.loc[fit,FEATURES])
    cs=StandardScaler().fit(np.asarray(raw[fit]).transpose(0,2,1).reshape(-1,3))
    features=fs.transform(f[FEATURES]).astype(np.float32)
    curves=cs.transform(np.asarray(raw).transpose(0,2,1).reshape(-1,3)).reshape(len(raw),176,3).transpose(0,2,1).astype(np.float32)
    groups={role:np.flatnonzero(meta.battery.isin(batteries)) for role,batteries in split.items()}
    if any(not len(v) for v in groups.values()):
        raise ValueError('Empty battery partition')
    info={'partitions':split,'fit_cycle_ids':f.loc[fit,'cycle_id'].tolist(),'feature_order':FEATURES,
          'feature_mean':fs.mean_.tolist(),'feature_scale':fs.scale_.tolist(),
          'curve_order':['Q_Ah','dQdV_Ah_per_V','I_A'],'curve_mean':cs.mean_.tolist(),'curve_scale':cs.scale_.tolist(),
          'target_scale':'none; validated discharge Ah / 1.35 Ah, fractional, unclipped, unsmoothed',
          'cohort':'matched complete raw-coverage, consecutive original-cycle targets',
          'length':config.get('length',5),'matched20':config.get('matched20',False)}
    return {'frame':f,'features':features,'curves':curves,'indices':index,'metadata':meta,
            'groups':groups,'scalers':{'features':fs,'curves':cs},'preprocessing':info}
