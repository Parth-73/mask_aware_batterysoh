"""Shared validation-only fitting; outer inference is an explicit separate frozen stage."""
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from src.benchmark.engine import metrics
from src.training.config import json_write
from src.common import digest
from .prepare import OUT
from .dataset import prepare
from .runtime import seed_all,identity,atomic_torch,atomic_csv,rng_state,restore_rng,synchronize,definition,build


def gpu_data(data,device):
    return (torch.tensor(data['features'],device=device),torch.tensor(data['curves'],device=device),
            torch.tensor(data['indices'],device=device),
            torch.tensor(data['metadata'].soh.to_numpy(np.float32),device=device))


def encode_cache(model, tensors, sample_ids):
    if not getattr(model,'frozen_encoder',False):return None
    f,c,index,_=tensors
    unique=torch.unique(index[sample_ids].reshape(-1))
    cache=torch.zeros((len(f),64),device=f.device)
    model.eval()
    with torch.inference_mode():
        for ids in unique.split(256):cache[ids]=model.encode_cycles(f[ids],c[ids]).float()
    return cache


def forward(model,tensors,ids,cache=None):
    f,c,index,_=tensors
    windows=index[ids]
    if cache is not None:return model.forward_embeddings(cache[windows])
    return model(f[windows],c[windows])


def predict(model,tensors,ids,cache=None,batch=128):
    model.eval();values={}
    with torch.inference_mode():
        for chunk in torch.as_tensor(ids,device=tensors[0].device).split(batch):
            output=forward(model,tensors,chunk,cache)
            for key,value in output.items():
                if isinstance(value,torch.Tensor):values.setdefault(key,[]).append(value.float().cpu().numpy())
    return {key:np.concatenate(v) for key,v in values.items()}


def save_predictions(directory,config,data,ids,output,role):
    meta=data['metadata'].iloc[ids].reset_index(drop=True).copy()
    if config['module']=='stage1':meta['input_cycle_ids']=meta.cycle_id.map(lambda x:json.dumps([x]))
    for key in ['stage','outer','validation','seed','purpose','length']:
        meta[key]=config.get(key)
    meta['configuration_id']=directory.name
    meta['evaluation_split']=role
    meta['actual_soh']=meta.soh
    meta['predicted_soh']=output['prediction']
    meta['signed_error']=meta.predicted_soh-meta.actual_soh
    meta['absolute_error']=abs(meta.signed_error)
    meta['squared_error']=meta.signed_error**2
    meta['verified_rest_status']='not_used; see independently audited inventory'
    meta['rest_duration_s']=np.nan
    atomic_csv(directory/f'predictions_{role}.csv',meta)
    for key,value in output.items():
        if key!='prediction':np.save(directory/f'{key}_{role}.npy',value)
    return meta


def fit(config, progress=None):
    frozen=definition(config)
    ident=f"{config['stage']}_{config['outer']}_{config['validation']}_s{config['seed']}_"+identity(frozen)[:12]
    directory=OUT/'checkpoints'/ident
    directory.mkdir(parents=True,exist_ok=True)
    result_path=directory/'validation_result.json'
    if result_path.exists():
        result=json.loads(result_path.read_text())
        assert digest(directory/'best.pt')==result['checkpoint_sha256']
        return directory,result
    if (OUT/'configs'/f"{config['outer']}_outer_frozen.json").exists():
        raise RuntimeError('Outer evaluation choices are frozen; no further fitting in this study')
    json_write(directory/'configuration.json',frozen)
    seed_all(config['seed']);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data=prepare(config);tensors=gpu_data(data,device)
    model=build(config).to(device)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=config.get('lr',5e-4),weight_decay=config.get('weight_decay',1e-3))
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=.5,patience=5,min_lr=1e-6)
    joblib.dump(data['scalers'],directory/'scalers.joblib')
    json_write(directory/'preprocessing.json',data['preprocessing'])
    tr=data['groups']['train'];va=data['groups']['validation']
    cache=encode_cache(model,tensors,np.r_[tr,va])
    target=tensors[3]
    start_epoch,best,stale,history,best_epoch,seconds=1,float('inf'),0,[],0,0.
    best_state=None
    resume=directory/'resume.pt'
    if resume.exists():
        saved=torch.load(resume,map_location='cpu',weights_only=False)
        assert saved['definition']==frozen
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        start_epoch,best,stale,history,best_epoch,seconds=saved['epoch']+1,saved['best'],saved['stale'],saved['history'],saved['best_epoch'],saved['seconds']
        best_state=saved['best_state'];restore_rng(saved['rng'])
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    synchronize(device);begin=time.perf_counter()
    for epoch in range(start_epoch,config['epochs']+1):
        if stale>=config.get('patience',20):break
        model.train();order=torch.as_tensor(tr,device=device)[torch.randperm(len(tr),device=device)]
        total=0.;batch=config.get('batch_size',32);accum=config.get('accumulation',1)
        for group in order.split(batch*accum):
            optimizer.zero_grad(set_to_none=True)
            for ids in group.split(batch):
                with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=config.get('amp',False) and device.type=='cuda'):
                    pred=forward(model,tensors,ids,cache)['prediction']
                loss=(pred.float()-target[ids]).square().mean()
                if not torch.isfinite(loss):raise FloatingPointError(ident)
                (loss*len(ids)/len(group)).backward();total+=loss.detach().item()*len(ids)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        out=predict(model,tensors,va,cache)
        actual=data['metadata'].soh.to_numpy()[va]
        score=float(np.sqrt(np.mean((actual-out['prediction'])**2)))
        scheduler.step(score)
        if score<best-1e-8:
            best,best_epoch,stale=score,epoch,0
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:stale+=1
        history.append({'epoch':epoch,'data_loss':total/len(tr),'validation_battery_macro_rmse':score,
                        'learning_rate':optimizer.param_groups[0]['lr'],'physics_loss':0.,'smoothness_loss':0.})
        synchronize(device)
        atomic_torch(resume,{'definition':frozen,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
            'epoch':epoch,'best':best,'best_epoch':best_epoch,'stale':stale,'history':history,'best_state':best_state,
            'seconds':seconds+time.perf_counter()-begin,'rng':rng_state()})
        atomic_csv(directory/'training.csv',pd.DataFrame(history))
        if progress is not None:progress(epoch,score)
    synchronize(device);seconds+=time.perf_counter()-begin
    model.load_state_dict(best_state)
    atomic_torch(directory/'best.pt',{'state_dict':best_state,'config':config,'definition':frozen,'best_epoch':best_epoch})
    out=predict(model,tensors,va,cache)
    save_predictions(directory,config,data,va,out,'validation')
    result={**config,'configuration_id':ident,'evaluation_split':'validation','held_out_battery':config['validation'],
            'training_seconds':seconds,'best_epoch':best_epoch,'epochs_completed':len(history),
            'parameter_count':sum(p.numel() for p in model.parameters()),
            'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'peak_gpu_memory_mb':torch.cuda.max_memory_allocated()/1024**2 if device.type=='cuda' else 0.,
            'checkpoint_path':str(directory/'best.pt'),'checkpoint_sha256':digest(directory/'best.pt'),
            'cohort_ids_sha256':identity(data['metadata'].cycle_id.tolist()),**metrics(data['metadata'].soh.to_numpy()[va],out['prediction'])}
    json_write(result_path,result)
    print(f"FITTED {ident}: validation RMSE={best:.6f}, {seconds:.1f}s",flush=True)
    return directory,result


def evaluate_outer(directory):
    directory=Path(directory)
    result=json.loads((directory/'validation_result.json').read_text());config=result.copy()
    config=json.loads((directory/'configuration.json').read_text())['parameters']
    marker=OUT/'configs'/f"{config['outer']}_outer_frozen.json"
    frozen=json.loads(marker.read_text())
    if directory.name not in frozen['selected_run_ids']:raise ValueError('Run not in frozen outer selection')
    path=directory/'outer_result.json'
    if path.exists():return json.loads(path.read_text())
    assert digest(directory/'best.pt')==frozen['checkpoint_hashes'][directory.name]
    seed_all(config['seed']);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data=prepare(config);tensors=gpu_data(data,device)
    model=build(config).to(device)
    model.load_state_dict(torch.load(directory/'best.pt',map_location=device,weights_only=False)['state_dict'])
    ids=data['groups']['outer'];cache=encode_cache(model,tensors,ids)
    model.eval()
    synchronize(device);begin=time.perf_counter();out=predict(model,tensors,ids,cache);synchronize(device)
    elapsed=time.perf_counter()-begin
    table=save_predictions(directory,config,data,ids,out,'outer')
    record={**result,'evaluation_split':'outer','held_out_battery':config['outer'],
            'inference_seconds':elapsed,'inference_ms_per_sample':elapsed*1000/len(ids),
            'inference_scope':'forward and CPU transfer; frozen-encoder preprocessing excluded',
            **metrics(table.actual_soh,table.predicted_soh)}
    json_write(path,record)
    return record
