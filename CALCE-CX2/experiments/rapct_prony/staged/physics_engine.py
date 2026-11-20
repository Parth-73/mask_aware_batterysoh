"""Chronological truncated-state training and coherent segment-level inference."""
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from src.common import digest
from src.training.config import json_write
from src.benchmark.engine import metrics
from .prepare import OUT
from .dataset import prepare
from .runtime import seed_all,identity,atomic_torch,atomic_csv,rng_state,restore_rng,synchronize,definition,build
from .engine import gpu_data,encode_cache,save_predictions
from .stage3 import activation


def segments(data,ids):
    meta=data['metadata'].iloc[ids]
    result=[np.asarray(group.index,dtype=int) for _,group in meta.groupby('segment_id',sort=False)]
    assert np.array_equal(np.concatenate(result),ids)
    return result


def infer(model,data,tensors,ids,cache,chunk_size=32):
    model.eval();values={};ordered=[]
    with torch.inference_mode():
        for segment in segments(data,ids):
            state=None
            for chunk in np.array_split(segment,np.ceil(len(segment)/chunk_size).astype(int)):
                index=torch.as_tensor(chunk,device=cache.device)
                output=model.forward_embeddings(cache[tensors[2][index]],state)
                state=output['state'];ordered.extend(chunk.tolist())
                for name,value in output.items():
                    if isinstance(value,torch.Tensor) and value.ndim>0 and len(value)==len(chunk):
                        values.setdefault(name,[]).append(value.cpu().numpy())
    assert np.array_equal(ordered,ids)
    return {name:np.concatenate(value) for name,value in values.items()}


def save_latent(directory,data,ids,out,role):
    frame=data['metadata'].iloc[ids][['battery','cycle_id','lifetime_cycle','segment_id','prediction_cutoff']].reset_index(drop=True).copy()
    for key in ['degradation','rate','state_proposal','observation_residual','reference']:
        frame[key]=out[key]
    frame['initial_state_policy']='encoder-estimated unknown history; carried within verified target segment'
    atomic_csv(directory/f'latent_{role}.csv',frame)
    increments=[]
    for _,group in frame.groupby('segment_id'):
        increments.extend(np.diff(group.degradation).tolist())
    return {'valid_transitions':len(increments),
            'negative_increment_fraction':float(np.mean(np.asarray(increments)<-1e-7)) if increments else None,
            'proxy_consistency_mse':float(np.mean((frame.state_proposal-frame.degradation)**2)),
            'observation_boundary_fraction':float(np.mean(abs(frame.observation_residual)>=.0297)),
            'minimum_rate':float(frame.rate.min()),'reference_range':[float(frame.reference.min()),float(frame.reference.max())]}


def fit(config,progress=None):
    frozen=definition(config)
    ident=f"A4_{config['outer']}_{config['validation']}_s{config['seed']}_"+identity(frozen)[:12]
    directory=OUT/'checkpoints'/ident;directory.mkdir(parents=True,exist_ok=True)
    path=directory/'validation_result.json'
    if path.exists():
        result=json.loads(path.read_text());assert digest(directory/'best.pt')==result['checkpoint_sha256']
        return directory,result
    if (OUT/'configs'/f"{config['outer']}_outer_frozen.json").exists():raise RuntimeError('Outer selection already frozen')
    json_write(directory/'configuration.json',frozen);seed_all(config['seed'])
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data=prepare(config);tensors=gpu_data(data,device);model=build(config).to(device)
    tr=data['groups']['train'];va=data['groups']['validation'];target=tensors[3]
    cache=encode_cache(model,tensors,np.r_[tr,va]);training=segments(data,tr)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=config.get('lr',5e-4),weight_decay=config.get('weight_decay',1e-3))
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=.5,patience=5,min_lr=1e-6)
    joblib.dump(data['scalers'],directory/'scalers.joblib');json_write(directory/'preprocessing.json',data['preprocessing'])
    epoch_start,best,stale,best_epoch,history,seconds=1,float('inf'),0,0,[],0.;best_state=None
    resume=directory/'resume.pt'
    if resume.exists():
        state=torch.load(resume,map_location='cpu',weights_only=False);assert state['definition']==frozen
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        epoch_start,best,stale,best_epoch,history,seconds=state['epoch']+1,state['best'],state['stale'],state['best_epoch'],state['history'],state['seconds']
        best_state=state['best_state'];restore_rng(state['rng'])
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    synchronize(device);begin=time.perf_counter()
    for epoch in range(epoch_start,config['epochs']+1):
        if stale>=config.get('patience',20):break
        model.train();totals=np.zeros(5);count=0
        # Shuffle whole segments only. Each segment's state is carried through ordered chunks.
        order=torch.randperm(len(training)).tolist()
        for position in order:
            state=None;segment=training[position]
            for chunk in np.array_split(segment,int(np.ceil(len(segment)/config.get('batch_size',32)))):
                ids=torch.as_tensor(chunk,device=device);optimizer.zero_grad(set_to_none=True)
                output=model.forward_embeddings(cache[tensors[2][ids]],state)
                data_loss=(output['prediction'].float()-target[ids]).square().mean()
                factor=activation(epoch,config)
                loss=data_loss+factor*(config['physics_weight']*output['physics_loss']+config['smooth_weight']*output['smoothness_loss'])
                if not torch.isfinite(loss):raise FloatingPointError(ident)
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
                state={k:v.detach() for k,v in output['state'].items()}
                totals+=np.array([data_loss.item(),output['physics_loss'].item(),output['smoothness_loss'].item(),output['proxy_consistency'].item(),output['observation_penalty'].item()])*len(ids)
                count+=len(ids)
        out=infer(model,data,tensors,va,cache,config.get('batch_size',32))
        score=float(np.sqrt(np.mean((data['metadata'].soh.to_numpy()[va]-out['prediction'])**2)))
        scheduler.step(score)
        if score<best-1e-8:
            best,best_epoch,stale=score,epoch,0
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:stale+=1
        totals/=count
        history.append({'epoch':epoch,'data_loss':totals[0],'physics_loss':totals[1],'smoothness_loss':totals[2],
            'proxy_consistency':totals[3],'observation_penalty':totals[4],
            'effective_physics_weight':activation(epoch,config)*config['physics_weight'],
            'effective_smooth_weight':activation(epoch,config)*config['smooth_weight'],
            'validation_battery_macro_rmse':score,'learning_rate':optimizer.param_groups[0]['lr']})
        synchronize(device)
        atomic_torch(resume,{'definition':frozen,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
            'epoch':epoch,'best':best,'best_epoch':best_epoch,'stale':stale,'history':history,'best_state':best_state,
            'seconds':seconds+time.perf_counter()-begin,'rng':rng_state()})
        atomic_csv(directory/'training.csv',pd.DataFrame(history))
        if progress:progress(epoch,score)
    synchronize(device);seconds+=time.perf_counter()-begin
    model.load_state_dict(best_state)
    atomic_torch(directory/'best.pt',{'state_dict':best_state,'config':config,'definition':frozen,'best_epoch':best_epoch})
    out=infer(model,data,tensors,va,cache)
    save_predictions(directory,config,data,va,out,'validation');audit=save_latent(directory,data,va,out,'validation')
    json_write(directory/'physical_audit_validation.json',audit)
    result={**config,'configuration_id':ident,'evaluation_split':'validation','held_out_battery':config['validation'],
        'training_seconds':seconds,'best_epoch':best_epoch,'epochs_completed':len(history),
        'parameter_count':sum(p.numel() for p in model.parameters()),
        'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
        'peak_gpu_memory_mb':torch.cuda.max_memory_allocated()/1024**2 if device.type=='cuda' else 0.,
        'checkpoint_path':str(directory/'best.pt'),'checkpoint_sha256':digest(directory/'best.pt'),
        'cohort_ids_sha256':identity(data['metadata'].cycle_id.tolist()),'state_policy':'TBPTT chunk carry and unknown-state segment reset',
        **metrics(data['metadata'].soh.to_numpy()[va],out['prediction'])}
    json_write(path,result);print(f'FITTED {ident}: validation RMSE={best:.6f}, {seconds:.1f}s',flush=True)
    return directory,result


def evaluate_outer(directory):
    directory=Path(directory);result=json.loads((directory/'validation_result.json').read_text())
    config=json.loads((directory/'configuration.json').read_text())['parameters']
    frozen=json.loads((OUT/'configs'/f"{config['outer']}_outer_frozen.json").read_text())
    assert directory.name in frozen['selected_run_ids'] and digest(directory/'best.pt')==frozen['checkpoint_hashes'][directory.name]
    path=directory/'outer_result.json'
    if path.exists():return json.loads(path.read_text())
    seed_all(config['seed']);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data=prepare(config);tensors=gpu_data(data,device);model=build(config).to(device)
    model.load_state_dict(torch.load(directory/'best.pt',map_location=device,weights_only=False)['state_dict'])
    ids=data['groups']['outer'];cache=encode_cache(model,tensors,ids)
    synchronize(device);begin=time.perf_counter();out=infer(model,data,tensors,ids,cache);synchronize(device)
    elapsed=time.perf_counter()-begin
    table=save_predictions(directory,config,data,ids,out,'outer');audit=save_latent(directory,data,ids,out,'outer')
    json_write(directory/'physical_audit_outer.json',audit)
    row={**result,'evaluation_split':'outer','held_out_battery':config['outer'],'inference_seconds':elapsed,
         'inference_ms_per_sample':elapsed*1000/len(ids),'inference_scope':'stateful forward and CPU transfer; frozen CNN preprocessing excluded',
         **metrics(table.actual_soh,table.predicted_soh)}
    json_write(path,row);return row
