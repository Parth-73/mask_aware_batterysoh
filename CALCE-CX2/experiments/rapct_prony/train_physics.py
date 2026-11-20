"""Train Stage 3 on whole chronological segments with a validation-selected frozen encoder."""
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.common import save_json,digest
from .audit import OUT
from .data import FEATURES,chronological_segments
from .metrics import metrics
from .physics import DegradationBranch,loss_terms
from .temporal import Stage2
from .train import seed_all,validate_pretraining,aggregate


def selected_encoder():
    choices=[]
    for path in (OUT/'checkpoints').glob('transformer_*/metrics.csv'):
        m=pd.read_csv(path);v=m[m.split.eq('validation')]
        cfg=json.loads((path.parent/'config.json').read_text())
        if cfg['outer']=='CX2_38' and cfg['validation']=='CX2_36' and cfg['seed']==42:
            choices.append((float(v.rmse.mean()),path.parent))
    if not choices:raise ValueError('no_completed_transformer_gate')
    return min(choices,key=lambda p:p[0])[1]


def train_physics(physics_weight=.01,smooth_weight=.001,epochs=50):
    seed_all(42);source=selected_encoder()
    previous=json.loads((source/'config.json').read_text())
    splits=json.loads((source/'splits.json').read_text())
    manifest=json.loads((OUT/'processed/manifest.json').read_text())
    validate_pretraining(source/'best.pt',splits['train'],manifest)
    config=dict(stage='physics',seed=42,label='reduced_budget_physics_pilot',source_encoder=str(source),source_checkpoint_sha256=digest(source/'best.pt'),
        physics_weight=physics_weight,smooth_weight=smooth_weight,epochs=epochs,lr=.0005,weight_decay=.001,batch_size_segments=16,
        outer='CX2_38',validation='CX2_36',warmup_supervised_epochs=10,warmup_ramp_end=30,
        state_initialization='encoder_estimated_per_unknown_segment',units='fractional_soh_and_observed_cycle_increments',
        interpretation='physics_guided_mathematical_priors_not_electrochemical_PDE')
    identifier=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:12]
    name=('physics_zero_' if physics_weight==smooth_weight==0 else 'physics_guided_')+identifier
    run=OUT/'checkpoints'/name;run.mkdir(exist_ok=False)
    save_json(run/'config.json',config);save_json(run/'splits.json',splits);save_json(run/'dataset_manifest.json',manifest)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    encoder=Stage2(previous['pathway'],previous['channels'],previous['embedding'],previous['frozen']).to(device)
    encoder.load_state_dict(torch.load(source/'best.pt',map_location=device,weights_only=True)['model']);encoder.eval()
    for p in encoder.parameters():p.requires_grad_(False)
    f=pd.read_csv(OUT/'processed/samples.csv');curves=np.load(OUT/'processed/curves.npy')
    scales=joblib.load(source/'scalers.joblib');fs,cs=scales['features'],scales['curves']
    x=fs.transform(f[FEATURES].to_numpy(float)).astype('float32')
    c=cs.transform(curves.transpose(0,2,1).reshape(-1,3)).reshape(len(curves),176,3).transpose(0,2,1).astype('float32')
    windows=np.load(OUT/'processed/windows_L5.npy');target=f.iloc[windows[:,-1]].reset_index(drop=True)
    target['input_cycle_ids']=[json.dumps(f.iloc[w].cycle_id.tolist()) for w in windows]
    def embed(indices):
        parts=[]
        with torch.no_grad():
            for start in range(0,len(indices),64):
                w=windows[indices[start:start+64]]
                z=encoder.encode(torch.tensor(x[w],device=device),torch.tensor(c[w],device=device))
                parts.append(z.cpu().numpy())
        return np.concatenate(parts)
    cache=OUT/'processed'/f'transformer_embeddings_{source.name}.npy'
    h=np.load(cache) if cache.exists() else embed(np.arange(len(target)))
    if not cache.exists():np.save(cache,h)
    save_json(OUT/'features'/f'{source.name}_embedding_manifest.json',dict(path=str(cache),encoder_sha256=digest(source/'best.pt'),windows_sha256=digest(OUT/'processed/windows_L5.npy'),shape=list(h.shape)))
    all_segments=chronological_segments(target)
    segments={split:[s for s in all_segments if target.battery.iloc[s[0]] in batteries] for split,batteries in
        {'train':splits['train'],'validation':splits['validation'],'outer':splits['outer']}.items()}
    def collate(items):
        maxlen=max(map(len,items));a=np.zeros((len(items),maxlen,h.shape[1]),dtype='float32');y=np.zeros((len(items),maxlen),dtype='float32');mask=np.zeros((len(items),maxlen),dtype=bool)
        for i,s in enumerate(items):a[i,:len(s)]=h[s];y[i,:len(s)]=target.soh.iloc[s];mask[i,:len(s)]=True
        return torch.tensor(a,device=device),torch.tensor(y,device=device),torch.tensor(mask,device=device)
    loader=DataLoader(segments['train'],batch_size=16,shuffle=True,collate_fn=collate,num_workers=0,generator=torch.Generator().manual_seed(42))
    model=DegradationBranch(h.shape[1]).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
    def predict(split,representations=None):
        model.eval();data=h if representations is None else representations;tables=[]
        with torch.no_grad():
            for segment_id,s in enumerate(segments[split]):
                output=model(torch.tensor(data[s][None],device=device))
                table=target.iloc[s].copy()
                for key in ['prediction','degradation','rate','correction']:table[key]=output[key][0].cpu().numpy()
                table['reference']=float(output['reference']);table['segment_id']=segment_id;tables.append(table)
        return pd.concat(tables,ignore_index=True)
    history=[];best=float('inf');patience=0;start=time.perf_counter()
    for epoch in range(1,epochs+1):
        model.train();totals={'data':0.,'correction_prior':0.,'rate_smoothness':0.};n=0
        for a,y,mask in loader:
            optimizer.zero_grad(set_to_none=True);out=model(a)
            loss,terms=loss_terms(out,y,mask,epoch,physics_weight,smooth_weight)
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite_physics_loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            count=int(mask.sum());n+=count
            for k in totals:totals[k]+=float(terms[k].detach())*count
        table=predict('validation');score=float(np.mean([metrics(g.soh,g.prediction)['rmse'] for _,g in table.groupby('battery')]))
        history.append(dict(epoch=epoch,**{k:v/n for k,v in totals.items()},warmup=max(0.,min(1.,(epoch-10)/20)),validation_battery_macro_rmse=score))
        if score<best-1e-7:
            best=score;patience=0;torch.save({'model':model.state_dict(),'config':config,'epoch':epoch,'validation_score':score},run/'best.pt')
        else:patience+=1
        pd.DataFrame(history).to_csv(run/'training.csv',index=False)
        torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'config':config},run/'last.pt')
        if patience>=20:break
    training_time=time.perf_counter()-start
    model.load_state_dict(torch.load(run/'best.pt',map_location=device,weights_only=True)['model'])
    results=[];audits=[]
    for split in ['validation','outer']:
        ids=np.concatenate(segments[split])
        if device.type=='cuda':torch.cuda.synchronize()
        t=time.perf_counter();fresh=h.copy();fresh[ids]=embed(ids);table=predict(split,fresh)
        if device.type=='cuda':torch.cuda.synchronize()
        inference=time.perf_counter()-t
        table=table.rename(columns={'prediction':'predicted_soh'});table['actual_soh']=table.soh
        table['stage']='A4';table['config_id']=identifier;table['fold']='CX2_38';table['seed']=42
        table['verified_rest_status']='not_used';table['rest_duration_s']=np.nan
        table['signed_error']=table.predicted_soh-table.soh;table['absolute_error']=abs(table.signed_error)
        table.to_csv(run/f'predictions_{split}.csv',index=False)
        table.to_csv(OUT/'latent_states'/f'{name}_{split}.csv',index=False)
        increments=np.concatenate([np.diff(g.degradation) for _,g in table.groupby('segment_id')])
        audits.append(dict(split=split,negative_irreversible_increment_fraction=float(np.mean(increments<0)) if len(increments) else 0.,
            correction_boundary_saturation_fraction=float(np.mean(abs(table.correction)>.0297)),reference_factor=float(table.reference.iloc[0]),
            degradation_dynamics_residual='not_applicable_recurrence_is_definition_not_independent_residual',
            constraints='nonnegative_rate_by_parameterization; bounded_signed_measurement_correction; relative_rate_smoothness_prior'))
        for battery,g in table.groupby('battery'):
            r=metrics(g.soh,g.predicted_soh);r.update(model=name.split(identifier)[0].rstrip('_'),stage='A4',config_id=identifier,fold='CX2_38',seed=42,split=split,battery=battery,
                training_time_s=training_time,inference_time_s=inference,parameter_count=sum(p.numel() for p in encoder.parameters())+sum(p.numel() for p in model.parameters()),
                trainable_parameter_count=sum(p.numel() for p in model.parameters()),peak_gpu_memory_bytes=torch.cuda.max_memory_allocated() if device.type=='cuda' else 0,
                status=config['label'],epochs_completed=len(history),training_scope='new_state_head_only_frozen_encoder_pretraining_reported_separately',
                encoder_training_time_s=float(pd.read_csv(source/'metrics.csv').training_time_s.iloc[0]),physics_weight=physics_weight,smooth_weight=smooth_weight)
            results.append(r)
    pd.DataFrame(results).to_csv(run/'metrics.csv',index=False);pd.DataFrame(audits).to_csv(run/'physical_audit.csv',index=False)
    aggregate()
    print(json.dumps(dict(run=name,best_validation_rmse=best,epochs=len(history),state_head_training_s=training_time)),flush=True)

if __name__=='__main__':
    train_physics(0.,0.);train_physics(.01,.001)
