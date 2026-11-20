"""Reproducible Stage-1 training on a matched complete-case curve cohort."""
import argparse
import copy
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.common import ROOT, save_json
from .audit import OUT
from .data import FEATURES, BATTERIES, fit_scalers
from .metrics import metrics,read_metrics
from .models import Stage1


@dataclass
class Config:
    pathway:str='fusion'
    seed:int=42
    epochs:int=150
    patience:int=20
    batch_size:int=32
    lr:float=.0005
    weight_decay:float=.001
    channels:int=32
    embedding:int=64
    accumulation:int=1
    amp:bool=False
    workers:int=0
    outer:str='CX2_38'
    validation:str='CX2_36'
    label:str='pilot'
    initialization:str=''
    stage:str='cycle'
    sequence_length:int=5
    frozen:bool=False
    encoder_checkpoint:str=''
    refit_all_inner:bool=False
    transformer_dimension:int=64
    transformer_heads:int=4
    transformer_layers:int=2
    transformer_ffn:int=128
    transformer_dropout:float=.1


def validate_pretraining(checkpoint, training_batteries, dataset_manifest):
    source=Path(checkpoint).parent
    if not (source/'splits.json').exists() or not (source/'dataset_manifest.json').exists():raise ValueError('pretraining_provenance_missing')
    previous=json.loads((source/'splits.json').read_text())
    if not set(previous['train']).issubset(set(training_batteries)):raise ValueError('pretraining_contains_held_out_battery')
    if json.loads((source/'dataset_manifest.json').read_text())!=dataset_manifest:raise ValueError('pretraining_dataset_changed')


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.set_num_threads(4)


def train(config, trial=None, resume_run=None):
    seed_all(config.seed)
    f=pd.read_csv(OUT/'processed/samples.csv');c=np.load(OUT/'processed/curves.npy')
    train_batteries=[b for b in BATTERIES if b!=config.outer and (config.refit_all_inner or b!=config.validation)]
    if config.outer==config.validation:raise ValueError('overlapping_battery_splits')
    fs,cs=fit_scalers(f[FEATURES].to_numpy(float),c,f.battery.to_numpy(),train_batteries)
    ft=fs.transform(f[FEATURES].to_numpy(float)).astype('float32')
    ct=cs.transform(c.transpose(0,2,1).reshape(-1,3)).reshape(len(c),c.shape[2],3).transpose(0,2,1).astype('float32')
    if config.stage=='transformer':
        from .data import chronological_windows
        windows,audit=chronological_windows(f,config.sequence_length)
        if len(windows)==0:raise ValueError('no_chronological_windows')
        cycle_ids=f.cycle_id.to_numpy()
        f=f.iloc[windows[:,-1]].copy()
        f['input_cycle_ids']=[json.dumps(cycle_ids[w].tolist()) for w in windows]
        ft=ft[windows];ct=ct[windows]
    model_name=(config.pathway if config.stage=='cycle' else 'transformer_'+('frozen' if config.frozen else 'joint'))+('_refit' if config.refit_all_inner else '')
    sets={s:np.flatnonzero(f.battery.isin(bs)) for s,bs in {'train':train_batteries,'validation':[config.validation],'outer':[config.outer]}.items()}
    if any(len(v)==0 for v in sets.values()):raise ValueError('empty_split')
    config_id=__import__('hashlib').sha256(json.dumps(asdict(config),sort_keys=True).encode()).hexdigest()[:12]
    run=Path(resume_run) if resume_run else OUT/'checkpoints'/f'{model_name}_{config_id}'
    if resume_run and json.loads((run/'dataset_manifest.json').read_text()) != json.loads((OUT/'processed/manifest.json').read_text()):raise ValueError('resume_dataset_changed')
    run.mkdir(exist_ok=bool(resume_run))
    save_json(run/'config.json',asdict(config))
    save_json(run/'dataset_manifest.json',json.loads((OUT/'processed/manifest.json').read_text()))
    save_json(run/'splits.json',dict(train=train_batteries,validation=[] if config.refit_all_inner else [config.validation],outer=[config.outer],status='outer_refit_fixed_epochs_no_validation_selection' if config.refit_all_inner else 'cross_battery_pilot_not_nested_final'))
    joblib.dump({'features':fs,'curves':cs},run/'scalers.joblib')
    save_json(run/'normalization.json',dict(features=FEATURES,feature_mean=fs.mean_.tolist(),feature_scale=fs.scale_.tolist(),
        curve_channels=['Q_Ah','dQdV_Ah_per_V','I_A'],curve_mean=cs.mean_.tolist(),curve_scale=cs.scale_.tolist(),fit_batteries=train_batteries))
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    if config.stage=='cycle':model=Stage1(config.pathway,config.channels,config.embedding).to(device)
    elif config.stage=='transformer':
        from .temporal import Stage2
        if not config.encoder_checkpoint:raise ValueError('stage2_requires_executed_stage1_checkpoint')
        validate_pretraining(config.encoder_checkpoint,train_batteries,json.loads((OUT/'processed/manifest.json').read_text()))
        from src.common import digest
        save_json(run/'encoder_initialization.json',dict(path=config.encoder_checkpoint,sha256=digest(config.encoder_checkpoint),training_batteries=train_batteries))
        model=Stage2(config.pathway,config.channels,config.embedding,config.frozen,config.encoder_checkpoint,d_model=config.transformer_dimension,heads=config.transformer_heads,layers=config.transformer_layers,ffn=config.transformer_ffn,dropout=config.transformer_dropout).to(device)
    else:raise ValueError('unknown_stage')
    if config.initialization:
        validate_pretraining(config.initialization,train_batteries,json.loads((OUT/'processed/manifest.json').read_text()))
        model.load_state_dict(torch.load(config.initialization,map_location=device,weights_only=True)['model'])
        from src.common import digest
        save_json(run/'initialization.json',dict(path=config.initialization,sha256=digest(config.initialization),optimizer='new_adamw_not_exact_resume'))
    optimizer=torch.optim.AdamW(model.parameters(),lr=config.lr,weight_decay=config.weight_decay)
    scaler=torch.amp.GradScaler('cuda',enabled=config.amp and device.type=='cuda')
    def loader(indices,shuffle=False):
        return DataLoader(TensorDataset(torch.from_numpy(ft[indices]),torch.from_numpy(ct[indices]),torch.tensor(f.soh.to_numpy()[indices],dtype=torch.float32)),
            batch_size=config.batch_size,shuffle=shuffle,num_workers=min(config.workers,2),generator=torch.Generator().manual_seed(config.seed))
    training=loader(sets['train'],True);validation=loader(sets['validation'])
    def predict(data,embeddings=False):
        model.eval();pred=[];zs=[]
        with torch.no_grad():
            for a,b,y in data:
                a,b=a.to(device),b.to(device)
                z=model.encode(a,b)
                pred.append(model.head(z).squeeze(-1).float().cpu().numpy())
                if embeddings:zs.append(z.cpu().numpy())
        return np.concatenate(pred),np.concatenate(zs) if zs else None
    best=float('inf');wait=0;history=[];start_epoch=0;previous_elapsed=0.
    if resume_run:
        saved=torch.load(run/'last.pt',map_location=device,weights_only=True)
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler'])
        best=saved['best'];wait=saved['wait'];start_epoch=saved['epoch'];previous_elapsed=saved.get('elapsed_s',0.)
        history=pd.read_csv(run/'training.csv').iloc[:start_epoch].to_dict('records')
        torch.set_rng_state(saved['torch_rng'].cpu())
        if saved['cuda_rng']:torch.cuda.set_rng_state_all([x.cpu() for x in saved['cuda_rng']])
        if 'loader_rng' in saved:training.generator.set_state(saved['loader_rng'].cpu())
        else:raise ValueError('legacy_checkpoint_has_no_loader_rng_use_explicit_weights_continuation')
    start=time.perf_counter()
    for epoch in range(start_epoch,config.epochs):
        model.train();optimizer.zero_grad(set_to_none=True);total=0
        for step,(a,b,y) in enumerate(training):
            a,b,y=a.to(device),b.to(device),y.to(device)
            with torch.autocast(device_type=device.type,enabled=config.amp,dtype=torch.float16):
                pred=model(a,b);loss=(pred.float()-y).square().mean()
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite_training_loss')
            # Final partial accumulation uses its actual number of minibatches.
            group_start=(step//config.accumulation)*config.accumulation
            divisor=min(config.accumulation,len(training)-group_start)
            scaler.scale(loss/divisor).backward();total+=loss.item()*len(y)
            if (step+1)%config.accumulation==0 or step+1==len(training):
                scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
        p,_=predict(validation)
        vf=f.iloc[sets['validation']].copy();vf['p']=p
        score=float(np.mean([metrics(g.soh,g.p)['rmse'] for _,g in vf.groupby('battery')]))
        row=dict(epoch=epoch+1,train_mse=total/len(sets['train']))
        row['refit_train_diagnostic_rmse' if config.refit_all_inner else 'validation_battery_macro_rmse']=score
        history.append(row)
        if config.refit_all_inner or score<best-1e-7:
            best=score;wait=0
            torch.save({'model':model.state_dict(),'config':asdict(config),'epoch':epoch+1,('training_diagnostic_rmse' if config.refit_all_inner else 'validation_score'):best},run/'best.pt')
        else:wait+=1
        pd.DataFrame(history).to_csv(run/'training.csv',index=False)
        torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),'epoch':epoch+1,'config':asdict(config),'best':best,'wait':wait,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else [],'loader_rng':training.generator.get_state(),'elapsed_s':previous_elapsed+time.perf_counter()-start},run/'last.pt')
        if trial is not None:
            if config.refit_all_inner:raise ValueError('outer_refit_must_not_tune')
            trial.report(score,epoch)
            if trial.should_prune():
                import optuna
                raise optuna.TrialPruned()
        if wait>=config.patience:break
    elapsed=previous_elapsed+time.perf_counter()-start
    model.load_state_dict(torch.load(run/'best.pt',map_location=device,weights_only=True)['model'])
    results=[]
    # Outer evaluation is excluded during optimization; it cannot influence trial ranking.
    for split in (['outer'] if config.refit_all_inner else ['validation'] if trial is not None else ['validation','outer']):
        ids=sets[split]
        if device.type=='cuda':torch.cuda.synchronize()
        t=time.perf_counter();p,z=predict(loader(ids),True)
        if device.type=='cuda':torch.cuda.synchronize()
        inference=time.perf_counter()-t
        table=f.iloc[ids][['battery','cycle_id','prediction_cutoff','soh']].copy()
        table['predicted_soh']=p;table['stage']='A3' if config.stage=='transformer' else 'A'+{'mlp':'0','cnn':'1','fusion':'2'}[config.pathway]
        table['config_id']=config_id;table['fold']=config.outer;table['seed']=config.seed
        table['input_cycle_ids']=f.iloc[ids].input_cycle_ids.to_numpy() if 'input_cycle_ids' in f else table.cycle_id.map(lambda s:json.dumps([s]));table['verified_rest_status']='not_used'
        table['rest_duration_s']=np.nan;table['signed_error']=p-table.soh;table['absolute_error']=abs(table.signed_error)
        table['actual_soh']=table.soh
        table.to_csv(run/f'predictions_{split}.csv',index=False);np.save(run/f'embeddings_{split}.npy',z)
        if getattr(model,'last_attention',None) is not None:
            np.save(run/f'attention_{split}_last_batch.npy',model.last_attention.cpu().numpy())
            (run/'ATTENTION.md').write_text('Attention summaries average heads-independent maps over the final inference minibatch, with dimensions layer/head/query/key. They are descriptive and do not establish causal feature importance. No future cycles can be attended.',encoding='utf-8')
        for battery,g in table.groupby('battery'):
            r=metrics(g.soh,g.predicted_soh);r.update(model=model_name,stage=table.stage.iloc[0],config_id=config_id,fold=config.outer,seed=config.seed,
                split=split,battery=battery,training_time_s=elapsed,inference_time_s=inference,parameter_count=sum(p.numel() for p in model.parameters()),
                device=str(device),peak_gpu_memory_bytes=torch.cuda.max_memory_allocated() if device.type=='cuda' else 0,status=config.label,epochs_completed=len(history))
            results.append(r)
    pd.DataFrame(results).to_csv(run/'metrics.csv',index=False)
    print(json.dumps(dict(run=str(run),selection_score=best,score_scope='training_diagnostic_only' if config.refit_all_inner else 'inner_validation',training_time_s=elapsed)),flush=True)
    return best,run


def aggregate():
    paths=list((OUT/'checkpoints').glob('*/metrics.csv'))
    if not paths:return
    tables=[]
    for path in paths:
        table=read_metrics(path);cfg=json.loads((path.parent/'config.json').read_text())
        table['dataset']='CALCE-CX2'
        pathway=cfg.get('pathway','fusion')
        table['feature_configuration']={'mlp':'five_engineered_only','cnn':'raw_Q_dQdV_I_only','fusion':'independently_encoded_raw_and_five_engineered'}[pathway]
        table['sequence_length']=cfg.get('sequence_length',5 if cfg.get('stage')=='physics' else 1) if cfg.get('stage','cycle')!='cycle' else 1
        table['prediction_contract']='end_of_current_charge'
        tables.append(table)
    all_runs=pd.concat(tables,ignore_index=True)
    all_runs.to_csv(OUT/'metrics/all_runs.csv',index=False)
    all_runs.to_csv(OUT/'metrics/stage_comparison.csv',index=False)
    all_runs[all_runs.split.eq('outer')].to_csv(OUT/'metrics/outer_fold_metrics.csv',index=False)
    grouping=['model','config_id','fold','seed','split']
    all_runs.groupby(grouping).mean(numeric_only=True).to_csv(OUT/'metrics/battery_macro_summary.csv')
    all_runs[['model','config_id','fold','seed','split','parameter_count','training_time_s','inference_time_s','peak_gpu_memory_bytes']].to_csv(OUT/'metrics/model_complexity.csv',index=False)
    all_runs.to_csv(OUT/'metrics/feature_ablation.csv',index=False)



def main():
    p=argparse.ArgumentParser();p.add_argument('--pathway',choices=['mlp','cnn','fusion','all'],default='all');p.add_argument('--epochs',type=int,default=150)
    p.add_argument('--batch-size',type=int,default=32);p.add_argument('--seed',type=int,default=42);p.add_argument('--label',default='pilot');p.add_argument('--amp',action='store_true');p.add_argument('--initialize',default='');p.add_argument('--resume',type=Path);p.add_argument('--stage',choices=['cycle','transformer'],default='cycle');p.add_argument('--encoder-checkpoint',default='');p.add_argument('--frozen',action='store_true');p.add_argument('--sequence-length',type=int,default=5);p.add_argument('--accumulation',type=int,default=1);p.add_argument('--workers',type=int,default=0)
    a=p.parse_args()
    if a.resume:
        train(Config(**json.loads((a.resume/'config.json').read_text())),resume_run=a.resume);aggregate();return
    for pathway in (['mlp','cnn','fusion'] if a.pathway=='all' else [a.pathway]):
        train(Config(pathway=pathway,epochs=a.epochs,batch_size=a.batch_size,seed=a.seed,label=a.label,amp=a.amp,initialization=a.initialize,stage=a.stage,encoder_checkpoint=a.encoder_checkpoint,frozen=a.frozen,sequence_length=a.sequence_length,accumulation=a.accumulation,workers=a.workers))
        aggregate()

if __name__=='__main__':main()
