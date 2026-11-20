"""Paper-style within-battery LSTM reproduction, separate from the frozen benchmark.

python -m src.paper_reproduction.run --stage prepare
python -m src.paper_reproduction.run --smoke
python -m src.paper_reproduction.run --stage train
python -m src.paper_reproduction.run --stage test
python -m src.paper_reproduction.run --stage report
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,gc,json,time
from dataclasses import replace,asdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .config import ROOT,OUTPUT,BATTERIES,SEEDS,VARIANTS,Config,definition,file_sha,json_write
from .features import extract
from .dataset import prepare,Windows
from .model import PaperLSTM
from ..training.train_lstm import set_seed,setup_runtime
from ..training.evaluate import metrics,predict,synchronize


def prepare_all(config,output):
    output.mkdir(parents=True,exist_ok=True);data=extract(config,output)
    counts=[]
    for battery in BATTERIES:
        for variant in VARIANTS:
            windows=prepare(data,battery,variant,config,output)
            counts.append(dict(battery=battery,variant=variant,**{role+'_windows':len(w.y) for role,w in windows.items()}))
    pd.DataFrame(counts).to_csv(output/'sample_counts.csv',index=False)
    print(pd.DataFrame(counts).to_string(index=False),flush=True)


def load_data(output,battery,variant):
    directory=output/'datasets'/f'{battery}_{variant}'
    with np.load(directory/'windows.npz') as data:
        return {role:Windows(data[role+'_x'],data[role+'_y'],pd.read_csv(directory/f'{role}_windows.csv')) for role in ['inner_train','inner_validation','final_train','test']}


def identifier(battery,variant,seed):return f'{battery}_{variant}_seed{seed}'


def train_one(output,battery,variant,seed,config,smoke=False):
    name=identifier(battery,variant,seed);directory=output/'runs'/name;directory.mkdir(parents=True,exist_ok=True)
    done=directory/'training_metrics.json'
    if done.exists():return json.loads(done.read_text())
    if (output/'test_started.json').exists():raise RuntimeError('No new fitting after paper-experiment test exposure; use a new experiment directory')
    windows=load_data(output,battery,variant)
    tr,va,full=[windows[k] for k in ['inner_train','inner_validation','final_train']]
    if smoke:
        tr=Windows(tr.x[:128],tr.y[:128],tr.metadata.iloc[:128]);va=Windows(va.x[:64],va.y[:64],va.metadata.iloc[:64]);full=Windows(full.x[:128],full.y[:128],full.metadata.iloc[:128])
    device=torch.device('cuda');set_seed(seed);x=torch.from_numpy(tr.x).to(device);y=torch.from_numpy(tr.y).to(device);vx=torch.from_numpy(va.x).to(device)
    model=PaperLSTM(config).to(device);optimizer=torch.optim.Adam(model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=config.scheduler_factor,patience=config.scheduler_patience,min_lr=config.scheduler_min_lr)
    torch.cuda.reset_peak_memory_stats(device);best=float('inf');best_epoch=0;best_state=None;stale=0;history=[];loss_fn=torch.nn.MSELoss()
    synchronize(device);start=time.perf_counter()
    for epoch in range(1,config.max_epochs+1):
        model.train();permutation=torch.randperm(len(y),device=device);loss_sum=[]
        for j in range(0,len(y),config.batch_size):
            indices=permutation[j:j+config.batch_size];optimizer.zero_grad(set_to_none=True)
            prediction=model(x[indices]);loss=loss_fn(prediction,y[indices])
            if not torch.isfinite(loss):raise FloatingPointError(name+' has nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),config.gradient_clip);optimizer.step();loss_sum.append(loss.detach()*len(indices))
        p,_,_=predict(model,vx,device);m=metrics(va.y,p);lr=optimizer.param_groups[0]['lr'];scheduler.step(m['rmse_fraction'])
        if m['rmse_fraction']<best-config.min_delta:
            best=m['rmse_fraction'];best_epoch=epoch;stale=0;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:stale+=1
        history.append(dict(run_id=name,stage='inner_epoch_selection',epoch=epoch,train_mse=float(torch.stack(loss_sum).sum().item()/len(y)),validation_rmse=m['rmse_fraction'],learning_rate=lr))
        if epoch==1 or epoch%50==0:print(name,'selection epoch',epoch,'val_RMSE',round(m['rmse_fraction'],6),flush=True)
        if stale>=config.patience:break
    model.load_state_dict(best_state);vp,_,_=predict(model,vx,device)
    validation=metrics(va.y,vp);frame=va.metadata.copy();frame['actual_soh_fraction']=va.y;frame['predicted_soh_fraction']=vp;frame.to_csv(directory/'validation_predictions.csv',index=False)
    (output/'checkpoints').mkdir(exist_ok=True);torch.save(dict(state_dict=best_state,config=asdict(config),epoch=best_epoch,battery=battery,variant=variant,seed=seed),output/'checkpoints'/f'{name}_inner.pt')
    lr_schedule=[r['learning_rate'] for r in history[:best_epoch]]
    del model,optimizer,scheduler,x,y,vx;gc.collect();torch.cuda.empty_cache()
    # Final paper predictor trains on the entire first 50%, using the epoch budget and
    # LR schedule chosen only inside that first half. Test labels never enter this fit.
    set_seed(seed);model=PaperLSTM(config).to(device);optimizer=torch.optim.Adam(model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay)
    x=torch.from_numpy(full.x).to(device);y=torch.from_numpy(full.y).to(device)
    for epoch,lr in enumerate(lr_schedule,1):
        model.train();optimizer.param_groups[0]['lr']=lr;permutation=torch.randperm(len(y),device=device);losses=[]
        for j in range(0,len(y),config.batch_size):
            indices=permutation[j:j+config.batch_size];optimizer.zero_grad(set_to_none=True);p=model(x[indices]);loss=loss_fn(p,y[indices])
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),config.gradient_clip);optimizer.step();losses.append(loss.detach()*len(indices))
        history.append(dict(run_id=name,stage='refit_complete_training_half',epoch=epoch,train_mse=float(torch.stack(losses).sum().item()/len(y)),validation_rmse=np.nan,learning_rate=lr))
    synchronize(device);seconds=time.perf_counter()-start
    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    final_checkpoint=output/'checkpoints'/f'{name}_final.pt'
    torch.save(dict(state_dict=state,config=asdict(config),epoch=best_epoch,battery=battery,variant=variant,seed=seed,preprocessing=json.loads((output/'datasets'/f'{battery}_{variant}'/'preprocessing.json').read_text())['final'],target='validated fractional SOH'),final_checkpoint)
    result=dict(run_id=name,battery=battery,variant=variant,seed=seed,selected_epochs=best_epoch,inner_epochs_trained=sum(r['stage']=='inner_epoch_selection' for r in history),inner_train_windows=len(tr.y),validation_windows=len(va.y),final_train_windows=len(full.y),training_seconds=seconds,parameter_count=sum(p.numel() for p in model.parameters()),peak_gpu_memory_mb=torch.cuda.max_memory_allocated(device)/1024**2,checkpoint_sha256=file_sha(final_checkpoint),**{'validation_'+k:v for k,v in validation.items()})
    pd.DataFrame(history).to_csv(directory/'training_history.csv',index=False);json_write(done,result)
    print('FINISHED',name,'selected epochs',best_epoch,'validation RMSE',round(validation['rmse_fraction'],6),'seconds',round(seconds,1),flush=True)
    del model,optimizer,x,y;gc.collect();torch.cuda.empty_cache();return result


def freeze_configuration(config,output):
    manifest=definition(config)
    manifest['source_hashes']={name:file_sha(ROOT/'outputs'/name) for name in ['cx2_features_all.csv','cycle_quality.csv','inspection_manifest.csv']}
    manifest['source_hashes']['paper_pdf']=file_sha(ROOT.parent/'batteries-12-00115.pdf')
    manifest['scientific_code_hashes']={p.name:file_sha(p) for p in sorted(Path(__file__).parent.glob('*.py')) if p.name!='report.py'}
    manifest['environment']=dict(torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name(0),deterministic_algorithms=True,cublas_workspace=os.environ['CUBLAS_WORKSPACE_CONFIG'])
    path=output/'experiment_config.json'
    if path.exists() and json.loads(path.read_text())!=manifest:raise RuntimeError('Paper experiment configuration or source changed; use another output directory to preserve results')
    if not path.exists():json_write(path,manifest)


def train_all(config,output):
    freeze_configuration(config,output)
    if not (output/'sample_counts.csv').exists():prepare_all(config,output)
    for battery in BATTERIES:
        for variant in VARIANTS:
            for seed in SEEDS:train_one(output,battery,variant,seed,config)
    rows=[json.loads(p.read_text()) for p in sorted((output/'runs').glob('*/training_metrics.json'))]
    expected=len(BATTERIES)*len(VARIANTS)*len(SEEDS)
    if len(rows)!=expected:raise RuntimeError('Incomplete paper reproduction grid')
    freeze=dict(policy='Predeclared primary fixed_peak plus full_cc_peak sensitivity, all three seeds; no test-driven configuration selection',runs={r['run_id']:r['checkpoint_sha256'] for r in rows},prepared_data_hashes={str(p.relative_to(output)):file_sha(p) for p in (output/'datasets').glob('*/windows.npz')})
    path=output/'frozen_models.json'
    if path.exists() and json.loads(path.read_text())!=freeze:raise RuntimeError('Frozen paper checkpoints changed')
    if not path.exists():json_write(path,freeze)
    collect(output);print('FROZEN',expected,'final predictors before test evaluation.',flush=True)


def test_all(config,output):
    frozen=json.loads((output/'frozen_models.json').read_text());marker=output/'test_started.json'
    if marker.exists():
        if json.loads(marker.read_text())['frozen_models_sha256']!=file_sha(output/'frozen_models.json'):raise RuntimeError('Frozen models changed after test started')
    else:json_write(marker,dict(frozen_models_sha256=file_sha(output/'frozen_models.json'),policy='One forward pass per committed final predictor; no further fitting'))
    for path,h in frozen['prepared_data_hashes'].items():
        if file_sha(output/path)!=h:raise RuntimeError('Prepared windows changed after freezing')
    directory=output/'test_evaluations';directory.mkdir(exist_ok=True)
    for name,expected in frozen['runs'].items():
        done=directory/f'{name}.json'
        if done.exists():continue
        begun=directory/f'{name}.started.json'
        if begun.exists():raise RuntimeError('Incomplete prior test evaluation needs investigation: '+name)
        checkpoint_path=output/'checkpoints'/f'{name}_final.pt'
        if file_sha(checkpoint_path)!=expected:raise RuntimeError('Checkpoint modified after freezing')
        checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
        model=PaperLSTM(config).cuda();model.load_state_dict(checkpoint['state_dict']);model.eval()
        w=load_data(output,checkpoint['battery'],checkpoint['variant'])['test'];x=torch.from_numpy(w.x).cuda()
        with torch.inference_mode():model(torch.zeros((64,5,3),device='cuda'))
        json_write(begun,dict(run_id=name,checkpoint_sha256=expected))
        prediction,_,seconds=predict(model,x,'cuda')
        result=dict(run_id=name,battery=checkpoint['battery'],variant=checkpoint['variant'],seed=checkpoint['seed'],selected_epochs=checkpoint['epoch'],test_inference_seconds=seconds,**metrics(w.y,prediction))
        frame=w.metadata.copy();frame['run_id']=name;frame['actual_soh_fraction']=w.y;frame['predicted_soh_fraction']=prediction;frame['error_percentage_points']=(prediction-w.y)*100
        frame.to_csv(directory/f'{name}_predictions.csv',index=False);json_write(done,result)
        print('TEST',name,'RMSE',round(result['rmse_fraction'],6),flush=True)
        del model,x;gc.collect();torch.cuda.empty_cache()
    collect(output)


def collect(output):
    training=pd.DataFrame([json.loads(p.read_text()) for p in sorted((output/'runs').glob('*/training_metrics.json'))]);training.to_csv(output/'validation_results.csv',index=False)
    rows=[json.loads(p.read_text()) for p in sorted((output/'test_evaluations').glob('*.json')) if not p.name.endswith('.started.json')]
    tests=pd.DataFrame(rows);tests.to_csv(output/'test_results.csv',index=False) if rows else pd.DataFrame(columns=['run_id','battery','variant','seed','rmse_fraction']).to_csv(output/'test_results.csv',index=False)
    if rows:
        columns=['rmse_fraction','mae_fraction','mse_fraction_squared','rmse_percentage_points','mae_percentage_points','mape_percent','r2']
        summary=tests.groupby(['battery','variant'])[columns].agg(['mean','std']);summary.columns=['_'.join(c) for c in summary.columns];summary.reset_index().to_csv(output/'seed_summary.csv',index=False)
        pd.concat([pd.read_csv(p) for p in sorted((output/'test_evaluations').glob('*_predictions.csv'))],ignore_index=True).to_csv(output/'predictions.csv',index=False)
    histories=[pd.read_csv(p) for p in sorted((output/'runs').glob('*/training_history.csv'))]
    if histories:pd.concat(histories,ignore_index=True).to_csv(output/'training_history.csv',index=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=OUTPUT);parser.add_argument('--stage',choices=['prepare','train','test','report','all'],default='all');parser.add_argument('--smoke',action='store_true');args=parser.parse_args();config=Config();output=args.output;output.mkdir(parents=True,exist_ok=True)
    needs_preparation=not (output/'sample_counts.csv').exists()
    if args.stage=='prepare' and (output/'frozen_models.json').exists():
        raise RuntimeError('Prepared windows are frozen with evaluated models; use a new output directory to rebuild them')
    if needs_preparation or args.stage=='prepare':prepare_all(config,output)
    if args.stage=='prepare':return
    if args.smoke:
        import shutil
        smoke_output=output/'smoke';source=output/'datasets'/f'{BATTERIES[0]}_{VARIANTS[0]}'
        shutil.copytree(source,smoke_output/'datasets'/source.name,dirs_exist_ok=True)
        setup_runtime();train_one(smoke_output,BATTERIES[0],VARIANTS[0],42,replace(config,max_epochs=2,patience=2),smoke=True);return
    if args.stage in ['train','test','all']:setup_runtime()
    if args.stage in ['train','all']:train_all(config,output)
    if args.stage in ['test','all']:test_all(config,output)
    if args.stage in ['report','all']:
        collect(output)
        from .report import generate
        generate(output,config)
if __name__=='__main__':main()
