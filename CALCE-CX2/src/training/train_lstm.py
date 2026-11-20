"""Run staged LSTM experiments, freeze validation choices, then test exactly once.

python -m src.training.train_lstm --smoke
python -m src.training.train_lstm --stage train
python -m src.training.train_lstm --stage test
python -m src.training.train_lstm --stage report
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,gc,hashlib,json,random,time
from dataclasses import asdict,replace
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .config import ROOT,OUTPUT,SPLITS,FEATURES,ARCHITECTURES,SEQUENCES,SEEDS,TrainingConfig,experiment_definition,json_write,file_sha
from .dataset import WindowSet,prepare
from .evaluate import metrics,predict,synchronize
from ..models.lstm_baselines import LSTMBaseline,parameter_count


def set_seed(seed,deterministic=True):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=deterministic
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(deterministic)


def setup_runtime():
    torch.set_num_threads(2)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:pass
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required for these experiments; CUDA-enabled PyTorch/GPU is unavailable')


def run_id(architecture,features,length,seed):return f'{architecture}_{features}_L{length}_seed{seed}'


def load_prepared(output):
    arrays=np.load(output/'prepared_windows.npz')
    return {(split,length):WindowSet(arrays[f'{split}_{length}_x'],arrays[f'{split}_{length}_y'],pd.read_csv(output/f'windows_{split}_{length}.csv')) for split in SPLITS for length in SEQUENCES}


def prepared_for(prepared,split,length,feature_set):return prepared[(split,length)].subset(FEATURES[feature_set])


def save_prediction_table(path,window,predictions,run,split,attention=None):
    frame=window.metadata.copy();frame.insert(0,'run_id',run);frame['evaluation_split']=split
    frame['actual_soh_fraction']=window.y;frame['predicted_soh_fraction']=predictions;frame['error_fraction']=predictions-window.y
    frame['actual_soh_percent']=window.y*100;frame['predicted_soh_percent']=predictions*100;frame['error_percentage_points']=(predictions-window.y)*100
    frame.to_csv(path,index=False)
    if attention is not None:
        a=window.metadata[['battery','cycle_id','lifetime_cycle','input_cycle_ids','input_ordinals']].copy()
        a.insert(0,'run_id',run);a['split']=split;a['attention_weights']=list(map(json.dumps,attention.tolist()))
        a.to_csv(path.with_name(path.stem+'_attention.csv'),index=False)


def fit_one(prepared,output,config,architecture,feature_set,length,seed,smoke=False):
    ident=run_id(architecture,feature_set,length,seed)
    run_dir=output/'runs'/ident;run_dir.mkdir(parents=True,exist_ok=True)
    result_path=run_dir/'metrics.json'
    if result_path.exists():return json.loads(result_path.read_text())
    if (output/'test_evaluation_started.json').exists():raise RuntimeError('Test evaluation already started; no new training or redesign is allowed in this experiment directory')
    set_seed(seed,config.deterministic);device=torch.device(config.device)
    tr=prepared_for(prepared,'train',length,feature_set);va=prepared_for(prepared,'validation',length,feature_set)
    if smoke:
        tr=WindowSet(tr.x[:256],tr.y[:256],tr.metadata.iloc[:256]);va=WindowSet(va.x[:128],va.y[:128],va.metadata.iloc[:128])
    x=torch.from_numpy(tr.x).to(device);y=torch.from_numpy(tr.y).to(device);vx=torch.from_numpy(va.x).to(device)
    model=LSTMBaseline(len(FEATURES[feature_set]),architecture,config.hidden_size).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=config.scheduler_factor,patience=config.scheduler_patience,min_lr=config.scheduler_min_lr)
    loss_fn=torch.nn.MSELoss()
    torch.cuda.reset_peak_memory_stats(device)
    best=float('inf');best_epoch=0;stale=0;best_state=None;history=[]
    synchronize(device);start=time.perf_counter()
    for epoch in range(1,config.max_epochs+1):
        ep_start=time.perf_counter();model.train();permutation=torch.randperm(len(y),device=device);losses=[]
        for offset in range(0,len(y),config.batch_size):
            index=permutation[offset:offset+config.batch_size]
            optimizer.zero_grad(set_to_none=True);prediction=model(x[index]);loss=loss_fn(prediction,y[index])
            if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite training loss in {ident} epoch {epoch}')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),config.gradient_clip);optimizer.step()
            losses.append(loss.detach()*len(index))
        train_mse=float(torch.stack(losses).sum().item()/len(y))
        vp,_,_=predict(model,vx,device);validation=metrics(va.y,vp);rmse=validation['rmse_fraction']
        if not np.isfinite(rmse):raise FloatingPointError('Nonfinite validation RMSE')
        lr=optimizer.param_groups[0]['lr'];scheduler.step(rmse)
        if rmse<best-config.early_stop_min_delta:
            best=rmse;best_epoch=epoch;stale=0;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:stale+=1
        history.append(dict(run_id=ident,architecture=architecture,feature_set=feature_set,sequence_length=length,seed=seed,epoch=epoch,
                            train_mse_fraction_squared=train_mse,validation_mse_fraction_squared=validation['mse_fraction_squared'],validation_rmse_fraction=rmse,
                            learning_rate=lr,epoch_time_seconds=time.perf_counter()-ep_start,is_checkpoint_epoch=best_epoch==epoch))
        if epoch==1 or epoch%25==0:print(f'{ident} epoch={epoch} train_mse={train_mse:.6g} val_RMSE={rmse:.6g} best={best:.6g}',flush=True)
        if stale>=config.patience:break
    synchronize(device);training_seconds=time.perf_counter()-start
    model.load_state_dict(best_state);model.eval()
    # Warm up on synthetic inputs; test observations are never used for warmup.
    with torch.inference_mode():
        model(torch.zeros((min(64,len(va.y)),length,len(FEATURES[feature_set])),device=device))
    vp,attention,inference_seconds=predict(model,vx,device,timed=True)
    result=dict(run_id=ident,architecture=architecture,feature_set=feature_set,sequence_length=length,seed=seed,
                split='validation',best_epoch=best_epoch,epochs_trained=len(history),train_samples=len(y),validation_samples=len(va.y),
                training_time_seconds=training_seconds,inference_time_seconds=inference_seconds,inference_ms_per_sample=1000*inference_seconds/len(va.y),
                parameter_count=parameter_count(model),peak_gpu_memory_mb=torch.cuda.max_memory_allocated(device)/1024**2,
                peak_gpu_reserved_mb=torch.cuda.max_memory_reserved(device)/1024**2,checkpoint_path=str(output/'checkpoints'/f'{ident}.pt'),
                preprocessing_path=str(output/'feature_preprocessing.json'),**metrics(va.y,vp))
    checkpoint=dict(state_dict=best_state,run_id=ident,architecture=architecture,feature_set=feature_set,feature_names=FEATURES[feature_set],sequence_length=length,seed=seed,
                    hidden_size=config.hidden_size,best_epoch=best_epoch,hyperparameters=asdict(config),preprocessing=json.loads((output/'feature_preprocessing.json').read_text()),target_units='fraction',splits=SPLITS)
    (output/'checkpoints').mkdir(exist_ok=True);torch.save(checkpoint,output/'checkpoints'/f'{ident}.pt')
    pd.DataFrame(history).to_csv(run_dir/'training_history.csv',index=False)
    save_prediction_table(run_dir/'validation_predictions.csv',va,vp,ident,'validation',attention)
    json_write(result_path,result)
    print(f'FINISHED {ident}: val_RMSE={result["rmse_fraction"]:.6g}, epoch={best_epoch}/{len(history)}, {training_seconds:.1f}s',flush=True)
    del model,optimizer,scheduler,x,y,vx;gc.collect();torch.cuda.empty_cache()
    return result


def validation_table(output):
    rows=[json.loads(p.read_text()) for p in sorted((output/'runs').glob('*/metrics.json'))]
    return pd.DataFrame(rows)


def group_scores(rows,columns):
    return rows.groupby(columns,dropna=False).agg(validation_rmse_mean=('rmse_fraction','mean'),validation_rmse_std=('rmse_fraction','std'),seeds=('seed','nunique')).reset_index()


def train_experiments(output,config):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    definition=experiment_definition(config)
    definition['source_hashes']={name:file_sha(ROOT/'outputs'/name) for name in ['cx2_features_all.csv','cycle_quality.csv','rejected_cycles.csv']}
    definition['scientific_code_hashes']={str(p.relative_to(ROOT)):file_sha(p) for folder in ['models','training'] for p in sorted((ROOT/'src'/folder).glob('*.py')) if p.name!='report.py'}
    definition['environment']=dict(torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name(0),gpu_total_memory_mb=torch.cuda.get_device_properties(0).total_memory/1024**2,
        cublas_workspace=os.environ['CUBLAS_WORKSPACE_CONFIG'],tf32=False,cpu_threads=torch.get_num_threads(),nondeterminism='Deterministic PyTorch algorithms and cuDNN requested; reproducibility across software/hardware versions is not guaranteed.')
    existing=output/'experiment_config.json'
    if existing.exists():
        old=json.loads(existing.read_text())
        if old!=definition:raise RuntimeError('Experiment config/data/code differs from existing run. Preserve results and use a distinct output directory for changed scientific configurations.')
    else:json_write(existing,definition)
    if (output/'final_selection.json').exists():
        print('Choices already frozen. Reusing completed training.',flush=True);return
    if not (output/'prepared_windows.npz').exists():prepared=prepare(output,config.max_acquisition_gap_s)
    else:prepared=load_prepared(output)
    for features in FEATURES:
        for seed in SEEDS:fit_one(prepared,output,config,'B1',features,5,seed)
    values=validation_table(output)
    a=group_scores(values[(values.architecture=='B1')&(values.sequence_length==5)],['feature_set']).sort_values(['validation_rmse_mean','feature_set'])
    selected_features=a.iloc[0].feature_set
    json_write(output/'feature_selection.json',dict(policy='Mean validation RMSE across 3 seeds, B1/L5, common cohort',selected_features=selected_features,scores=a.to_dict('records')))
    print('VALIDATION SELECTED FEATURES:',selected_features,flush=True)
    for architecture in ARCHITECTURES:
        for length in SEQUENCES:
            for seed in SEEDS:fit_one(prepared,output,config,architecture,selected_features,length,seed)
    # Predictor-settings comparison is a separate pre-specified configuration.
    for seed in SEEDS:fit_one(prepared,output,config,'B2','F1',5,seed)
    values=validation_table(output);primary=values[values.feature_set==selected_features]
    b=group_scores(primary[primary.sequence_length==5],['architecture']).sort_values(['validation_rmse_mean','architecture'])
    architecture=b.iloc[0].architecture
    c=group_scores(primary[primary.architecture==architecture],['sequence_length']).sort_values(['validation_rmse_mean','sequence_length'])
    length=int(c.iloc[0].sequence_length)
    selected_ids=[run_id(architecture,selected_features,length,seed) for seed in SEEDS]
    literature_ids=[run_id('B2','F1',5,seed) for seed in SEEDS]
    selection=dict(feature_set=selected_features,architecture=architecture,sequence_length=length,seeds=SEEDS,selected_run_ids=selected_ids,literature_run_ids=literature_ids,
                   feature_scores=a.to_dict('records'),architecture_scores=b.to_dict('records'),sequence_scores=c.to_dict('records'),
                   policy='Staged validation-only feature then architecture-at-L5 then sequence selection; seed-mean RMSE, deterministic lexical ties; no best-seed selection',test_opened=False,
                   checkpoint_hashes={ident:file_sha(output/'checkpoints'/f'{ident}.pt') for ident in sorted(set(selected_ids+literature_ids))})
    json_write(output/'final_selection.json',selection)
    collect(output)
    print('FROZEN SELECTION:',architecture,selected_features,'length',length,flush=True)


def test_selected(output):
    output=Path(output);selection=json.loads((output/'final_selection.json').read_text());prepared=load_prepared(output)
    ident_list=sorted(set(selection['selected_run_ids']+selection['literature_run_ids']))
    marker=output/'test_evaluation_started.json'
    if not marker.exists():json_write(marker,dict(frozen_selection_sha256=file_sha(output/'final_selection.json'),run_ids=ident_list,policy='No training or model redesign after this marker; one prediction pass per selected checkpoint'))
    elif json.loads(marker.read_text())['frozen_selection_sha256']!=file_sha(output/'final_selection.json'):raise RuntimeError('Frozen selection changed after test exposure')
    directory=output/'test_evaluations';directory.mkdir(exist_ok=True)
    for ident in ident_list:
        record=directory/f'{ident}.json'
        if record.exists():continue
        started=directory/f'{ident}.started.json'
        if started.exists():raise RuntimeError(f'Test {ident} previously started without a committed result. Investigate interruption before any repeat evaluation.')
        checkpoint_path=output/'checkpoints'/f'{ident}.pt'
        if file_sha(checkpoint_path)!=selection['checkpoint_hashes'][ident]:raise RuntimeError('Selected checkpoint changed after selection')
        checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
        window=prepared_for(prepared,'test',checkpoint['sequence_length'],checkpoint['feature_set']);device=torch.device('cuda')
        model=LSTMBaseline(len(checkpoint['feature_names']),checkpoint['architecture'],checkpoint['hidden_size']).to(device);model.load_state_dict(checkpoint['state_dict'])
        set_seed(checkpoint['seed']);model.eval()
        with torch.inference_mode():model(torch.zeros((64,checkpoint['sequence_length'],len(checkpoint['feature_names'])),device=device))
        x=torch.from_numpy(window.x).to(device);torch.cuda.reset_peak_memory_stats(device)
        json_write(started,dict(run_id=ident,checkpoint_sha256=selection['checkpoint_hashes'][ident],test_battery='CX2_38'))
        p,attention,seconds=predict(model,x,device,timed=True)
        result=dict(run_id=ident,architecture=checkpoint['architecture'],feature_set=checkpoint['feature_set'],sequence_length=checkpoint['sequence_length'],seed=checkpoint['seed'],split='test',test_battery='CX2_38',
                    roles=';'.join([role for role,ids in [('selected',selection['selected_run_ids']),('literature_predictor',selection['literature_run_ids'])] if ident in ids]),
                    inference_time_seconds=seconds,inference_ms_per_sample=1000*seconds/len(window.y),parameter_count=parameter_count(model),peak_gpu_memory_mb=torch.cuda.max_memory_allocated(device)/1024**2,**metrics(window.y,p))
        save_prediction_table(directory/f'{ident}_predictions.csv',window,p,ident,'test',attention);json_write(record,result)
        print('TEST',ident,'RMSE',result['rmse_fraction'],flush=True)
        del model,x;gc.collect();torch.cuda.empty_cache()
    collect(output)


def collect(output):
    output=Path(output);validation=validation_table(output)
    validation.to_csv(output/'validation_results.csv',index=False)
    tests=[json.loads(p.read_text()) for p in sorted((output/'test_evaluations').glob('*.json')) if not p.name.endswith('.started.json')]
    test=pd.DataFrame(tests);test.to_csv(output/'test_results.csv',index=False) if tests else pd.DataFrame(columns=['run_id','split','rmse_fraction']).to_csv(output/'test_results.csv',index=False)
    pd.concat([validation,test],ignore_index=True).to_csv(output/'results.csv',index=False)
    histories=[pd.read_csv(p) for p in sorted((output/'runs').glob('*/training_history.csv'))]
    if histories:pd.concat(histories,ignore_index=True).to_csv(output/'training_history.csv',index=False)
    prediction_files=list((output/'runs').glob('*/validation_predictions.csv'))+list((output/'test_evaluations').glob('*_predictions.csv'))
    if prediction_files:pd.concat([pd.read_csv(p) for p in prediction_files],ignore_index=True).to_csv(output/'predictions.csv',index=False)
    attention_files=list((output/'runs').glob('*/*_attention.csv'))+list((output/'test_evaluations').glob('*_attention.csv'))
    if attention_files:pd.concat([pd.read_csv(p) for p in attention_files],ignore_index=True).to_csv(output/'attention_weights.csv',index=False)
    if not validation.empty:
        names=['mae_fraction','rmse_fraction','mse_fraction_squared','mape_percent','r2','training_time_seconds','inference_time_seconds','peak_gpu_memory_mb']
        aggregate=validation.groupby(['architecture','feature_set','sequence_length'])[names].agg(['mean','std'])
        aggregate.columns=['_'.join(c) for c in aggregate.columns];aggregate.reset_index().to_csv(output/'validation_seed_summary.csv',index=False)
    if tests:
        aggregate=test.groupby(['architecture','feature_set','sequence_length'])[['mae_fraction','rmse_fraction','mse_fraction_squared','mape_percent','r2']].agg(['mean','std'])
        aggregate.columns=['_'.join(c) for c in aggregate.columns];aggregate.reset_index().to_csv(output/'test_seed_summary.csv',index=False)


def smoke(output,config):
    smoke_output=Path(output)/'smoke';smoke_output.mkdir(parents=True,exist_ok=True)
    prepared=prepare(smoke_output,config.max_acquisition_gap_s)
    smoke_config=replace(config,max_epochs=2,patience=2)
    results=[fit_one(prepared,smoke_output,smoke_config,architecture,'F1',5,42,smoke=True) for architecture in ARCHITECTURES]
    json_write(smoke_output/'smoke_results.json',results)
    print('All four CUDA smoke tests passed; no test-model evaluations performed.',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=OUTPUT);parser.add_argument('--stage',choices=['all','train','test','report'],default='all');parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();config=TrainingConfig();setup_runtime()
    if args.smoke:smoke(args.output,config);return
    if args.stage in ['all','train']:train_experiments(args.output,config)
    if args.stage in ['all','test']:test_selected(args.output)
    if args.stage in ['all','report']:
        collect(args.output)
        from .report import generate
        generate(args.output)
if __name__=='__main__':main()
