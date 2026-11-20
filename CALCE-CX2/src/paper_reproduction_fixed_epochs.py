"""Pre-test fixed-epoch sensitivity for the underspecified paper training budget.

python -m src.paper_reproduction_fixed_epochs --stage train
python -m src.paper_reproduction_fixed_epochs --stage test
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,gc,json,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .paper_reproduction.config import ROOT,OUTPUT,BATTERIES,SEEDS,Config,file_sha,json_write
from .paper_reproduction.model import PaperLSTM
from .paper_reproduction.run import load_data
from .training.train_lstm import set_seed,setup_runtime
from .training.evaluate import metrics,predict,synchronize


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--parent',type=Path,default=OUTPUT);parser.add_argument('--stage',choices=['train','test'],required=True);args=parser.parse_args()
    parent=args.parent;output=parent/'fixed_epoch';output.mkdir(parents=True,exist_ok=True);config=Config();setup_runtime()
    protocol=dict(epochs=200,optimizer='Adam',learning_rate=.001,weight_decay=0.,dropout=0.,scheduler='none; constant LR',feature_variant='fixed_peak',seeds=SEEDS,batteries=BATTERIES,
        rationale='Epoch budget unreported. Additional fixed-budget comparison registered before any new paper-reproduction test evaluation; do not choose a budget based on test error.',
        training='Entire first 50% with final training-only trends/MinMax scaler; no validation-dependent stopping or refit',
        source_script_sha256=file_sha(Path(__file__)),model_source_sha256=file_sha(ROOT/'src/paper_reproduction/model.py'),
        prepared_hashes={b:file_sha(parent/'datasets'/f'{b}_fixed_peak'/'windows.npz') for b in BATTERIES})
    path=output/'protocol.json'
    if path.exists():
        if json.loads(path.read_text())!=protocol:raise RuntimeError('Fixed-epoch protocol changed; use a separate experiment')
    else:
        if (parent/'test_started.json').exists():raise RuntimeError('Must register fixed-budget comparison before parent test exposure')
        json_write(path,protocol)
    if args.stage=='train':
        if (output/'test_started.json').exists():raise RuntimeError('No further fitting after test exposure')
        for battery in BATTERIES:
            windows=load_data(parent,battery,'fixed_peak')['final_train']
            for seed in SEEDS:
                ident=f'{battery}_fixed200_seed{seed}';directory=output/'runs'/ident;directory.mkdir(parents=True,exist_ok=True)
                if (directory/'metrics.json').exists():continue
                set_seed(seed);model=PaperLSTM(config).cuda();optimizer=torch.optim.Adam(model.parameters(),lr=.001);x=torch.from_numpy(windows.x).cuda();y=torch.from_numpy(windows.y).cuda();loss_fn=torch.nn.MSELoss();history=[]
                synchronize('cuda');start=time.perf_counter()
                for epoch in range(1,201):
                    model.train();permutation=torch.randperm(len(y),device='cuda');losses=[]
                    for j in range(0,len(y),64):
                        indices=permutation[j:j+64];optimizer.zero_grad(set_to_none=True);p=model(x[indices]);loss=loss_fn(p,y[indices]);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();losses.append(loss.detach()*len(indices))
                    history.append(dict(run_id=ident,epoch=epoch,train_mse=float(torch.stack(losses).sum().item()/len(y)),learning_rate=.001))
                synchronize('cuda');seconds=time.perf_counter()-start
                checkpoint=directory/'model.pt';torch.save(dict(state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},battery=battery,seed=seed,epochs=200,config=config.__dict__),checkpoint)
                json_write(directory/'metrics.json',dict(run_id=ident,battery=battery,seed=seed,epochs=200,train_windows=len(y),training_seconds=seconds,checkpoint_sha256=file_sha(checkpoint)))
                pd.DataFrame(history).to_csv(directory/'history.csv',index=False);print('FIXED200 FIT',ident,round(seconds,2),'s',flush=True)
                del model,optimizer,x,y;gc.collect();torch.cuda.empty_cache()
        rows=[json.loads(p.read_text()) for p in sorted((output/'runs').glob('*/metrics.json'))];assert len(rows)==12
        pd.DataFrame(rows).to_csv(output/'training_results.csv',index=False);json_write(output/'frozen_models.json',{r['run_id']:r['checkpoint_sha256'] for r in rows});return
    frozen=json.loads((output/'frozen_models.json').read_text());marker=output/'test_started.json'
    if not marker.exists():json_write(marker,dict(protocol_sha256=file_sha(path),frozen_models_sha256=file_sha(output/'frozen_models.json')))
    else:
        m=json.loads(marker.read_text());assert m['protocol_sha256']==file_sha(path) and m['frozen_models_sha256']==file_sha(output/'frozen_models.json')
    directory=output/'test_evaluations';directory.mkdir(exist_ok=True)
    for ident,h in frozen.items():
        done=directory/f'{ident}.json'
        if done.exists():continue
        started=directory/f'{ident}.started.json'
        if started.exists():raise RuntimeError('Incomplete fixed-epoch evaluation: '+ident)
        checkpoint_path=output/'runs'/ident/'model.pt';assert file_sha(checkpoint_path)==h
        checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False);battery=checkpoint['battery'];w=load_data(parent,battery,'fixed_peak')['test']
        assert file_sha(parent/'datasets'/f'{battery}_fixed_peak'/'windows.npz')==protocol['prepared_hashes'][battery]
        model=PaperLSTM(config).cuda();model.load_state_dict(checkpoint['state_dict']);model.eval();x=torch.from_numpy(w.x).cuda()
        with torch.inference_mode():model(torch.zeros((64,5,3),device='cuda'))
        json_write(started,dict(run_id=ident,checkpoint_sha256=h));p,_,seconds=predict(model,x,'cuda')
        frame=w.metadata.copy();frame['run_id']=ident;frame['actual_soh_fraction']=w.y;frame['predicted_soh_fraction']=p;frame['error_percentage_points']=(p-w.y)*100;frame.to_csv(directory/f'{ident}_predictions.csv',index=False)
        json_write(done,dict(run_id=ident,battery=battery,seed=checkpoint['seed'],epochs=200,inference_seconds=seconds,**metrics(w.y,p)))
        print('FIXED200 TEST',ident,'RMSE',round(metrics(w.y,p)['rmse_fraction'],6),flush=True)
        del model,x;gc.collect();torch.cuda.empty_cache()
    rows=[json.loads(p.read_text()) for p in sorted(directory.glob('*.json')) if not p.name.endswith('.started.json')]
    results=pd.DataFrame(rows);results.to_csv(output/'test_results.csv',index=False)
    summary=results.groupby('battery')[['rmse_fraction','mae_fraction','mse_fraction_squared','mape_percent','r2']].agg(['mean','std']);summary.columns=['_'.join(c) for c in summary.columns];summary.reset_index().to_csv(output/'seed_summary.csv',index=False)
if __name__=='__main__':main()
