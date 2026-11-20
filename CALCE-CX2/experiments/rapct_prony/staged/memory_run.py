"""Synthetic mode-count and timescale ablations; no CALCE recovery result is produced."""
import json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.training.config import json_write
from .prepare import OUT
from .runtime import seed_all,atomic_torch
from .prony import NeuralPronyMemory
from .hybrid import RecoveryAwareHybrid
from .diagnostics import save


def main():
    if (OUT/'configs/gate_memory_synthetic.json').exists():
        assert json.loads((OUT/'configs/gate_memory_synthetic.json').read_text())['passed']
        print('Reusing completed multi-mode synthetic experiments.',flush=True);return
    assert json.loads((OUT/'configs/gate_single_memory.json').read_text())['passed']
    directory=OUT/'latent_states/synthetic';directory.mkdir(parents=True,exist_ok=True)
    rows=[];predictions=[];parameters=[]
    train_t=torch.linspace(2.,900.,128);validation_t=torch.linspace(5.,1000.,151)
    truth=lambda t:.03*(.65*torch.exp(-t/30.)+.35*torch.exp(-t/300.))
    for modes in [1,2,4,8]:
        for learnable in [False,True]:
            seed_all(42);model=NeuralPronyMemory(modes,tau_bounds=(5.,1200.),learnable=learnable)
            optimized=[model.amplitude_raw]+([model.tau_raw] if learnable else [])
            optimizer=torch.optim.Adam(optimized,lr=.05)
            for epoch in range(400):
                optimizer.zero_grad()
                output=model.step(torch.zeros(len(train_t),64),train_t,torch.ones(len(train_t),dtype=torch.bool),state=torch.full((len(train_t),modes),.03))
                loss=(output['penalty']-truth(train_t)).square().mean();loss.backward();optimizer.step()
            with torch.no_grad():
                out=model.step(torch.zeros(len(validation_t),64),validation_t,torch.ones(len(validation_t),dtype=torch.bool),state=torch.full((len(validation_t),modes),.03))
            mse=float((out['penalty']-truth(validation_t)).square().mean())
            record={'domain':'synthetic_only','modes':modes,'learnable':learnable,'synthetic_penalty_mse':mse,
                    'seed':42,'epochs':400,'time_range_source':'declared artificial signal; not CALCE rest distribution',
                    'experimental_capacity_recovery_evidence':False}
            rows.append(record)
            parameters.append({**record,'timescales_s':model.timescales().detach().tolist(),'normalized_amplitudes':model.amplitudes().detach().tolist()})
            for t,y,p in zip(validation_t,truth(validation_t),out['penalty']):
                predictions.append({'domain':'synthetic_only','modes':modes,'learnable':learnable,'elapsed_s':float(t),
                    'actual_synthetic_penalty':float(y),'predicted_synthetic_penalty':float(p)})
            atomic_torch(directory/f'prony_M{modes}_learnable{learnable}.pt',{'state_dict':model.state_dict(),'configuration':record})
    pd.DataFrame(rows).to_csv(directory/'ablation.csv',index=False)
    table=pd.DataFrame(predictions);table.to_csv(directory/'predictions.csv',index=False)
    json_write(directory/'learned_parameters.json',parameters)
    seed_all(42);hybrid=RecoveryAwareHybrid().eval()
    with torch.no_grad():
        h=hybrid(torch.randn(8,5,5),torch.randn(8,5,3,176),torch.tensor([0.,30.,90.,30.,125.,30.,90.,30.]),
                 torch.tensor([False,False,True,False,True,False,True,False]))
    np.savez(directory/'full_hybrid_interface_trace.npz',**{key:value.cpu().numpy() for key,value in h.items() if isinstance(value,torch.Tensor)})
    atomic_torch(directory/'full_hybrid_interface.pt',{'state_dict':hybrid.state_dict(),'domain':'synthetic_interface_only',
                 'trained_on_CALCE':False,'config':{'modes':4,'learnable':True,'tau_bounds':[5.,1200.]}})
    fig,ax=plt.subplots(figsize=(8,4))
    ax.plot(validation_t,truth(validation_t),'k',label='Known artificial two-mode signal')
    for (m,learn),group in table.groupby(['modes','learnable']):
        ax.plot(group.elapsed_s,group.predicted_synthetic_penalty,label=f'M={m}, learn={learn}',linewidth=.8)
    ax.set(xlabel='Artificial rest duration (s)',ylabel='Synthetic capacity-penalty fraction',title='Synthetic memory fit; no CALCE recovery validation');ax.legend(fontsize=6)
    save(fig,'synthetic_prony_ablation')
    fig,ax=plt.subplots(figsize=(7,4))
    for parameter in parameters:
        ax.scatter(parameter['timescales_s'],[str(parameter['modes'])+' learn='+str(parameter['learnable'])]*parameter['modes'],s=20)
    ax.set(xscale='log',xlabel='Artificial fitted/fixed timescale (s)',ylabel='Mode configuration',title='Effective synthetic timescales; not electrochemical mechanisms')
    save(fig,'synthetic_timescales')
    json_write(OUT/'configs/gate_memory_synthetic.json',{'passed':True,'domain':'synthetic_only','experimental_A5_A8':'unavailable',
        'mode_ablation_runs':8,'single_mode_baseline_fitted':True,'full_hybrid_interface_tested':True,
        'true_synthetic_timescales_s':[30.,300.],'timescale_bounds_source':'declared synthetic protocol, never validation/test CALCE statistics',
        'interpretation':'Mathematical and numerical validation only; no experimental recovery or novelty claim.'})
    print(pd.DataFrame(rows).to_string(index=False),flush=True)


if __name__=='__main__':main()
