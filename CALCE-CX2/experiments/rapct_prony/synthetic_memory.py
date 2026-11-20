"""Synthetic memory identification only; never merged with CALCE SOH experiments."""
import json
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from src.common import save_json
from .audit import OUT
from .memory import NeuralProny
from .train import seed_all


def run():
    rows=[]
    times=torch.linspace(0.,1000.,161)
    truth=.035*torch.exp(-times/30.)+.015*torch.exp(-times/300.)
    training=torch.arange(len(times))%4!=0
    base=OUT/'checkpoints/synthetic_memory';base.mkdir(exist_ok=True)
    for modes in [1,2,4,8]:
        for learnable in [False,True]:
            for seed in [42,123,2026]:
                seed_all(seed);layer=NeuralProny(modes,10.,1000.,learnable)
                initial_raw=nn.Parameter(torch.randn(modes)*.1)
                optimizer=torch.optim.Adam([p for p in layer.parameters() if p.requires_grad]+[initial_raw],lr=.02)
                config=dict(dataset='synthetic_biexponential_penalty_only',modes=modes,learnable_timescales=learnable,seed=seed,epochs=400,
                    tau_bounds_s=[10.,1000.],ground_truth_timescales_s=[30.,300.],ground_truth_amplitudes=[.035,.015],
                    claim='mathematical_memory_identification_not_CALCE_recovery_performance')
                directory=base/f'M{modes}_learn{int(learnable)}_seed{seed}';directory.mkdir(exist_ok=False)
                save_json(directory/'config.json',config)
                losses=[];best=float('inf');start=time.perf_counter()
                for epoch in range(400):
                    optimizer.zero_grad(set_to_none=True)
                    initial=.1*torch.sigmoid(initial_raw)
                    states=initial[None]*torch.exp(-times[:,None]/layer.timescales)
                    prediction=layer.penalty(states)
                    loss=(prediction[training]-truth[training]).square().mean()
                    if not torch.isfinite(loss):raise FloatingPointError('synthetic_memory_nonfinite')
                    # Save the actual parameters producing this loss, before the update.
                    if loss.item()<best:
                        best=loss.item();torch.save(dict(model=layer.state_dict(),initial_raw=initial_raw.detach(),epoch=epoch+1),directory/'best.pt')
                    loss.backward();optimizer.step();losses.append(loss.item())
                elapsed=time.perf_counter()-start
                checkpoint=torch.load(directory/'best.pt',weights_only=True);layer.load_state_dict(checkpoint['model'])
                with torch.no_grad():
                    states=(.1*torch.sigmoid(checkpoint['initial_raw']))[None]*torch.exp(-times[:,None]/layer.timescales)
                    prediction=layer.penalty(states)
                error=(prediction[~training]-truth[~training]).numpy()
                pd.DataFrame(dict(time_s=times.numpy(),true_synthetic_penalty=truth.numpy(),predicted_penalty=prediction.numpy(),training_point=training.numpy())).to_csv(directory/'predictions.csv',index=False)
                pd.DataFrame({'epoch':np.arange(1,401),'train_mse':losses}).to_csv(directory/'training.csv',index=False)
                save_json(directory/'parameters.json',dict(timescales_s=layer.timescales.detach().tolist(),weights=layer.weights.detach().tolist()))
                rows.append(dict(dataset=config['dataset'],experimental_CALCE=False,modes=modes,learnable_timescales=learnable,seed=seed,
                    heldout_penalty_rmse_fraction=float(np.sqrt(np.mean(error**2))),heldout_penalty_mae_fraction=float(np.mean(abs(error))),
                    training_time_s=elapsed,epochs=400,checkpoint=str(directory/'best.pt'),timescales_s=json.dumps(layer.timescales.detach().tolist())))
    result=pd.DataFrame(rows);result.to_csv(OUT/'metrics/prony_ablation.csv',index=False)
    print(result.groupby(['modes','learnable_timescales']).heldout_penalty_rmse_fraction.mean().to_string(),flush=True)

if __name__=='__main__':run()
