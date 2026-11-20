"""Execute the one-mode synthetic gate before implementing multiple memory modes."""
import json
import numpy as np
import torch
from src.training.config import json_write
from .prepare import OUT
from .runtime import seed_all,atomic_torch
from .single_memory import SingleRelaxation


def main():
    if (OUT/'configs/gate_single_memory.json').exists():
        assert json.loads((OUT/'configs/gate_single_memory.json').read_text())['passed']
        print('Reusing completed single-mode synthetic experiment.',flush=True);return
    feasibility=json.loads((OUT/'rest_events/feasibility.json').read_text())
    assert not feasibility['independent_long_rest_recovery_evaluation']
    seed_all(42);model=SingleRelaxation(tau_bounds=(5.,1000.),tau_initial=50.)
    optimizer=torch.optim.Adam(model.parameters(),lr=.05)
    dt=torch.linspace(5,300,64);target=.03*torch.exp(-dt/80.)
    context=torch.zeros(64,64);initial=torch.full((64,1),.03)
    for _ in range(200):
        optimizer.zero_grad();out=model.step(context,dt,torch.ones(64,dtype=torch.bool),state=initial)
        loss=(out['penalty']-target).square().mean();loss.backward();optimizer.step()
    validation_dt=torch.linspace(10,350,71)
    with torch.no_grad():
        pred=model.step(torch.zeros(71,64),validation_dt,torch.ones(71,dtype=torch.bool),state=torch.full((71,1),.03))['penalty']
    score=float((pred-.03*torch.exp(-validation_dt/80.)).square().mean())
    directory=OUT/'latent_states/synthetic';directory.mkdir(parents=True,exist_ok=True)
    record={'domain':'synthetic_only','true_tau_s':80.,'fitted_tau_s':float(model.timescales().detach()),
            'validation_mse':score,'epochs':200,'seed':42,'calibrated_battery_parameter':False,
            'experimental_recovery_evidence':False}
    json_write(directory/'single_mode_fit.json',record)
    atomic_torch(directory/'single_mode.pt',{'state_dict':model.state_dict(),'config':record})
    assert score<1e-8
    json_write(OUT/'configs/gate_single_memory.json',{'passed':True,**record})
    print(json.dumps(record),flush=True)


if __name__=='__main__':main()
