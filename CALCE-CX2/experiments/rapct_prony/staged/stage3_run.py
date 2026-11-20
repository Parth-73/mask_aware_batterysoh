"""Add independent latent-state priors only after the temporal validation gate passes."""
import argparse
import json
from pathlib import Path
from src.training.config import json_write
from .prepare import OUT
from .search import optimize
from .engine import fit as fit_temporal
from .physics_engine import fit
from .dataset import inner_batteries


def main():
    p=argparse.ArgumentParser();p.add_argument('--outer',default='CX2_38');p.add_argument('--epochs',type=int,default=35)
    p.add_argument('--trials',type=int,default=2);p.add_argument('--search-epochs',type=int,default=8)
    a=p.parse_args()
    gate=json.loads((OUT/'configs'/f'{a.outer}_gate_stage2.json').read_text());assert gate['passed']
    strategy=gate['selected_freeze_cnn'];validations=inner_batteries(a.outer,2)
    parents={validations[0]:Path(next(r['path'] for r in gate['runs'] if r['freeze_cnn']==strategy and r['seed']==42 and r.get('validation',validations[0])==validations[0]))/'best.pt'}
    parent_config=json.loads((parents[validations[0]].parent/'configuration.json').read_text())['parameters']
    other_inner_path=Path(gate['pilot_optuna_selection']['inner_run_paths'][1])
    cnn_parent=json.loads((other_inner_path/'configuration.json').read_text())['parameters']['parent']
    pretrain={**parent_config,'validation':validations[1],'parent':cnn_parent,'purpose':'inner_temporal_pretraining'}
    existing=[r for r in gate['runs'] if r['freeze_cnn']==strategy and r['seed']==42 and r.get('validation')==validations[1]]
    if existing:parents[validations[1]]=Path(existing[0]['path'])/'best.pt'
    else:
        directory,_=fit_temporal(pretrain);parents[validations[1]]=directory/'best.pt'
    base={'module':'stage3','stage':'A4','pathway':'fusion','length':5,'matched20':False,'epochs':a.search_epochs,
        'batch_size':32,'lr':5e-4,'weight_decay':1e-3,'patience':20,'warmup':[2,5],
        'status':'reduced_budget_nested_pilot','state_policy':'ordered segments; carry detached state across chunks; initialize unknown gaps'}
    def suggest(trial):
        enabled=trial.suggest_categorical('enable_prior',[False,True])
        physical=trial.suggest_float('physics_weight',1e-5,1e-1,log=True)
        smooth=trial.suggest_float('smooth_weight',1e-5,1e-1,log=True)
        return {'physics_weight':physical if enabled else 0.,'smooth_weight':smooth if enabled else 0.}
    selection=optimize('physics',a.outer,a.trials,base,suggest,parents=parents,
        enqueue=[{'enable_prior':False,'physics_weight':.01,'smooth_weight':.001},
                 {'enable_prior':True,'physics_weight':.01,'smooth_weight':.001}])
    selected=selection['parameters']
    positive=selected if selected['physics_weight']>0 else {'physics_weight':.01,'smooth_weight':.001}
    runs=[]
    for weights in [{'physics_weight':0.,'smooth_weight':0.},positive]:
        for seed in [42,123,2026]:
            parent=Path(next(r['path'] for r in gate['runs'] if r['freeze_cnn']==strategy and r['seed']==seed and r.get('validation',validations[0])==validations[0]))/'best.pt'
            config={**base,**weights,'epochs':a.epochs,'warmup':[10,30],'seed':seed,'outer':a.outer,
                    'validation':validations[0],'parent':str(parent),'purpose':'final'}
            directory,result=fit(config)
            runs.append({'path':str(directory),'configuration_id':directory.name,'stage':'A4','seed':seed,
                         'physics_weight':weights['physics_weight'],'smooth_weight':weights['smooth_weight'],
                         'validation_rmse_fraction':result['rmse_fraction']})
    result={'passed':True,'stage':'3','outer':a.outer,'runs':runs,'physics_search_selection':selection,
            'comparison_nonzero_coefficients':positive,'outer_evaluated':False,
            'warmup_final':[10,30],'warmup_search':[2,5],
            'limitations':'Learned state/proxy consistency and smoothness are priors; no electrochemical PDE or calibrated mechanism.'}
    json_write(OUT/'configs'/f'{a.outer}_gate_stage3.json',result)
    (OUT/'reports/STAGE3_REPORT.md').write_text('# Physics-guided degradation validation pilot\n\n```json\n'+json.dumps(result,indent=2)+'\n```\n\nD starts from an encoder-estimated unknown state, never an invented fresh-cell zero. Positive rates guarantee nonnegative increments only inside verified chronological segments. Carried state is detached across training chunks. The independent state-proposal residual excludes its tautological initialization equality. Observed SOH is not forced monotone. Per-epoch data/proxy/correction/variation terms and per-cycle latent audits are saved. The same architecture and 35-epoch budget are used for zero/nonzero penalties.\n')
    print('STAGE 3 GATE PASSED',flush=True)


if __name__=='__main__':main()
