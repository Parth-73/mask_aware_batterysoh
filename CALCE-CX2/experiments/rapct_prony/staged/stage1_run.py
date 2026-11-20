"""Execute the CNN gate before any Transformer implementation is introduced."""
import argparse
import json
from pathlib import Path
from src.training.config import json_write
from .prepare import OUT
from .search import optimize
from .engine import fit
from .dataset import inner_batteries


def main():
    p=argparse.ArgumentParser();p.add_argument('--outer',default='CX2_38');p.add_argument('--epochs',type=int,default=15)
    p.add_argument('--trials',type=int,default=2);p.add_argument('--search-epochs',type=int,default=3)
    a=p.parse_args()
    protocol={'outer':a.outer,'stage1_epochs':a.epochs,'stage2_epochs':15,'stage3_epochs':35,'seeds':[42,123,2026],
        'pilot':True,'requested_full_outer_folds':7,'executed_outer_folds':1,
        'search_spaces':{'cnn_channels':[16,32,64],'cnn_embedding':[32,64,128],
            'transformer_d_model':[32,64,128],'transformer_heads':[2,4,8],'transformer_layers':[1,2,3],
            'transformer_dropout':[.05,.30],'ffn':[128,256],'lengths':[5,10,20],
            'physics_weight':[0.,1e-5,1e-1],'memory_weight':[0.,1e-5,1e-1],
            'prony_modes':[1,2,4,8],'lr':[1e-4,1e-3],'weight_decay':[1e-6,1e-2],'batch':[16,32,64]},
        'bounded_search':'sequential component subsets; two startup trials each in this resource-limited pilot',
        'recovery':'experimental A6-A8 withheld by measured-rest feasibility; synthetic implementation/testing only',
        'outer_policy':'no new outer predictions until all stage configurations are frozen'}
    frozen=OUT/'configs'/f'{a.outer}_preregistered_protocol.json'
    if frozen.exists():assert json.loads(frozen.read_text())==protocol
    else:json_write(frozen,protocol)
    base={'module':'stage1','stage':'A2','pathway':'fusion','length':5,'matched20':False,'epochs':a.search_epochs,
          'batch_size':32,'accumulation':1,'amp':False,'lr':5e-4,'weight_decay':1e-3,'patience':20,
          'status':'reduced_budget_nested_pilot'}
    def suggest(trial):
        return {'channels':trial.suggest_categorical('channels',[16,32,64]),
                'embedding':trial.suggest_categorical('embedding',[32,64,128])}
    selection=optimize('cnn',a.outer,a.trials,base,suggest,enqueue=[{'channels':32,'embedding':64},{'channels':16,'embedding':64}])
    selected=selection['parameters'];runs=[]
    for stage,pathway in [('A0','mlp'),('A1','cnn'),('A2','fusion')]:
        for seed in [42,123,2026]:
            config={**base,**selected,'stage':stage,'pathway':pathway,'epochs':a.epochs,'seed':seed,
                    'outer':a.outer,'validation':inner_batteries(a.outer,2)[0],'purpose':'final'}
            run,result=fit(config);runs.append({'path':str(run),'configuration_id':run.name,'stage':stage,'seed':seed,
                                              'validation_rmse_fraction':result['rmse_fraction']})
    gate={'passed':True,'stage':'1','outer':a.outer,'selected_cnn':selected,'runs':runs,'outer_evaluated':False,
          'status':'reduced_budget_pilot; one outer fold, all three final seeds; two inner validation cells for search'}
    json_write(OUT/'configs'/f'{a.outer}_gate_stage1.json',gate)
    (OUT/'reports/STAGE1_REPORT.md').write_text('# Stage 1 completed: validation-only pilot\n\n```json\n'+json.dumps(gate,indent=2)+'\n```\n\nNo outer results were inspected or used. Curves are measured Q, dQ/dV and I on the voltage axis, not arbitrary engineered-feature positions. All pathways share target cycle IDs and train-only scalers. Full 150-epoch and seven-fold experiments remain unexecuted.\n')
    print('STAGE 1 GATE PASSED',flush=True)


if __name__=='__main__':main()
