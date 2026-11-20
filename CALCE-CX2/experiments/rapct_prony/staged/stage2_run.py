"""CNN gate required before fitting any temporal model; outer scores remain unopened."""
import argparse
import json
from pathlib import Path
from src.training.config import json_write
from .prepare import OUT
from .engine import fit
from .search import optimize
from .dataset import inner_batteries


def full_transformer_search_space(trial):
    width=trial.suggest_categorical('d_model',[32,64,128])
    heads=trial.suggest_categorical('heads',[2,4,8])
    assert width%heads==0
    return {'d_model':width,'heads':heads,'layers':trial.suggest_int('layers',1,3),
        'ffn':trial.suggest_categorical('ffn',[128,256]),'dropout':trial.suggest_float('dropout',.05,.30),
        'length':trial.suggest_categorical('length',[5,10,20]),'lr':trial.suggest_float('lr',1e-4,1e-3,log=True),
        'weight_decay':trial.suggest_float('weight_decay',1e-6,1e-2,log=True),
        'batch_size':trial.suggest_categorical('batch_size',[16,32,64])}


def main():
    p=argparse.ArgumentParser();p.add_argument('--outer',default='CX2_38');p.add_argument('--epochs',type=int,default=15)
    p.add_argument('--trials',type=int,default=2);p.add_argument('--search-epochs',type=int,default=3)
    a=p.parse_args()
    gate=json.loads((OUT/'configs'/f'{a.outer}_gate_stage1.json').read_text());assert gate['passed']
    selected=gate['selected_cnn'];validations=inner_batteries(a.outer,2)
    parents={validations[0]:Path(next(r['path'] for r in gate['runs'] if r['stage']=='A2' and r['seed']==42))/'best.pt'}
    initial=json.loads((parents[validations[0]].parent/'configuration.json').read_text())['parameters']
    # The second inner validation battery must NEVER occur in its own encoder pretraining.
    pretrain={**initial,'validation':validations[1],'purpose':'inner_encoder_pretraining'}
    directory,_=fit(pretrain);parents[validations[1]]=directory/'best.pt'
    base={'module':'stage2','stage':'A3','pathway':'fusion','length':5,'matched20':False,
          'epochs':a.search_epochs,'batch_size':32,'lr':5e-4,'weight_decay':1e-3,'patience':20,
          'd_model':64,'heads':4,'layers':2,'ffn':128,'dropout':.1,'status':'reduced_budget_nested_pilot'}
    def suggest(trial):return {'freeze_cnn':trial.suggest_categorical('freeze_cnn',[True,False])}
    selection=optimize('transformer',a.outer,a.trials,base,suggest,parents=parents,
                       enqueue=[{'freeze_cnn':True},{'freeze_cnn':False}])
    runs=[]
    # Evaluate both strategies at the same budget and all seeds; select only using validation.
    for strategy in [True,False]:
        for seed in [42,123,2026]:
            parent=Path(next(r['path'] for r in gate['runs'] if r['stage']=='A2' and r['seed']==seed))/'best.pt'
            config={**base,'freeze_cnn':strategy,'epochs':a.epochs,'seed':seed,'outer':a.outer,
                    'validation':validations[0],'parent':str(parent),'purpose':'final'}
            directory,result=fit(config)
            runs.append({'path':str(directory),'configuration_id':directory.name,'stage':'A3','seed':seed,
                         'validation':validations[0],'freeze_cnn':strategy,'validation_rmse_fraction':result['rmse_fraction']})
    # Equal-weight two-cell validation for final strategy choice, at equal seed and epoch
    # budgets. Each inner encoder is pretrained with that validation cell excluded.
    for seed in [42,123,2026]:
        first_parent=Path(next(r['path'] for r in gate['runs'] if r['stage']=='A2' and r['seed']==seed))/'best.pt'
        cnn_config=json.loads((first_parent.parent/'configuration.json').read_text())['parameters']
        cnn_config={**cnn_config,'validation':validations[1],'purpose':'inner_encoder_pretraining'}
        cnn_run,_=fit(cnn_config)
        for strategy in [True,False]:
            config={**base,'freeze_cnn':strategy,'epochs':a.epochs,'seed':seed,'outer':a.outer,
                    'validation':validations[1],'parent':str(cnn_run/'best.pt'),'purpose':'inner_strategy_validation'}
            directory,result=fit(config)
            runs.append({'path':str(directory),'configuration_id':directory.name,'stage':'A3','seed':seed,
                         'validation':validations[1],'freeze_cnn':strategy,'validation_rmse_fraction':result['rmse_fraction']})
    scores={str(strategy):sum(r['validation_rmse_fraction'] for r in runs if r['freeze_cnn']==strategy)/6 for strategy in [True,False]}
    strategy=min([True,False],key=lambda value:(scores[str(value)],not value))
    result={'passed':True,'stage':'2','outer':a.outer,'runs':runs,'selected_freeze_cnn':strategy,
            'strategy_validation_seed_means':scores,'pilot_optuna_selection':selection,'outer_evaluated':False,
            'selection_policy':'Equal battery-macro mean over two inner cells and three seeds selects strategy; outer scores unopened',
            'search_policy':'Bounded pilot searches strategy only; full search-space helper is provided but unexecuted'}
    json_write(OUT/'configs'/f'{a.outer}_gate_stage2.json',result)
    (OUT/'reports/STAGE2_REPORT.md').write_text('# Stage 2 validation pilot completed\n\n```json\n'+json.dumps(result,indent=2)+'\n```\n\nBoth strategies use identical targets and training-only scalers. All attention is causal within the available charge window. Saved attention arrays describe model behavior; they do not identify causal physical features. No outer scores select a strategy.\n')
    print('STAGE 2 GATE PASSED',flush=True)


if __name__=='__main__':main()
