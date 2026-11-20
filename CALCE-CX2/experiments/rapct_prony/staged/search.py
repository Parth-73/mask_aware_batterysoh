"""Sequential SQLite-backed TPE search; outer labels never enter an objective."""
import json
from contextlib import contextmanager
from pathlib import Path
import os
import numpy as np
import pandas as pd
import optuna
from src.training.config import json_write
from .prepare import OUT
from .dataset import inner_batteries
from .runtime import identity
from .engine import fit


@contextmanager
def study_lock(path):
    file=open(path,'a+b')
    file.seek(0);file.write(b'0');file.flush();file.seek(0)
    try:
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(file.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(file,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield
    finally:
        file.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(file.fileno(),msvcrt.LK_UNLCK,1)
        else:
            import fcntl
            fcntl.flock(file,fcntl.LOCK_UN)
        file.close()


def optimize(name,outer,budget,base,suggest,parents=None,enqueue=None):
    """Last inner fold reports/prunes at equal progress using prior-fold histories."""
    directory=OUT/'optuna_studies';directory.mkdir(exist_ok=True)
    database=directory/f'{outer}_{name}.sqlite3'
    protocol={'name':name,'outer':outer,'base':base,'inner_validation_batteries':inner_batteries(outer,2),
              'objective':'mean per-inner-battery validation RMSE; only remaining six cells',
              'sampler':'TPESampler(seed=42)','pruner':'MedianPruner(startup=2,warmup=1)',
              'pruning':'last inner model compares battery-macro scores at identical completed epoch',
              'pilot_trials':budget}
    definition_path=directory/f'{outer}_{name}_protocol.json'
    if definition_path.exists():
        if json.loads(definition_path.read_text())!=protocol:
            raise ValueError('Study protocol changed; use a new study name')
    else:json_write(definition_path,protocol)
    with study_lock(database.with_suffix('.lock')):
        study=optuna.create_study(study_name=f'{outer}_{name}',storage='sqlite:///'+database.as_posix(),
            load_if_exists=True,direction='minimize',sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=2,n_warmup_steps=1))
        # The OS lock proves no other writer is live. Requeue abandoned configs; fit identity
        # excludes trial number so complete fits are reused and interrupted fits resume exactly.
        for old in study.get_trials(states=[optuna.trial.TrialState.RUNNING]):
            study.tell(old.number,state=optuna.trial.TrialState.FAIL)
            if old.params:study.enqueue_trial(old.params,skip_if_exists=False)
        if not study.trials and enqueue:
            for params in enqueue:study.enqueue_trial(params)
        def objective(trial):
            if base['module']=='stage3':
                from .physics_engine import fit as fit_candidate
            else:fit_candidate=fit
            parameters=suggest(trial)
            previous=[];scores=[];runs=[]
            for position,validation in enumerate(inner_batteries(outer,2)):
                config={**base,**parameters,'outer':outer,'validation':validation,'purpose':'search',
                        'seed':42,'search_candidate':identity(parameters)[:12]}
                if parents:config['parent']=str(parents[validation])
                def progress(epoch,score):
                    if position!=1:return
                    comparable=[h[epoch-1] for h in previous if len(h)>=epoch]
                    if len(comparable)!=len(previous):return
                    macro=float(np.mean(comparable+[score]))
                    trial.report(macro,epoch)
                    if trial.should_prune():raise optuna.TrialPruned()
                run,result=fit_candidate(config,progress=progress)
                history=pd.read_csv(run/'training.csv').validation_battery_macro_rmse.tolist()
                previous.append(history);scores.append(result['rmse_fraction']);runs.append(str(run))
                trial.set_user_attr('inner_run_paths',runs)
                trial.set_user_attr('inner_scores',scores)
            trial.set_user_attr('resolved_configuration',parameters)
            return float(np.mean(scores))
        finished=sum(t.state in [optuna.trial.TrialState.COMPLETE,optuna.trial.TrialState.PRUNED] for t in study.trials)
        if finished<budget:study.optimize(objective,n_trials=budget-finished)
        table=study.trials_dataframe()
        table.to_csv(directory/f'{outer}_{name}_trials.csv',index=False)
        best=study.best_trial
        selection={'study':study.study_name,'best_trial':best.number,'best_inner_battery_macro_rmse':best.value,
                   'parameters':best.user_attrs['resolved_configuration'],'inner_run_paths':best.user_attrs['inner_run_paths'],
                   'trial_count':len(study.trials),'pilot':True,'outer_evaluated':False}
        json_write(OUT/'configs'/f'{outer}_{name}_selection.json',selection)
        return selection
