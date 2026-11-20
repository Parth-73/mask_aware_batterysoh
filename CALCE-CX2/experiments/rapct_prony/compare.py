"""Recompute all comparisons on identical prediction-cutoff/target cycle IDs."""
import json
import numpy as np
import pandas as pd
from .audit import OUT
from .metrics import metrics


def compare():
    results=[]
    runs=[]
    for p in (OUT/'checkpoints').glob('*/metrics.csv'):
        config=json.loads((p.parent/'config.json').read_text())
        if config['label'].startswith(('optuna','nested_CNN_refit')):continue
        for split in ['validation','outer']:
            pred=p.parent/f'predictions_{split}.csv'
            if pred.exists():runs.append((p.parent.name,config,split,pd.read_csv(pred)))
    for split in ['validation','outer']:
        selected=[r for r in runs if r[2]==split]
        if not selected:continue
        common=set.intersection(*[set(t.cycle_id) for _,_,_,t in selected])
        if not common:raise ValueError('empty_matched_comparison')
        truth=None
        for name,cfg,_,table in selected:
            table=table[table.cycle_id.isin(common)].sort_values('cycle_id').reset_index(drop=True)
            if truth is None:truth=table[['cycle_id','prediction_cutoff','soh']]
            elif not np.array_equal(truth.cycle_id,table.cycle_id) or not np.array_equal(truth.prediction_cutoff,table.prediction_cutoff) or not np.allclose(truth.soh,table.soh,atol=1e-12,rtol=0):raise ValueError('matched_target_or_cutoff_mismatch')
            for battery,g in table.groupby('battery'):
                row=metrics(g.soh,g.predicted_soh)
                row.update(run=name,stage=table.stage.iloc[0],split=split,battery=battery,seed=cfg['seed'],status=cfg['label'],
                    completed_epochs=pd.read_csv(OUT/'checkpoints'/name/'training.csv').epoch.max(),cohort='intersection_of_all_completed_primary_runs')
                results.append(row)
    result=pd.DataFrame(results)
    result.to_csv(OUT/'metrics/matched_stage_comparison.csv',index=False)
    result.to_csv(OUT/'metrics/stage_comparison.csv',index=False)
    print(result[result.split.eq('validation')][['run','rmse_pp','mae_pp','n']].to_string(index=False),flush=True)

if __name__=='__main__':compare()
