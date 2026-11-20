"""Complete inner checks, freeze choices, then evaluate each outer checkpoint once."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from src.common import digest
from src.training.config import json_write
from .prepare import OUT
from .dataset import inner_batteries
from .engine import fit,evaluate_outer
from .physics_engine import fit as fit_physics,evaluate_outer as evaluate_physics
from .bundles import save_bundle
from .report import label,generate


def existing_equivalent(config):
    ignored={'purpose','status','search_candidate'}
    wanted={k:v for k,v in config.items() if k not in ignored}
    for directory in (OUT/'checkpoints').iterdir():
        if not (directory/'validation_result.json').exists():continue
        old=json.loads((directory/'configuration.json').read_text())['parameters']
        if {k:v for k,v in old.items() if k not in ignored}==wanted:return directory
    return None


def main():
    p=argparse.ArgumentParser();p.add_argument('--outer',default='CX2_38');p.add_argument('--skip-length-ablation',action='store_true')
    args=p.parse_args();outer=args.outer;validations=inner_batteries(outer,2)
    gates=[json.loads((OUT/'configs'/f'{outer}_gate_stage{n}.json').read_text()) for n in [1,2,3]]
    assert all(gate['passed'] for gate in gates)
    memory=json.loads((OUT/'configs/gate_memory_synthetic.json').read_text());assert memory['passed']
    primary=[r['path'] for gate in gates for r in gate['runs'] if r.get('validation',validations[0])==validations[0]]
    primary=list(dict.fromkeys(primary));assert len(primary)==21
    marker=OUT/'configs'/f'{outer}_outer_frozen.json'
    if not marker.exists():
        inner_results=[]
        for path in primary:
            directory=Path(path)
            cfg=json.loads((directory/'configuration.json').read_text())['parameters']
            result=json.loads((directory/'validation_result.json').read_text());result['model']=label(result)
            inner_results.append(result)
            candidate={**cfg,'validation':validations[1],'purpose':'inner_architecture_validation'}
            if cfg['module']=='stage2':
                parent=Path(cfg['parent']).parent
                parent_cfg=json.loads((parent/'configuration.json').read_text())['parameters']
                counterpart={**parent_cfg,'validation':validations[1]}
                found=existing_equivalent(counterpart)
                if found is None:found,_=fit({**counterpart,'purpose':'inner_encoder_pretraining'})
                candidate['parent']=str(found/'best.pt')
            elif cfg['module']=='stage3':
                strategy=gates[1]['selected_freeze_cnn']
                found=next(r['path'] for r in gates[1]['runs'] if r['freeze_cnn']==strategy and r['seed']==cfg['seed'] and r.get('validation')==validations[1])
                candidate['parent']=str(Path(found)/'best.pt')
            found=existing_equivalent(candidate)
            if found is None:
                found,_=(fit_physics(candidate) if cfg['module']=='stage3' else fit(candidate))
            result=json.loads((found/'validation_result.json').read_text());result['model']=label(result)
            inner_results.append(result)
        table=pd.DataFrame(inner_results)
        # Equal seed means within each validation battery, then equal inner-battery means.
        scores=table.groupby(['model','validation']).rmse_fraction.mean().groupby('model').mean().sort_values()
        table.to_csv(OUT/'metrics'/f'inner_architecture_validation_{outer}.csv',index=False)
        pd.concat([pd.read_csv(path) for path in (OUT/'metrics').glob('inner_architecture_validation_CX2_*.csv')],ignore_index=True).to_csv(OUT/'metrics/inner_architecture_validation.csv',index=False)
        json_write(OUT/'configs'/f'{outer}_validation_recommendation.json',{'objective':'two-inner-battery macro RMSE after averaging all three seeds',
            'scores':scores.to_dict(),'recommendation':scores.index[0],'outer_opened':False,
            'budget_caution':'A0-A3:15 epochs; A4:35 epochs; this is a pilot configuration recommendation, not isolated architecture superiority.'})
        extras=[]
        if not args.skip_length_ablation:
            source=Path(next(r['path'] for r in gates[1]['runs'] if r['seed']==42 and r.get('validation',validations[0])==validations[0]))
            cfg=json.loads((source/'configuration.json').read_text())['parameters']
            for length in [5,10,20]:
                config={**cfg,'freeze_cnn':True,'length':length,'matched20':True,'epochs':8,'seed':42,
                        'purpose':'sequence_ablation','status':'single_seed_matched_target_pilot'}
                path,_=fit(config);extras.append(str(path))
        selected=[Path(path) for path in primary+extras]
        frozen={'selected_run_ids':[path.name for path in selected],
                'checkpoint_hashes':{path.name:digest(path/'best.pt') for path in selected},
                'recommendation':scores.index[0],'objective':'two-inner-battery macro RMSE, all three seeds',
                'outer_battery':outer,'all_stage_choices_frozen_before_outer':True,
                'status':'partial_nested_pilot; one outer fold; experimental recovery stages unavailable'}
        json_write(marker,frozen)
    else:
        frozen=json.loads(marker.read_text());selected=[OUT/'checkpoints'/ident for ident in frozen['selected_run_ids']]
    for directory in selected:
        cfg=json.loads((directory/'configuration.json').read_text())['parameters']
        # Include frozen-CNN feature computation in device inference timing, without a
        # second held-out prediction pass or changing any trained model/source identity.
        import experiments.rapct_prony.staged.engine as ordinary
        import experiments.rapct_prony.staged.physics_engine as physical
        module=physical if cfg['module']=='stage3' else ordinary
        original=module.encode_cache;encoder_seconds=[0.]
        def timed_cache(model,tensors,ids):
            if not getattr(model,'frozen_encoder',False):return original(model,tensors,ids)
            if tensors[0].device.type=='cuda':torch.cuda.synchronize()
            start=time.perf_counter();value=original(model,tensors,ids)
            if tensors[0].device.type=='cuda':torch.cuda.synchronize()
            encoder_seconds[0]+=time.perf_counter()-start
            return value
        new_evaluation=not (directory/'outer_result.json').exists()
        module.encode_cache=timed_cache
        try:
            record=evaluate_physics(directory) if cfg['module']=='stage3' else evaluate_outer(directory)
        finally:module.encode_cache=original
        if new_evaluation:
            record['forward_only_seconds']=record['inference_seconds']
            record['encoder_preprocessing_seconds']=encoder_seconds[0]
            record['inference_seconds']+=encoder_seconds[0]
            record['inference_ms_per_sample']=1000*record['inference_seconds']/record['n_samples']
            record['inference_scope']='device CNN encoding plus model/state forward and CPU prediction transfer; no second held-out pass'
            json_write(directory/'outer_result.json',record)
        save_bundle(directory/'best.pt',directory/'standalone.pt')
        print(f"OUTER {directory.name}: RMSE={record['rmse_percentage_points']:.4f} pp",flush=True)
    # Source metadata and all stage predictions remain auditable; no absent stage gets a score.
    sequence=[]
    for directory in selected:
        result=json.loads((directory/'outer_result.json').read_text())
        if result.get('purpose')=='sequence_ablation':sequence.append(result)
    pd.DataFrame(sequence).to_csv(OUT/'metrics'/f'sequence_ablation_{outer}.csv',index=False)
    pd.concat([pd.read_csv(path) for path in (OUT/'metrics').glob('sequence_ablation_CX2_*.csv') if path.stat().st_size>2],ignore_index=True).to_csv(OUT/'metrics/sequence_ablation.csv',index=False)
    generate()


if __name__=='__main__':main()
