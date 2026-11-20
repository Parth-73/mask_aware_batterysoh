"""Independent source, checkpoint, cutoff, scaling, cohort and metric integrity audit."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from src.common import ROOT,digest
from src.training.config import json_write
from .prepare import OUT,SOURCE,FEATURES
from .bundles import load_bundle


def main():
    samples=pd.read_csv(OUT/'processed/samples.csv').set_index('cycle_id')
    protected=json.loads((OUT/'data_audit/preexisting_snapshot.json').read_text())
    for name,sha in protected.items():assert digest(ROOT/name)==sha,f'Pre-existing artifact changed: {name}'
    prediction_rows=0;runs=0;outer=0;cohorts={};bundle_checks=0;physical=[]
    for directory in sorted((OUT/'checkpoints').iterdir()):
        definition=directory/'configuration.json'
        if not definition.exists() or not (directory/'validation_result.json').exists():continue
        frozen=json.loads(definition.read_text());config=frozen['parameters'];runs+=1
        for name,sha in frozen['source_hashes'].items():assert digest(SOURCE/name)==sha,f'Source drift: {name}'
        prep=json.loads((directory/'preprocessing.json').read_text());split=prep['partitions']
        assert len(set(sum(split.values(),[])))==7 and len(sum(split.values(),[]))==7
        assert all(i.split(':')[0] in split['train'] for i in prep['fit_cycle_ids'])
        fit=samples.loc[prep['fit_cycle_ids'],FEATURES].to_numpy()
        np.testing.assert_allclose(fit.mean(0),prep['feature_mean'],rtol=1e-10,atol=1e-10)
        np.testing.assert_allclose(np.where(fit.std(0)==0,1,fit.std(0)),prep['feature_scale'],rtol=1e-8,atol=1e-10)
        for role in ['validation','outer']:
            result_file=directory/f'{role}_result.json'
            if not result_file.exists():continue
            result=json.loads(result_file.read_text());assert digest(directory/'best.pt')==result['checkpoint_sha256']
            frame=pd.read_csv(directory/f'predictions_{role}.csv');prediction_rows+=len(frame)
            assert set(frame.battery)<=set(split[role])
            actual=samples.loc[frame.cycle_id,'soh'].to_numpy();pred=frame.predicted_soh.to_numpy()
            np.testing.assert_allclose(actual,frame.actual_soh,rtol=0,atol=1e-10)
            error=pred-actual
            derived={'mae_fraction':np.mean(abs(error)),'mse_fraction_squared':np.mean(error**2),
                'rmse_fraction':np.sqrt(np.mean(error**2)),'mae_percentage_points':100*np.mean(abs(error)),
                'rmse_percentage_points':100*np.sqrt(np.mean(error**2)),
                'mape_percent':100*np.mean(abs(error)/np.maximum(abs(actual),1e-8)),
                'smape_percent':200*np.mean(abs(error)/np.maximum(abs(actual)+abs(pred),1e-8)),
                'r2':1-np.sum(error**2)/np.sum((actual-actual.mean())**2),
                'bias_fraction':error.mean(),'median_absolute_error_fraction':np.median(abs(error)),
                'max_absolute_error_fraction':max(abs(error)),'p95_absolute_error_fraction':np.quantile(abs(error),.95)}
            assert result['n_samples']==len(frame)
            for metric,value in derived.items():assert np.isclose(result[metric],value,atol=2e-6,rtol=1e-5),(directory.name,metric)
            for row in frame.itertuples():
                ids=json.loads(row.input_cycle_ids);history=samples.loc[ids]
                assert ids[-1]==row.cycle_id and len(ids)==(1 if config['module']=='stage1' else config['length'])
                assert set(history.battery)=={row.battery}
                assert (pd.to_datetime(history.prediction_cutoff)<=pd.Timestamp(row.prediction_cutoff)).all()
                assert (np.diff(history.lifetime_cycle)==1).all()
                if len(ids)>1:assert history.sources.nunique()==1 and ';' not in history.sources.iloc[0]
            if config.get('purpose')=='final':
                key=(config['outer'],config['validation'],role,config.get('matched20',False))
                ids=set(frame.cycle_id)
                if key in cohorts:assert cohorts[key]==ids
                else:cohorts[key]=ids
            if role=='outer':outer+=1
        history=pd.read_csv(directory/'training.csv')
        assert np.isfinite(history.select_dtypes('number')).all().all()
        chosen=json.loads((directory/'validation_result.json').read_text())['best_epoch']
        assert chosen==int(history.loc[history.validation_battery_macro_rmse.idxmin(),'epoch'])
        audit=directory/'physical_audit_outer.json'
        if audit.exists():
            values=json.loads(audit.read_text());assert values['negative_increment_fraction'] in [0.,None]
            physical.append({'configuration_id':directory.name,**values})
        bundle=directory/'standalone.pt'
        if bundle.exists():
            # Reload equivalence uses validation observations, never a second outer pass.
            model,_=load_bundle(bundle);model.eval()
            p=pd.read_csv(directory/'predictions_validation.csv')
            p=p[p.segment_id.eq(p.segment_id.iloc[0])].head(3)
            windows=[json.loads(ids) for ids in p.input_cycle_ids]
            source=pd.read_csv(OUT/'processed/samples.csv');all_curves=np.load(OUT.parent/'processed/curves.npy',mmap_mode='r')
            positions=source.reset_index().set_index('cycle_id')['index']
            index=np.array([[positions.loc[i] for i in ids] for ids in windows])
            f=(samples.loc[np.asarray(windows).reshape(-1),FEATURES].to_numpy()-np.array(prep['feature_mean']))/np.array(prep['feature_scale'])
            f=torch.tensor(f.reshape(len(p),len(windows[0]),5),dtype=torch.float32)
            c=np.asarray(all_curves[index]);c=(c-np.array(prep['curve_mean'])[None,None,:,None])/np.array(prep['curve_scale'])[None,None,:,None]
            c=torch.tensor(c,dtype=torch.float32)
            with torch.inference_mode():
                if config['module']=='stage3':
                    encoded=model.encode_cycles(f.reshape(-1,5),c.reshape(-1,3,176)).reshape(len(p),len(windows[0]),64)
                    pred=model.forward_embeddings(encoded)['prediction']
                else:pred=model(f,c)['prediction']
            np.testing.assert_allclose(pred.numpy(),p.predicted_soh,rtol=2e-5,atol=2e-6)
            bundle_checks+=1
    if physical:pd.DataFrame(physical).to_csv(OUT/'metrics/physical_consistency_audit.csv',index=False)
    result={'passed':True,'completed_experimental_fits':runs,'outer_evaluations':outer,
        'validated_prediction_rows':prediction_rows,'matched_final_cohorts':len(cohorts),
        'independent_checkpoint_bundle_reloads':bundle_checks,'protected_artifacts_unchanged':True,
        'checks':['all primary and secondary metric formulas','checkpoint and actual training-source SHA256',
            'train-only scaler fit identities and moments','disjoint battery splits','charge cutoff and original consecutive input IDs',
            'matched final cohorts','finite logs and validation-only best epochs','nonnegative latent degradation increments',
            'single-file checkpoint reload equivalence on validation data only']}
    json_write(OUT/'data_audit/artifact_validation.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
