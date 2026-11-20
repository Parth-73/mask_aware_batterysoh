import json,tempfile,unittest
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from src.training.config import FEATURES,SPLITS
from src.training.dataset import build_windows,prepare,audit_source
from src.training.evaluate import metrics
from src.models.lstm_baselines import LSTMBaseline


def frame_for(battery='CX2_16',n=30):
    rows=[]
    for j in range(n):
        start=pd.Timestamp('2011-01-01')+pd.Timedelta(hours=j)
        r={f:.1+k*.2+j*.001 for k,f in enumerate(FEATURES['F2'])}
        y=.9-j*.001
        r.update(battery=battery,cycle_id=f'{battery}:{j+1:05d}',lifetime_cycle=j+1,start=start,end=start+pd.Timedelta(seconds=3590),sources=battery+'::channel',
                 local_counter_resets=0,validated_complete=True,soh=y,validated_discharge_capacity_ah=y*1.35,protocol='conventional',rejection_reasons='',benchmark_split='train',cycle_exclusion='')
        for phase in ['charge','discharge']:r[phase+'_integral_ah']=y*1.35;r[phase+'_counter_ah']=y*1.35
        rows.append(r)
    return pd.DataFrame(rows)


def windows(f,length=5):return build_windows(f,length,FEATURES['F2'],np.zeros(9),np.ones(9))

class LSTMTests(unittest.TestCase):
    def test_original_ordinals_and_current_target(self):
        d=frame_for(n=10);w,a=windows(d)
        self.assertEqual(w.x.shape,(6,5,9));self.assertAlmostEqual(float(w.y[0]),d.soh.iloc[4],places=6)
        self.assertEqual(json.loads(w.metadata.input_ordinals.iloc[0]),[1,2,3,4,5])
    def test_rejected_history_not_compacted(self):
        d=frame_for(n=10);d.loc[5,'cycle_exclusion']='quality_rejected';w,a=windows(d)
        self.assertEqual(len(w.y),1);self.assertEqual(w.metadata.lifetime_cycle.iloc[0],5)
        self.assertTrue(a.loc[a.lifetime_cycle==6,'rejection_reasons'].str.contains('rejected_or_missing').all())
    def test_ordinal_gap(self):
        d=frame_for(n=10);d.loc[5:,'lifetime_cycle']+=1;w,a=windows(d)
        self.assertEqual(len(w.y),2);self.assertTrue(a.rejection_reasons.str.contains('ordinal_gap').any())
    def test_session_boundaries(self):
        d=frame_for(n=10);d.loc[5:,'sources']='new_file::channel';w,a=windows(d)
        self.assertEqual(len(w.y),2);self.assertTrue(a.rejection_reasons.str.contains('session_boundary').any())
    def test_time_gap_and_battery_boundaries(self):
        d=frame_for(n=10);d.loc[5:,'start']+=pd.Timedelta(days=10);d.loc[5:,'end']+=pd.Timedelta(days=10)
        w,a=windows(pd.concat([d,frame_for('CX2_33',10)],ignore_index=True))
        self.assertEqual(len(w.y),8)
        for ids in w.metadata.input_cycle_ids.map(json.loads):self.assertEqual(len({s.split(':')[0] for s in ids}),1)
        self.assertTrue(a.rejection_reasons.str.contains('acquisition_gap').any())
    def test_no_future_observations(self):
        d=frame_for(n=10);a,_=windows(d);d.loc[6:,FEATURES['F2']]=999;b,_=windows(d)
        np.testing.assert_array_equal(a.x[:2],b.x[:2]);np.testing.assert_array_equal(a.y[:2],b.y[:2])
    def test_scaler_is_train_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);data=pd.concat([frame_for(b,25) for bs in SPLITS.values() for b in bs],ignore_index=True)
            feature=path/'f.csv';quality=path/'q.csv';rejected=path/'r.csv';data.to_csv(feature,index=False);data.to_csv(quality,index=False);data.iloc[:0].to_csv(rejected,index=False)
            prepare(path/'one',feature_path=feature,quality_path=quality,rejected_path=rejected)
            data.loc[data.battery.isin(SPLITS['test']+SPLITS['validation']),FEATURES['F2']]=1e6
            data.to_csv(feature,index=False);data.to_csv(quality,index=False)
            prepare(path/'two',feature_path=feature,quality_path=quality,rejected_path=rejected)
            a=json.loads((path/'one/feature_preprocessing.json').read_text());b=json.loads((path/'two/feature_preprocessing.json').read_text())
            self.assertEqual(a,b);self.assertEqual(a['fit_batteries'],sorted(SPLITS['train']))
    def test_fraction_target_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);data=pd.concat([frame_for(b,25) for bs in SPLITS.values() for b in bs],ignore_index=True);data['soh']*=100
            data.to_csv(p/'f.csv',index=False);data.to_csv(p/'q.csv',index=False);data.iloc[:0].to_csv(p/'r.csv',index=False)
            with self.assertRaisesRegex(ValueError,'fractional'):audit_source(p/'f.csv',p/'q.csv',p/'r.csv')
    def test_feature_sets_and_split(self):
        self.assertEqual(FEATURES['F1'],['ic_peak_height','partial_charge_ah','partial_dv_dq'])
        self.assertEqual(len(FEATURES['F2']),9);self.assertEqual(len(FEATURES['F3']),5)
        self.assertEqual(SPLITS['validation'],['CX2_36']);self.assertNotIn('CX2_36',SPLITS['train'])
        for columns in FEATURES.values():self.assertFalse(any('soh' in s or 'discharge' in s for s in columns))
    def test_bidirectional_final_states(self):
        torch.manual_seed(42);model=LSTMBaseline(3,'B3').eval();x=torch.randn(4,5,3)
        with torch.no_grad():
            output,(h,c)=model.lstm(x);expected=model.head(torch.cat([h[-2],h[-1]],dim=-1)).squeeze(-1)
            np.testing.assert_allclose(model(x).numpy(),expected.numpy(),atol=1e-7)
            self.assertFalse(torch.allclose(expected,model.head(output[:,-1]).squeeze(-1)))
    def test_attention_normalized_and_trainable(self):
        model=LSTMBaseline(3,'B4');p,w=model(torch.randn(4,5,3),True)
        self.assertEqual(w.shape,(4,5));torch.testing.assert_close(w.sum(1),torch.ones(4));self.assertTrue((w>=0).all())
        p.sum().backward();self.assertTrue(all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.attention.parameters()))
    def test_metric_units(self):
        m=metrics([.5,.5],[.51,.49]);self.assertAlmostEqual(m['rmse_fraction'],.01);self.assertAlmostEqual(m['rmse_percentage_points'],1.);self.assertAlmostEqual(m['mape_percent'],2.)
if __name__=='__main__':unittest.main()
