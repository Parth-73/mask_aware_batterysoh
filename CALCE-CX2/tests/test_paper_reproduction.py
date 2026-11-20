import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from src.paper_reproduction.config import Config
from src.paper_reproduction.features import gaussian_features
from src.paper_reproduction.dataset import split_roles,process,make_windows
from src.paper_reproduction.model import PaperLSTM


def synthetic(n=100):
    rows=[]
    for j in range(n):
        start=pd.Timestamp('2011-01-01')+pd.Timedelta(hours=j)
        rows.append(dict(battery='CX2_38',cycle_id=f'CX2_38:{j+1:05d}',lifetime_cycle=j+1,start=start,end=start+pd.Timedelta(seconds=3590),sources='one_source',local_counter_resets=0,soh=.99-j*.0015,
                         fixed_peak=3+j*.001,full_cc_peak=3+j*.001,partial_charge_ah=.45-j*.0005,partial_dv_dq=.2/(.45-j*.0005),feature_exclusion=''))
    return pd.DataFrame(rows)

class PaperTests(unittest.TestCase):
    def test_gaussian_ic_definition(self):
        c=Config();v=np.arange(3.5,4.1781,.002);q=.3*(v-3.5)+.01*np.tanh((v-3.9)*20)
        cc=np.column_stack([v,q,np.arange(len(v))*30,np.ones(len(v))*.675])
        with patch('src.paper_reproduction.features.charge_curve',return_value=(cc,np.empty((0,4)),0,0,0,False)):
            f,reason,curve=gaussian_features([],c)
        grid,ic=curve;expected=gaussian_filter1d(np.gradient(np.interp(grid,v,q),grid),1,mode='reflect')
        np.testing.assert_allclose(ic,expected,atol=1e-12)
        self.assertAlmostEqual(f['partial_charge_ah'],float(np.interp(4.,v,q)-np.interp(3.8,v,q)))
        self.assertAlmostEqual(f['partial_dv_dq'],.2/f['partial_charge_ah'])
    def test_chronological_half_and_inner_validation(self):
        f,s=split_roles(synthetic(),Config())
        self.assertEqual(s['training_half_cycles'],50);self.assertEqual(s['internal_fit_cycles'],40)
        self.assertEqual(s['internal_validation_cycles'],10);self.assertEqual(s['test_half_cycles'],50)
        self.assertEqual(f.role.iloc[39],'fit');self.assertEqual(f.role.iloc[40],'validation');self.assertEqual(f.role.iloc[50],'test')
    def test_scaler_and_trends_never_fit_test(self):
        f,s=split_roles(synthetic(),Config());a,x,p,cor=process(f,'fixed_peak',f.role.eq('fit'),Config())
        changed=f.copy();changed.loc[changed.role.eq('test'),'fixed_peak']=99
        b,z,r,cor=process(changed,'fixed_peak',changed.role.eq('fit'),Config())
        self.assertEqual(p,r)
        self.assertTrue(np.allclose(x[:40].min(0),0,atol=1e-10));self.assertTrue(np.allclose(x[:40].max(0),1,atol=1e-10))
        self.assertTrue((x[70]>1).any() or (x[70]<0).any())
    def test_observed_spike_corrected_but_soh_unchanged(self):
        raw=synthetic();raw.loc[80,'fixed_peak']+=1;raw.loc[80,'partial_charge_ah']+=.1
        f,s=split_roles(raw,Config());processed,x,p,cor=process(f,'fixed_peak',f.role.eq('fit'),Config())
        self.assertEqual(len(cor),2);self.assertAlmostEqual(processed.ic_peak_height_corrected.iloc[80],3.080,places=8)
        self.assertAlmostEqual(processed.partial_charge_ah_corrected.iloc[80],.410,places=8)
        pd.testing.assert_series_equal(processed.soh,f.soh)
        self.assertAlmostEqual(processed.partial_dv_dq_corrected.iloc[80],.2/.410,places=8)
    def test_missing_features_not_trend_filled(self):
        raw=synthetic();raw.loc[80,'fixed_peak']=np.nan;raw.loc[80,'feature_exclusion']='missing_measurement'
        f,s=split_roles(raw,Config());processed,x,p,cor=process(f,'fixed_peak',f.role.eq('fit'),Config())
        self.assertTrue(np.isnan(processed.ic_peak_height_corrected.iloc[80]));self.assertFalse(processed.processing_valid.iloc[80])
        windows,audit=make_windows(processed,x,Config())
        ids=sum(windows['test'].metadata.input_cycle_ids.map(__import__('json').loads).tolist(),[])
        self.assertNotIn(raw.cycle_id.iloc[80],ids)
    def test_window_boundary_uses_only_observed_history(self):
        f,s=split_roles(synthetic(),Config());p,x,meta,c=process(f,'fixed_peak',f.role.isin(['fit','validation']),Config());w,audit=make_windows(p,x,Config())
        first=w['test'].metadata.iloc[0]
        self.assertEqual(first.lifetime_cycle,51);self.assertEqual(__import__('json').loads(first.input_ordinals),[47,48,49,50,51])
    def test_missing_ordinals_and_sessions_are_not_compacted(self):
        raw=synthetic();raw.loc[70:,'lifetime_cycle']+=1;raw.loc[60:,'sources']='later_source'
        f,s=split_roles(raw,Config());p,x,meta,c=process(f,'fixed_peak',f.role.isin(['fit','validation']),Config());w,a=make_windows(p,x,Config())
        self.assertTrue(a.rejection_reasons.str.contains('ordinal_gap').any());self.assertTrue(a.rejection_reasons.str.contains('session_boundary').any())
    def test_reported_dimensions(self):
        m=PaperLSTM(Config());self.assertEqual(m.lstm.input_size,3);self.assertEqual(m.lstm.hidden_size,64);self.assertEqual(m.lstm.num_layers,2);self.assertEqual(m.lstm.dropout,0);self.assertEqual(m.head.out_features,1)
if __name__=='__main__':unittest.main()
