import unittest
import numpy as np
import pandas as pd
from src.common import integrate_current
from src.reconstruct_cycles import counter_delta,reconstruct,phase_integral,prepare_battery
from src.extract_features import extract,add_history
from src.feature_analysis import F1,F2,F3,SPLITS


def segment(phase,t0,t1,i,v0,v1,step=1,cycle=1,n=101,cc0=0,dc0=0):
    t=np.linspace(t0,t1,n);v=np.linspace(v0,v1,n);dt=(t1-t0)/(n-1)
    return pd.DataFrame(dict(timestamp=pd.Timestamp('2011-01-01')+pd.to_timedelta(t,unit='s'),test_s=t,step_s=t-t0,
        step=step,local_cycle=cycle,current=i,voltage=v,phase=phase,source='a',source_row=np.arange(n)+3,
        source_reversal=False,local_reset=False,raw_bad=False,gap_s=np.r_[0,np.diff(t)],
        charge_counter=cc0+np.maximum(i,0)*(t-t0)/3600,discharge_counter=dc0+np.maximum(-i,0)*(t-t0)/3600,
        prev_charge_counter=cc0+np.maximum(i,0)*np.maximum(0,t-t0-dt)/3600,
        prev_discharge_counter=dc0+np.maximum(-i,0)*np.maximum(0,t-t0-dt)/3600))

class PipelineTests(unittest.TestCase):
    def test_capacity(self):
        q,bad=integrate_current([0,60,120],[.675,.675,.675]);self.assertAlmostEqual(q,.0225);self.assertEqual(bad,0)
    def test_gap_not_bridged(self):
        q,bad=integrate_current([0,60,10000],[1,1,1]);self.assertAlmostEqual(q,1/60);self.assertEqual(bad,1)
    def test_event_logged_cv(self):
        cv=segment('charge',0,1200,.1,4.2,4.2,n=4)
        cv['current']=[.2,.15,.1,.05]
        q,bad,_=phase_integral(cv,1)
        self.assertEqual(bad,0);self.assertAlmostEqual(q,150/3600)
    def test_cc_gap_still_rejected(self):
        cc=segment('charge',0,1200,.675,3.5,4.1,n=4)
        q,bad,_=phase_integral(cc,1);self.assertEqual(bad,3)
    def test_counter_lifetime_offset(self):
        g=segment('discharge',0,3600,-1,4.1,2.7,dc0=100)
        q,_,reason=counter_delta(g,'discharge_counter');self.assertAlmostEqual(q,1);self.assertEqual(reason,'')
    def test_counter_reset_flag(self):
        g=segment('discharge',0,3600,-1,4.1,2.7);g.loc[50:,'discharge_counter']-=.5
        q,r,reason=counter_delta(g,'discharge_counter');self.assertTrue(np.isnan(q));self.assertGreater(r,0)
    def test_steps_and_repeated_local_cycle(self):
        a=segment('charge',0,3600,.675,3.5,4.17,1)
        b=segment('charge',3601,5401,.1,4.2,4.2,2,cc0=.675);b.loc[b.index[-1],'current']=.05
        c=segment('discharge',5402,7202,-.675,4.1,3.5,3,dc0=10)
        d=segment('discharge',7203,9003,-.675,3.5,2.7,4,dc0=10.3375)
        e=segment('charge',9004,12604,.675,3.5,4.17,1)
        data=pd.concat([a,b,c,d,e],ignore_index=True);data['segment']=np.repeat(range(5),101)
        cycles,q=reconstruct(data,'TEST');self.assertEqual(len(cycles),2);self.assertEqual(q.discharge_steps.iloc[0],2)
        self.assertAlmostEqual(q.discharge_integral_ah.iloc[0],.675)
        self.assertTrue(q.validated_complete.iloc[0]);self.assertAlmostEqual(q.soh.iloc[0],.5)
        self.assertFalse(q.validated_complete.iloc[1]);self.assertTrue(np.isnan(q.soh.iloc[1]))
    def test_continuous_file_boundary(self):
        from unittest.mock import patch
        whole=segment('charge',0,3600,.675,3.5,4.17,n=121)
        a=whole.iloc[:60].copy();b=whole.iloc[60:].copy()
        a['source']='a';b['source']='b';b['source_row']=np.arange(len(b))+2
        manifest=pd.DataFrame([dict(battery='TEST',status='ready',start=str(g.timestamp.min()),end=str(g.timestamp.max()),source_file=name,sheet='Channel',cache=name) for name,g in [('a',a),('b',b)]])
        with patch('src.reconstruct_cycles.pd.read_pickle',side_effect=lambda path:{'a':a.copy(),'b':b.copy()}[path]):
            d,audit=prepare_battery(manifest,'TEST')
        self.assertEqual(d.boundary_continuation.sum(),1)
        groups=[g for _,g in d.groupby('segment')]
        q=sum(phase_integral(g,1)[0] for g in groups)
        c=sum(counter_delta(g,'charge_counter')[0] for g in groups)
        self.assertAlmostEqual(q,.675);self.assertAlmostEqual(c,.675)
    def test_partial_features(self):
        cc=segment('charge',0,3600,.675,3.5,4.179,n=501)
        f,meta,curves=extract([cc]);expected=.675*.2/.679
        self.assertAlmostEqual(f['partial_charge_ah'],expected,places=6)
        self.assertAlmostEqual(f['partial_dv_dq'],.2/expected,places=6)
        self.assertTrue(np.isnan(f['cv_duration_s']));self.assertIn('ic',curves)
    def test_missing_voltage_coverage(self):
        f,_,_=extract([segment('charge',0,600,.675,4.05,4.17)])
        self.assertTrue(np.isnan(f['partial_charge_ah']));self.assertTrue(np.isnan(f['ic_peak_height']))
    def test_history_causal(self):
        g=pd.DataFrame(dict(battery=['a']*3,lifetime_cycle=[1,2,3],ic_peak_voltage=[4,4.01,4.02],partial_charge_ah=[.2,.19,.18],resistance_ohm=[np.nan]*3))
        a=add_history(g);g.loc[2,'ic_peak_voltage']=3.9;b=add_history(g)
        self.assertTrue(np.isnan(a.historical_ic_peak_shift_v.iloc[0]));self.assertEqual(a.historical_ic_peak_shift_v.iloc[1],b.historical_ic_peak_shift_v.iloc[1])
    def test_test_batteries_do_not_affect_selection(self):
        from unittest.mock import patch
        from pathlib import Path
        from src.feature_analysis import analyze
        from src.extract_features import FEATURES
        rows=[]
        for battery,split in [('CX2_16','train'),('CX2_33','train'),('CX2_38','test')]:
            for j in range(20):
                r={f:np.nan for f in FEATURES}
                r.update(battery=battery,split=split,eligible_analysis=True,soh=.7+j*.01,partial_charge_ah=.1+j*.005)
                rows.append(r)
        a=pd.DataFrame(rows);b=a.copy();b.loc[b.split=='test','soh']=999
        b.loc[b.split=='test','partial_charge_ah']=-999
        with patch('src.feature_analysis.save_json'),patch('src.feature_analysis.heatmap'),patch.object(pd.DataFrame,'to_csv'):
            ar,aset=analyze(a,Path('unused'));br,bset=analyze(b,Path('unused'))
        pd.testing.assert_frame_equal(ar,br);self.assertEqual(aset,bset);self.assertEqual(aset,['partial_charge_ah'])
    def test_vif_constant_complete_cases(self):
        from src.feature_analysis import compute_vif
        d=pd.DataFrame({'a':np.arange(20,dtype=float),'b':[0.]*10+[1.]*10,'c':[np.nan]*10+list(np.arange(10,dtype=float))})
        rows=compute_vif(d,['a','b','c'])
        b=next(r for r in rows if r['feature']=='b')
        self.assertEqual(b['status'],'constant_or_absent_in_complete_cases');self.assertTrue(np.isnan(b['vif']))
    def test_routine_and_high_rate_protocols(self):
        from src.reconstruct_cycles import classify_protocol
        self.assertEqual(classify_protocol(.5,.501,1),'conventional')
        self.assertEqual(classify_protocol(1.,1.001,1),'conventional')
        self.assertEqual(classify_protocol(3.,3.001,1),'unusual_discharge')
        self.assertEqual(classify_protocol(1.,1.001,2),'unusual_discharge')
    def test_no_target_leakage(self):
        for f in F1+F2+F3:self.assertNotIn('discharge',f);self.assertNotIn('soh',f)
        self.assertFalse(set(SPLITS['train'])&set(SPLITS['test']))
if __name__=='__main__':unittest.main()
