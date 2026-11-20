import unittest
import numpy as np
import pandas as pd
from pathlib import Path
from cs2_research_pipeline import battery_id,integral,interval_evidence
from cs2_research_analysis import forecast_pairs,phase_dependent_columns
class ResearchChecks(unittest.TestCase):
 def test_full_identifier(self):
  for k in [3,33,38]:self.assertEqual(battery_id(Path(f'CS2_{k}/CS2_{k}_1_1_10.xlsx')),f'CS2_{k}')
  with self.assertRaises(ValueError):battery_id(Path('CS2_3/CS2_33_1.xlsx'))
 def test_integration_units_and_gap(self):
  t=np.arange(0,3601,60);self.assertAlmostEqual(integral(t,np.ones(len(t)))/3600,1.)
  self.assertEqual(integral(np.array([0,1000]),np.array([1,1])),0.)
 def test_forecasts_do_not_cross_files_or_missing_episodes(self):
  d=pd.DataFrame([dict(battery='CS2_33',source_file='a',episode_in_file=1,charge_start_s=0,discharge_end_s=100,soh=1.),dict(battery='CS2_33',source_file='b',episode_in_file=2,charge_start_s=101,discharge_end_s=200,soh=.9)])
  self.assertEqual(len(forecast_pairs(d,1)),0)
  d.source_file='a';self.assertEqual(len(forecast_pairs(d,1)),1)
  d.loc[1,'episode_in_file']=3;self.assertEqual(len(forecast_pairs(d,1)),0)
 def test_long_rest_requires_matching_step_clock(self):
  rest=pd.DataFrame(dict(t=[0.,300.,600.],i=[0.,0.,0.],step=[1,1,1],step_t=[0.,300.,600.]))
  self.assertTrue(interval_evidence(rest).all())
  rest.loc[1,'step_t']=10.
  self.assertFalse(interval_evidence(rest).all())
 def test_phase_masks_remove_whole_charge_aggregates(self):
  cc,cv=phase_dependent_columns(['cc_duration_s','cv_duration_s','charge_voltage_mean_v','throughput_bracket_3p9_4p1_ah'])
  self.assertIn('charge_voltage_mean_v',cc)
  self.assertIn('charge_voltage_mean_v',cv)
  self.assertIn('throughput_bracket_3p9_4p1_ah',cc)
  self.assertNotIn('cc_duration_s',cv)
if __name__=='__main__':unittest.main()
