"""Execute representative gate, resumable extraction, forecasting and reports."""
import json,logging
from cs2_research_pipeline import OUT,extract,CONFIG
from cs2_research_analysis import analyze
from cs2_continuity import audit_continuity
from cs2_partial_views import run_partial_views
from cs2_acquisition_robustness import run_acquisition_robustness
from cs2_history_robustness import run_history_robustness
from finalize_cs2_research import finalize
if __name__=='__main__':
 OUT.mkdir(parents=True,exist_ok=True)
 logging.basicConfig(filename=str(OUT/'pipeline.log'),level=logging.INFO)
 representative,_=extract(representative=True)
 assert set(representative.battery)=={'CS2_33','CS2_34','CS2_3','CS2_5'}
 assert representative.loc[representative.label_valid,'discharge_end_v'].le(CONFIG['full_discharge_end_v']).all()
 data,inventory=extract()
 audit_continuity(data)
 analyze(data,inventory)
 run_acquisition_robustness()
 run_partial_views()
 run_history_robustness()
 print(json.dumps(finalize(),indent=2))
