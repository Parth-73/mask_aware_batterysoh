"""Frozen benchmark definitions, independent of the earlier feature-analysis split."""
from dataclasses import dataclass,asdict
from pathlib import Path
import hashlib,json

ROOT=Path(__file__).resolve().parents[2]
OUTPUT=ROOT/'outputs'/'lstm_baselines'
SPLITS={'train':['CX2_16','CX2_33','CX2_34','CX2_35','CX2_37'], 'validation':['CX2_36'],'test':['CX2_38']}
FEATURES={
 'F1':['ic_peak_height','partial_charge_ah','partial_dv_dq'],
 'F2':['ic_peak_height','partial_charge_ah','partial_dv_dq','ic_peak_area','ic_peak_voltage','ic_peak_width','cc_duration_s','cv_duration_s','dva_peak'],
 'F3':['cc_duration_s','cv_duration_s','dva_peak','ic_peak_voltage','ic_peak_width']}
ARCHITECTURES=['B1','B2','B3','B4']
SEQUENCES=[5,10,20]
SEEDS=[42,123,2026]

@dataclass(frozen=True)
class TrainingConfig:
    learning_rate:float=.0005
    weight_decay:float=.001
    batch_size:int=64
    max_epochs:int=200
    patience:int=20
    gradient_clip:float=1.
    scheduler_factor:float=.5
    scheduler_patience:int=5
    scheduler_min_lr:float=1e-6
    early_stop_min_delta:float=1e-6
    hidden_size:int=64
    max_acquisition_gap_s:float=3600.
    deterministic:bool=True
    device:str='cuda'

def file_sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def json_write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    def clean(x):
        import numpy as np
        if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)):return [clean(v) for v in x]
        if isinstance(x,(np.integer,)):return int(x)
        if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
        if isinstance(x,Path):return str(x)
        return x
    temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(json.dumps(clean(data),indent=2,allow_nan=False),encoding='utf-8');temporary.replace(path)

def experiment_definition(config):
    return dict(training=asdict(config),splits=SPLITS,features=FEATURES,architectures=ARCHITECTURES,sequences=SEQUENCES,seeds=SEEDS,
      target='validated SOH fraction = discharge capacity Ah / 1.35 Ah; current-charge features estimate same-cycle eventual discharge SOH',
      cohort='Common complete-case F2 feature union for every feature set; no imputation, no substitutions',
      continuity='Same battery and single source sheet, strictly consecutive reconstructed ordinals, every record quality-validated, nonoverlapping recorded times, inter-cycle gap <=3600 s, no local resets',
      scale='StandardScaler fitted only on common complete-case training-cycle features; target remains fractional and unscaled',
      selection='Mean validation RMSE across all three seeds. Select feature set at B1/L5; architecture at L5; sequence for chosen architecture. Full architecture x sequence grid also reported.',
      test_policy='Freeze all choices before evaluating selected and literature-comparison checkpoints once per seed. Test metrics never feed training or selection.',
      prior_feature_selection='F3 is the user-prespecified provisional compact set; its earlier exploratory selection included CX2_36. Do not regard current validation as independent of that prior exploratory feature work. CX2_38 remained excluded.',
      literature=dict(doi='10.3390/batteries12040115',url='https://www.mdpi.com/2313-0105/12/4/115',settings='F1, length 5, two LSTM layers, hidden 64; same AdamW/MSE benchmark settings. No physical loss or weight-generator branch.',paper_verification='Predictor settings supplied in task; detailed paper verification recorded separately in literature_verification.json.'))
