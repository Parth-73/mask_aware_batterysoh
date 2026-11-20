"""Reported settings and explicit assumptions for the paper-style LSTM experiment."""
from dataclasses import dataclass,asdict
from pathlib import Path
from ..training.config import file_sha,json_write
ROOT=Path(__file__).resolve().parents[2]
OUTPUT=ROOT/'outputs'/'paper_lstm_reproduction'
BATTERIES=['CX2_34','CX2_36','CX2_37','CX2_38']
SEEDS=[42,123,2026]
VARIANTS=['fixed_peak','full_cc_peak']
FEATURES=['ic_peak_height','partial_charge_ah','partial_dv_dq']
PUBLISHED={
 'CX2_34':dict(rmse_fraction=.0243,mae_fraction=.0192,mape_table_value=.0218),
 'CX2_36':dict(rmse_fraction=.0099,mae_fraction=.0078,mape_table_value=.0094),
 'CX2_37':dict(rmse_fraction=.0106,mae_fraction=.0084,mape_table_value=.0101),
 'CX2_38':dict(rmse_fraction=.0110,mae_fraction=.0089,mape_table_value=.0106)}

@dataclass(frozen=True)
class Config:
    sequence_length:int=5
    layers:int=2
    hidden_size:int=64
    gaussian_sigma_samples:float=1.
    voltage_grid_v:float=.002
    fixed_peak_low_v:float=3.8
    fixed_peak_high_v:float=4.15
    eol_soh_fraction:float=.8
    train_fraction:float=.5
    validation_fraction_of_train:float=.2
    dropout:float=0.
    optimizer:str='Adam'
    learning_rate:float=.001
    weight_decay:float=0.
    batch_size:int=64
    max_epochs:int=200
    patience:int=20
    min_delta:float=1e-6
    gradient_clip:float=1.
    scheduler_patience:int=5
    scheduler_factor:float=.5
    scheduler_min_lr:float=1e-6
    trend_degree:int=2
    trend_sigma_threshold:float=6.
    jump_level_floor:float=.03
    max_acquisition_gap_s:float=3600.

def definition(config):
    return dict(settings=asdict(config),batteries=BATTERIES,seeds=SEEDS,variants=VARIANTS,
       primary_variant='fixed_peak',features=FEATURES,
       paper=dict(pdf=str(ROOT.parent/'batteries-12-00115.pdf'),doi='10.3390/batteries12040115',reported_pages={'target_and_aging_domain':4,'partial_charge_features':5,'preprocessing':6,'predictor_dimensions':9,'chronological_split':12,'published_lstm_results':13}),
       assumptions=dict(optimizer_and_dropout='Not reported. Fixed Adam lr=0.001, no weight decay, zero dropout; no search against test or published scores.',
        training_budget='Not reported. Max 200 epochs; internal validation chooses epoch count, then a fresh model is fitted to the complete first half for that count. Same scheduler/epoch LR schedule is replayed.',
        peak_domain='Not reported. Primary retains the established [3.8,4.15] V CC interval. Predeclared sensitivity uses all observed CC voltages below 4.18 V. Neither variant is chosen using test performance.',
        gaussian_grid='Not reported. Uniform 2 mV grid, numerical dQ/dV then Gaussian sigma=1 grid sample (2 mV); reflect boundary mode.',
        outlier_rule='Not reported. Training-only robust quadratic feature trend plus 6-MAD residual AND large adjacent jump. Only observed spikes corrected; unavailable features and SOH labels never filled/smoothed.',
        normalization='Range [0,1] reported; fit partition unspecified. MinMaxScaler fits only fitting/training observations. Unseen values may fall outside [0,1]; no clipping or test fitting.',
        eol_rule='First validated routine SOH strictly below 0.8 ends the chronological prefix. No later recovery reentry; no SOH clipping at 1.0.',
        windows='Original consecutive ordinals, same single acquisition source, no rejected/missing rows, no counter reset, gap <=3600s. Past fitting-half inputs may inform a later-half target, but never future inputs.',
        split='First half of complete Gaussian-feature records within the aging prefix is training; latter half is test. Window targets preserve this split. Final model uses all eligible first-half windows.',
        validation='Last 20% of the training half is internal validation. Inner trend/scaler fit only the earlier 80%; final trend/scaler refit only the entire training half.',
        target='Existing validated discharge current integral / 1.35 Ah, fractional SOH; no same-cycle target or discharge capacity input.',
        dataset_scope='Only the four CX2 batteries in Table 4 are locally available. CS2 and Tongji results are not reproduced. Observed rates include 1C despite the paper table summary of 0.5C for CX2.',
        claim='Best-effort implementation of reported pure-LSTM predictor methodology; not exact author-code reproduction or AW-PINN. Previous test exposure and unspecified details prevent an independent superiority claim.'))
