"""Shared reproducible physical thresholds; units are seconds, volts, amperes, Ah."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'cx2-v1'
CONFIG = dict(nominal_capacity_ah=1.35, active_current_a=0.02,
              max_sample_gap_s=180., max_cv_sample_gap_s=3600., max_rest_gap_s=3600.,
              capacity_agreement_relative=0.05, capacity_agreement_absolute_ah=0.01,
              charge_voltage_v=4.18, termination_current_a=0.06,
              discharge_cutoff_v=2.75, conventional_crate_min=0.30,
              conventional_crate_max=1.10, conventional_peak_crate_max=1.20, voltage_grid_v=0.002,
              ic_interval_low_v=3.8, ic_interval_high_v=4.15,
              smoothing_windows=(7, 11, 15), seed=42)

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def save_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, default=str, allow_nan=False), encoding='utf-8')

def integrate_current(time_s, current_a, max_gap_s=180.):
    """Trapezoids only over strictly positive, finite, short intervals; return Ah and invalid count.

    Never bridge a recording gap. Integrate each step separately to avoid interpolating
    across current switches. Initial step interval is handled by phase_integral.
    """
    t, i = np.asarray(time_s, float), np.asarray(current_a, float)
    dt = np.diff(t)
    good = np.isfinite(dt) & (dt > 0) & (dt <= max_gap_s) & np.isfinite(i[:-1]) & np.isfinite(i[1:])
    return float(np.sum(dt[good]*(i[:-1][good]+i[1:][good])/2)/3600), int((~good).sum())
