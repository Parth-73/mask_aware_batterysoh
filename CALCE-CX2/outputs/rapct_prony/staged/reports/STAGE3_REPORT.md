# Physics-guided degradation validation pilot

```json
{
  "passed": true,
  "stage": "3",
  "outer": "CX2_38",
  "runs": [
    {
      "path": "C:\\Projects\\Battery_Soh\\benchmark-repository\\CALCE-CX2\\outputs\\rapct_prony\\staged\\checkpoints\\A4_CX2_38_CX2_16_s42_925ce8ea7cc9",
      "configuration_id": "A4_CX2_38_CX2_16_s42_925ce8ea7cc9",
      "stage": "A4",
      "seed": 42,
      "physics_weight": 0.0,
      "smooth_weight": 0.0,
      "validation_rmse_fraction": 0.015536077830366849
    },
    {
      "path": "C:\\Projects\\Battery_Soh\\benchmark-repository\\CALCE-CX2\\outputs\\rapct_prony\\staged\\checkpoints\\A4_CX2_38_CX2_16_s123_1cb2d2b4533a",
      "configuration_id": "A4_CX2_38_CX2_16_s123_1cb2d2b4533a",
      "stage": "A4",
      "seed": 123,
      "physics_weight": 0.0,
      "smooth_weight": 0.0,
      "validation_rmse_fraction": 0.01681268218266095
    },
    {
      "path": "C:\\Projects\\Battery_Soh\\benchmark-repository\\CALCE-CX2\\outputs\\rapct_prony\\staged\\checkpoints\\A4_CX2_38_CX2_16_s2026_695382a6d3b0",
      "configuration_id": "A4_CX2_38_CX2_16_s2026_695382a6d3b0",
      "stage": "A4",
      "seed": 2026,
      "physics_weight": 0.0,
      "smooth_weight": 0.0,
      "validation_rmse_fraction": 0.01265592232238583
    },
    {
      "path": "C:\\Projects\\Battery_Soh\\benchmark-repository\\CALCE-CX2\\outputs\\rapct_prony\\staged\\checkpoints\\A4_CX2_38_CX2_16_s42_685bdbbdac72",
      "configuration_id": "A4_CX2_38_CX2_16_s42_685bdbbdac72",
      "stage": "A4",
      "seed": 42,
      "physics_weight": 0.01,
      "smooth_weight": 0.001,
      "validation_rmse_fraction": 0.015469062980013736
    },
    {
      "path": "C:\\Projects\\Battery_Soh\\benchmark-repository\\CALCE-CX2\\outputs\\rapct_prony\\staged\\checkpoints\\A4_CX2_38_CX2_16_s123_00257dad1576",
      "configuration_id": "A4_CX2_38_CX2_16_s123_00257dad1576",
      "stage": "A4",
      "seed": 123,
      "physics_weight": 0.01,
      "smooth_weight": 0.001,
      "validation_rmse_fraction": 0.016812415145713307
    },
    {
      "path": "C:\\Projects\\Battery_Soh\\benchmark-repository\\CALCE-CX2\\outputs\\rapct_prony\\staged\\checkpoints\\A4_CX2_38_CX2_16_s2026_e1e3a8d1538a",
      "configuration_id": "A4_CX2_38_CX2_16_s2026_e1e3a8d1538a",
      "stage": "A4",
      "seed": 2026,
      "physics_weight": 0.01,
      "smooth_weight": 0.001,
      "validation_rmse_fraction": 0.01261997052438683
    }
  ],
  "physics_search_selection": {
    "study": "CX2_38_physics",
    "best_trial": 1,
    "best_inner_battery_macro_rmse": 0.013740212833889608,
    "parameters": {
      "physics_weight": 0.01,
      "smooth_weight": 0.001
    },
    "inner_run_paths": [
      "C:\\Projects\\Battery_Soh\\benchmark-repository\\CALCE-CX2\\outputs\\rapct_prony\\staged\\checkpoints\\A4_CX2_38_CX2_16_s42_dc74618f7dc9",
      "C:\\Projects\\Battery_Soh\\benchmark-repository\\CALCE-CX2\\outputs\\rapct_prony\\staged\\checkpoints\\A4_CX2_38_CX2_33_s42_1912c8e5ada8"
    ],
    "trial_count": 2,
    "pilot": true,
    "outer_evaluated": false
  },
  "comparison_nonzero_coefficients": {
    "physics_weight": 0.01,
    "smooth_weight": 0.001
  },
  "outer_evaluated": false,
  "warmup_final": [
    10,
    30
  ],
  "warmup_search": [
    2,
    5
  ],
  "limitations": "Learned state/proxy consistency and smoothness are priors; no electrochemical PDE or calibrated mechanism."
}
```

D starts from an encoder-estimated unknown state, never an invented fresh-cell zero. Positive rates guarantee nonnegative increments only inside verified chronological segments. Carried state is detached across training chunks. The independent state-proposal residual excludes its tautological initialization equality. Observed SOH is not forced monotone. Per-epoch data/proxy/correction/variation terms and per-cycle latent audits are saved. The same architecture and 35-epoch budget are used for zero/nonzero penalties.
