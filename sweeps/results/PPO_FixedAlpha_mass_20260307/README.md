# PPO Fixed-Alpha Mass Sweep (Coarse + Fine, Raw Reward)

This snapshot consolidates the fixed-alpha mass-axis comparison against `vanilla` and `tv_p95`.

## Setup

- Environment: `Walker2d-v4`
- Seeds: `5`
- Mass grid: `{0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.3,1.5,1.7,2.0}`
- Friction and damping held fixed at `1.0`
- Models:
  - `vanilla`
  - `tv_p95`
  - fixed-alpha candidates

Raw-reward validation:
- `sweeps/ppo_compare_alpha_mass_coarse_metrics.csv`: `eval_raw_rewards=1`
- `sweeps/ppo_compare_alpha_mass_fine_metrics.csv`: `eval_raw_rewards=1`

## Candidate sets

- Coarse: `a3200, a3600, a4000, a4400, a4800`
- Fine: `a3800, a3900, a4000, a4100, a4200`

## Main result

Best fixed-alpha candidate on this axis is `a4000` in both sweeps:

- Coarse AUC ranking:
  - `tv_a4000 = 4897.43`
  - `vanilla = 4758.89`
  - `tv_p95 = 4178.48`
- Fine AUC ranking:
  - `tv_a4000 = 4906.62`
  - `vanilla = 4693.78`
  - `tv_p95 = 4252.73`

Nominal point (`mass=1.0`):
- Coarse: `tv_a4000 3273.26 +/- 590.03`, `vanilla 3135.30 +/- 690.65`, `tv_p95 2739.77 +/- 444.05`
- Fine: `tv_a4000 3343.47 +/- 680.88`, `vanilla 3112.86 +/- 564.75`, `tv_p95 2824.56 +/- 489.20`

## Files

- Coarse outputs: `sweeps/results/PPO_FixedAlpha_mass_20260307/coarse/`
- Fine outputs: `sweeps/results/PPO_FixedAlpha_mass_20260307/fine/`
- Aggregated metrics:
  - `sweeps/ppo_compare_alpha_mass_coarse_metrics.csv`
  - `sweeps/ppo_compare_alpha_mass_fine_metrics.csv`
