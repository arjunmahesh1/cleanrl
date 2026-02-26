# PPO Damping Robustness (Raw Reward, 5 Seeds, Extended Grid)

This folder contains the damping-only evaluation snapshot.

## Experiment configuration
- Environment: `Walker2d-v4`
- Models: `vanilla`, `tv` (fixed cap from 200k AUC selection)
- Selected TV cap: `3500`
- Seeds: `1..5`
- Evaluation episodes per condition: `100`
- Reward reporting: raw rewards (`eval_raw_rewards=1`)
- Fixed perturbation dimensions:
  - `xml_body_mass_scale = 1.0`
  - `xml_geom_friction_scale = 1.0`
- Damping grid:
  - `{0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.3, 1.5, 1.7, 2.0}`

## Folder contents
- `inputs/selected_cap_from_auc_200k.txt`
- `outputs/eval_metrics_rawreward_damping_0p5_2p0.csv` (110 rows)
- `outputs/summary_rawreward_damping_0p5_2p0.csv`
- `outputs/gain_stats_rawreward_damping_0p5_2p0.csv`
- `outputs/per_seed/seed_*.csv` (22 rows per seed)
- `logs/summarize_damp_raw_10619539.out`

## Results by damping level (mean +/- 95% CI)

| Scenario | TV mean +/- CI | Vanilla mean +/- CI | Gain mean +/- CI | Gain median +/- CI | Gain IQM +/- CI |
| --- | --- | --- | --- | --- | --- |
| nominal | `3167.21 +/- 659.17` | `3046.23 +/- 704.05` | `-` | `-` | `-` |
| damping_0p5 | `3192.25 +/- 652.38` | `3185.45 +/- 674.04` | `-114.19 +/- 159.59` | `+42.97 +/- 283.61` | `-85.64 +/- 171.15` |
| damping_0p6 | `3190.74 +/- 652.17` | `3186.77 +/- 571.36` | `-117.01 +/- 220.30` | `-110.43 +/- 412.25` | `-75.51 +/- 52.96` |
| damping_0p7 | `3205.28 +/- 638.00` | `3157.14 +/- 721.55` | `-72.84 +/- 104.23` | `+78.21 +/- 207.87` | `-72.70 +/- 91.16` |
| damping_0p8 | `3120.56 +/- 780.52` | `3129.57 +/- 697.42` | `-129.99 +/- 218.65` | `-118.78 +/- 159.32` | `-177.86 +/- 94.56` |
| damping_0p9 | `3061.33 +/- 753.94` | `3169.50 +/- 699.72` | `-229.16 +/- 163.57` | `-191.79 +/- 273.26` | `-222.53 +/- 159.13` |
| damping_1p1 | `3190.38 +/- 685.65` | `3177.40 +/- 688.97` | `-108.00 +/- 125.83` | `-22.58 +/- 231.76` | `-116.75 +/- 156.43` |
| damping_1p3 | `3098.16 +/- 703.90` | `3056.42 +/- 649.96` | `-79.24 +/- 207.47` | `+2.72 +/- 308.75` | `-56.24 +/- 234.87` |
| damping_1p5 | `2997.98 +/- 707.06` | `3147.46 +/- 764.96` | `-270.46 +/- 193.99` | `-206.29 +/- 299.39` | `-304.19 +/- 177.06` |
| damping_1p7 | `3060.43 +/- 750.53` | `3088.56 +/- 773.63` | `-149.12 +/- 101.50` | `-87.36 +/- 177.19` | `-133.33 +/- 67.37` |
| damping_2p0 | `3094.30 +/- 677.79` | `3098.02 +/- 675.55` | `-124.70 +/- 136.53` | `-29.95 +/- 307.70` | `-107.32 +/- 16.76` |

## Robust gain summary (vanilla drop - TV drop; positive favors TV)
- Positive mean gain scenarios: `0 / 10`
- Positive IQM gain scenarios: `0 / 10`
- Mean gain across perturbed scenarios: `-139.47`
- Mean gain CI excludes zero in `3 / 10` scenarios
- IQM gain CI excludes zero in `6 / 10` scenarios

Interpretation: damping-only perturbations are currently adversarial for TV relative to vanilla in this setup. TV is higher nominally, but under damping changes it degrades more in drop-based robustness.
