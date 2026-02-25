# PPO Friction Robustness (Raw Reward, 5 Seeds, Extended Grid)

This folder contains the extended friction-only evaluation snapshot.

## Experiment configuration
- Environment: `Walker2d-v4`
- Models: `vanilla`, `tv` (fixed cap from 200k AUC selection)
- Selected TV cap: `3500`
- Seeds: `1..5`
- Evaluation episodes per condition: `100`
- Reward reporting: raw rewards (`eval_raw_rewards=1`)
- Fixed perturbation dimensions:
  - `xml_body_mass_scale = 1.0`
  - `xml_joint_damping_scale = 1.0`
- Friction grid:
  - `{0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.3, 1.5, 1.7, 2.0}`

## Folder contents
- `inputs/selected_cap_from_auc_200k.txt`
- `outputs/eval_metrics_rawreward_friction_0p5_2p0.csv` (110 rows)
- `outputs/summary_rawreward_friction_0p5_2p0.csv`
- `outputs/gain_stats_rawreward_friction_0p5_2p0.csv`
- `outputs/per_seed/seed_*.csv` (22 rows per seed)

## Results by friction level (mean +/- 95% CI)

| Scenario | TV mean +/- CI | Vanilla mean +/- CI | Gain mean +/- CI | Gain median +/- CI | Gain IQM +/- CI |
| --- | --- | --- | --- | --- | --- |
| nominal | `3144.68 +/- 721.99` | `3084.97 +/- 688.99` | `-` | `-` | `-` |
| friction_0p5 | `1303.35 +/- 592.99` | `1292.61 +/- 581.71` | `-48.96 +/- 96.09` | `-55.04 +/- 121.31` | `-66.78 +/- 68.91` |
| friction_0p6 | `1813.98 +/- 607.03` | `1784.98 +/- 575.21` | `-30.70 +/- 59.58` | `-48.83 +/- 94.57` | `-53.14 +/- 20.40` |
| friction_0p7 | `2259.52 +/- 552.03` | `2185.95 +/- 515.28` | `+13.86 +/- 142.32` | `+30.36 +/- 255.69` | `+20.16 +/- 152.16` |
| friction_0p8 | `2605.38 +/- 559.57` | `2695.64 +/- 486.95` | `-149.97 +/- 140.38` | `-165.59 +/- 230.06` | `-161.89 +/- 157.85` |
| friction_0p9 | `2952.74 +/- 510.24` | `2873.73 +/- 611.40` | `+19.31 +/- 131.40` | `+136.55 +/- 156.35` | `+16.92 +/- 99.03` |
| friction_1p1 | `3166.67 +/- 772.65` | `3177.13 +/- 715.61` | `-70.17 +/- 154.83` | `-82.07 +/- 297.52` | `-72.68 +/- 182.94` |
| friction_1p3 | `2341.97 +/- 759.87` | `2319.92 +/- 789.93` | `-37.65 +/- 137.42` | `-59.55 +/- 116.82` | `-59.51 +/- 122.40` |
| friction_1p5 | `1597.49 +/- 649.04` | `1614.47 +/- 607.13` | `-76.69 +/- 160.06` | `-2.57 +/- 193.16` | `-120.53 +/- 118.10` |
| friction_1p7 | `1039.19 +/- 330.16` | `1013.91 +/- 304.66` | `-34.41 +/- 77.06` | `-63.95 +/- 82.20` | `-37.58 +/- 82.93` |
| friction_2p0 | `559.95 +/- 226.85` | `557.81 +/- 222.15` | `-57.56 +/- 69.25` | `-61.89 +/- 101.39` | `-57.74 +/- 73.90` |

## Robust gain summary (vanilla drop - TV drop; positive favors TV)
- Positive mean gain scenarios: `2 / 10` (`friction_0p7`, `friction_0p9`)
- Positive IQM gain scenarios: `2 / 10` (`friction_0p7`, `friction_0p9`)
- Mean gain across perturbed scenarios: `-47.29`

Interpretation: over friction `0.5..2.0`, TV and vanilla remain close in absolute return, and drop-based robustness advantage is not consistently positive at 5 seeds.
