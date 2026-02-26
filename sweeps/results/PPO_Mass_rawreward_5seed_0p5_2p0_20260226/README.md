# PPO Mass Robustness (Raw Reward, 5 Seeds, Extended Grid)

This folder contains the mass-only evaluation snapshot.

## Experiment configuration
- Environment: `Walker2d-v4`
- Models: `vanilla`, `tv` (fixed cap from 200k AUC selection)
- Selected TV cap: `3500`
- Seeds: `1..5`
- Evaluation episodes per condition: `100`
- Reward reporting: raw rewards (`eval_raw_rewards=1`)
- Fixed perturbation dimensions:
  - `xml_geom_friction_scale = 1.0`
  - `xml_joint_damping_scale = 1.0`
- Mass grid:
  - `{0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.3, 1.5, 1.7, 2.0}`

## Folder contents
- `inputs/selected_cap_from_auc_200k.txt`
- `outputs/eval_metrics_rawreward_mass_0p5_2p0.csv` (110 rows)
- `outputs/summary_rawreward_mass_0p5_2p0.csv`
- `outputs/gain_stats_rawreward_mass_0p5_2p0.csv`
- `outputs/per_seed/seed_*.csv` (22 rows per seed)
- `logs/eval_mass_raw_10619425_*.out`
- `logs/summarize_mass_raw_10619426.out`

## Results by mass level (mean +/- 95% CI)

| Scenario | TV mean +/- CI | Vanilla mean +/- CI | Gain mean +/- CI | Gain median +/- CI | Gain IQM +/- CI |
| --- | --- | --- | --- | --- | --- |
| nominal | `3090.89 +/- 767.69` | `3143.11 +/- 603.72` | `-` | `-` | `-` |
| mass_0p5 | `3179.10 +/- 683.40` | `3088.05 +/- 634.19` | `+143.27 +/- 212.89` | `+312.03 +/- 288.71` | `+203.42 +/- 146.35` |
| mass_0p6 | `3138.49 +/- 685.84` | `3198.41 +/- 699.50` | `-7.71 +/- 305.31` | `+139.42 +/- 357.36` | `+50.27 +/- 304.73` |
| mass_0p7 | `3079.93 +/- 693.53` | `3124.04 +/- 704.32` | `+8.11 +/- 326.72` | `+82.59 +/- 403.56` | `+74.76 +/- 274.33` |
| mass_0p8 | `3136.47 +/- 784.38` | `3184.09 +/- 753.15` | `+4.59 +/- 424.93` | `+73.81 +/- 628.10` | `-6.45 +/- 426.36` |
| mass_0p9 | `3128.24 +/- 675.14` | `3077.87 +/- 596.04` | `+102.59 +/- 268.31` | `+188.99 +/- 147.40` | `+111.42 +/- 165.82` |
| mass_1p1 | `3127.04 +/- 720.11` | `3166.04 +/- 762.39` | `+13.21 +/- 245.01` | `+259.51 +/- 275.47` | `+29.59 +/- 173.20` |
| mass_1p3 | `3133.13 +/- 681.94` | `3120.73 +/- 666.77` | `+64.61 +/- 312.99` | `+323.09 +/- 313.43` | `+147.71 +/- 73.90` |
| mass_1p5 | `3070.02 +/- 654.34` | `3200.27 +/- 710.77` | `-78.03 +/- 334.91` | `-27.60 +/- 460.34` | `-44.54 +/- 297.36` |
| mass_1p7 | `3034.06 +/- 727.23` | `3141.59 +/- 705.31` | `-55.31 +/- 207.56` | `+57.25 +/- 249.74` | `+11.94 +/- 161.13` |
| mass_2p0 | `3149.67 +/- 709.91` | `3096.89 +/- 646.91` | `+104.99 +/- 296.81` | `+198.24 +/- 164.18` | `+132.01 +/- 316.28` |

## Robust gain summary (vanilla drop - TV drop; positive favors TV)
- Positive mean gain scenarios: `7 / 10`
- Positive IQM gain scenarios: `8 / 10`
- Mean gain across perturbed scenarios: `+30.03`
- Mean gain CI excludes zero in `0 / 10` scenarios

Interpretation: under mass-only perturbations, TV trends slightly better in drop-based robustness than vanilla, but uncertainty remains high at 5 seeds.
