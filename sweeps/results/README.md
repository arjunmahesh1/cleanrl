# Experiment Registry (Walker2d PPO)

This directory is the running registry of completed experiment snapshots.

## Methodology setup snapshots

| Setup | Purpose | Files |
| --- | --- | --- |
| Percentile fixed-cap Phase 1 setup | Shared TV caps derived from converged vanilla PPO returns (last 25% of training) for candidate percentiles `75/85/90/95` | `sweeps/results/PPO_PercentileFixedCap_setup_20260226/README.md` |

## Completed experiments

| Experiment | Seeds | Perturbation setup | Main result (mean +/- 95% CI) | Verdict | Files |
| --- | ---: | --- | --- | --- | --- |
| Single-seed random perturbation smoke (TV90 era) | 1 | Random XML scales; highlighted candidate `mass=1.1, friction=0.9, damping=1.2` | TV: `2181.88 -> 5332.22` (`+3150.34`), Vanilla: `3855.31 -> 6188.46` (`+2333.15`) | Positive single-seed signal, not conclusive | `sweeps/results/random_perturbations/README.md` |
| PPO friction-only robustness (raw reward) | 5 | `mass=1.0`, `damping=1.0`, friction grid `{1.0,1.1,1.2,1.3}` | Nominal: TV `3137.24 +/- 683.34`, Vanilla `3155.19 +/- 730.20` | Mixed gains; no clear consistent advantage at 5 seeds | `sweeps/results/PPO_Friction_rawreward_5seed_20260224/README.md` |
| PPO friction-only robustness (raw reward, extended grid) | 5 | `mass=1.0`, `damping=1.0`, friction grid `{0.5,0.6,0.7,0.8,0.9,1.1,1.3,1.5,1.7,2.0}` | Nominal: TV `3144.68 +/- 721.99`, Vanilla `3084.97 +/- 688.99` | TV absolute return is close to vanilla, but robustness gain is not consistently positive | `sweeps/results/PPO_Friction_rawreward_5seed_0p5_2p0_20260225/README.md` |
| PPO mass-only robustness (raw reward, extended grid) | 5 | `friction=1.0`, `damping=1.0`, mass grid `{0.5,0.6,0.7,0.8,0.9,1.1,1.3,1.5,1.7,2.0}` | Nominal: TV `3090.89 +/- 767.69`, Vanilla `3143.11 +/- 603.72` | Mild positive trend for TV under mass shifts, but wide CIs at 5 seeds | `sweeps/results/PPO_Mass_rawreward_5seed_0p5_2p0_20260226/README.md` |
| PPO damping-only robustness (raw reward, extended grid) | 5 | `mass=1.0`, `friction=1.0`, damping grid `{0.5,0.6,0.7,0.8,0.9,1.1,1.3,1.5,1.7,2.0}` | Nominal: TV `3167.21 +/- 659.17`, Vanilla `3046.23 +/- 704.05` | TV has higher nominal return, but robustness gains are consistently negative under damping shifts | `sweeps/results/PPO_Damping_rawreward_5seed_0p5_2p0_20260226/README.md` |

## Friction-only result table (extended grid canonical run)

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

Extended-grid summary:
- Positive gain scenarios (mean): `2 / 10` (`friction_0p7`, `friction_0p9`)
- Positive gain scenarios (IQM): `2 / 10`
- Mean gain over perturbed scenarios: `-47.29`

## Mass-only result table (extended grid canonical run)

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

Extended-grid summary:
- Positive gain scenarios (mean): `7 / 10`
- Positive gain scenarios (IQM): `8 / 10`
- Mean gain over perturbed scenarios: `+30.03`
- Mean-gain CI excludes zero in `0 / 10` scenarios (effect direction still uncertain at 5 seeds)

## Damping-only result table (extended grid canonical run)

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

Extended-grid summary:
- Positive gain scenarios (mean): `0 / 10`
- Positive gain scenarios (IQM): `0 / 10`
- Mean gain over perturbed scenarios: `-139.47`
- Mean-gain CI excludes zero in `3 / 10` scenarios
- IQM-gain CI excludes zero in `6 / 10` scenarios

## Canonical files

- Current canonical aggregated files in `sweeps/`:
  - `sweeps/ppo_robust_eval_metrics_friction_0p5_2p0.csv`
  - `sweeps/ppo_robust_summary_friction_0p5_2p0.csv`
  - `sweeps/ppo_robust_gain_all_stats_friction_0p5_2p0.csv`
  - `sweeps/ppo_robust_eval_metrics_mass_0p5_2p0.csv`
  - `sweeps/ppo_robust_summary_mass_0p5_2p0.csv`
  - `sweeps/ppo_robust_gain_all_stats_mass_0p5_2p0.csv`
  - `sweeps/ppo_robust_eval_metrics_damping_0p5_2p0.csv`
  - `sweeps/ppo_robust_summary_damping_0p5_2p0.csv`
  - `sweeps/ppo_robust_gain_all_stats_damping_0p5_2p0.csv`
- Frozen snapshots:
  - `sweeps/results/PPO_Friction_rawreward_5seed_20260224/`
  - `sweeps/results/PPO_Friction_rawreward_5seed_0p5_2p0_20260225/`
  - `sweeps/results/PPO_Mass_rawreward_5seed_0p5_2p0_20260226/`
  - `sweeps/results/PPO_Damping_rawreward_5seed_0p5_2p0_20260226/`
