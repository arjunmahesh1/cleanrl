# Experiment Registry (Walker2d PPO)

This directory is the running registry of completed experiment snapshots.

## Completed experiments

| Experiment | Seeds | Perturbation setup | Main result (mean +/- 95% CI) | Verdict | Files |
| --- | ---: | --- | --- | --- | --- |
| Single-seed random perturbation smoke (TV90 era) | 1 | Random XML scales; highlighted candidate `mass=1.1, friction=0.9, damping=1.2` | TV: `2181.88 -> 5332.22` (`+3150.34`), Vanilla: `3855.31 -> 6188.46` (`+2333.15`) | Positive single-seed signal, not conclusive | `sweeps/results/random_perturbations/README.md` |
| PPO friction-only robustness (raw reward) | 5 | `mass=1.0`, `damping=1.0`, friction grid `{1.0,1.1,1.2,1.3}` | Nominal: TV `3137.24 +/- 683.34`, Vanilla `3155.19 +/- 730.20` | Mixed gains; no clear consistent advantage at 5 seeds | `sweeps/results/PPO_Friction_rawreward_5seed_20260224/README.md` |
| PPO friction-only robustness (raw reward, extended grid) | 5 | `mass=1.0`, `damping=1.0`, friction grid `{0.5,0.6,0.7,0.8,0.9,1.1,1.3,1.5,1.7,2.0}` | Nominal: TV `3144.68 +/- 721.99`, Vanilla `3084.97 +/- 688.99` | TV absolute return is close to vanilla, but robustness gain is not consistently positive | `sweeps/results/PPO_Friction_rawreward_5seed_0p5_2p0_20260225/README.md` |

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

## Canonical files

- Current canonical aggregated files in `sweeps/`:
  - `sweeps/ppo_robust_eval_metrics_friction_0p5_2p0.csv`
  - `sweeps/ppo_robust_summary_friction_0p5_2p0.csv`
  - `sweeps/ppo_robust_gain_all_stats_friction_0p5_2p0.csv`
- Frozen snapshots:
  - `sweeps/results/PPO_Friction_rawreward_5seed_20260224/`
  - `sweeps/results/PPO_Friction_rawreward_5seed_0p5_2p0_20260225/`
