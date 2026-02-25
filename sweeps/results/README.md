# Experiment Registry (Walker2d PPO)

This directory is the running registry of completed experiment snapshots.

## Completed experiments

| Experiment | Seeds | Perturbation setup | Main result (mean +/- 95% CI) | Verdict | Files |
| --- | ---: | --- | --- | --- | --- |
| Single-seed random perturbation smoke (TV90 era) | 1 | Random XML scales; highlighted candidate `mass=1.1, friction=0.9, damping=1.2` | TV: `2181.88 -> 5332.22` (`+3150.34`), Vanilla: `3855.31 -> 6188.46` (`+2333.15`) | Positive single-seed signal, not conclusive | `sweeps/results/random_perturbations/README.md` |
| PPO friction-only robustness (raw reward) | 5 | `mass=1.0`, `damping=1.0`, friction grid `{1.0,1.1,1.2,1.3}` | Nominal: TV `3137.24 +/- 683.34`, Vanilla `3155.19 +/- 730.20` | Mixed gains; no clear consistent advantage at 5 seeds | `sweeps/results/PPO_Friction_rawreward_5seed_20260224/README.md` |

## Friction-only result table (current canonical run)

| Scenario | TV mean +/- CI | Vanilla mean +/- CI | Gain mean (vanilla drop - TV drop) |
| --- | --- | --- | --- |
| nominal | `3137.24 +/- 683.34` | `3155.19 +/- 730.20` | `-` |
| friction_1p0 | `3153.78 +/- 804.18` | `3049.49 +/- 707.00` | `+122.25 +/- 341.88` |
| friction_1p1 | `3162.17 +/- 757.72` | `3225.74 +/- 841.56` | `-45.61 +/- 247.91` |
| friction_1p2 | `2732.18 +/- 585.37` | `2700.01 +/- 573.32` | `+50.12 +/- 224.70` |
| friction_1p3 | `2400.87 +/- 751.03` | `2317.54 +/- 723.63` | `+101.28 +/- 134.18` |

## Canonical files

- Current canonical aggregated files in `sweeps/`:
  - `sweeps/ppo_robust_eval_metrics.csv`
  - `sweeps/ppo_robust_summary.csv`
  - `sweeps/ppo_robust_gain_all_stats.csv`
- Frozen snapshot for the friction run:
  - `sweeps/results/PPO_Friction_rawreward_5seed_20260224/`
