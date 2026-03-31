# PPO Alpha Grid - Multi-cap Robustness Evaluation

Date: 2026-03-31

This folder packages the final pinned robustness evaluation for normalized PPO.

## Contents

- Raw metrics: `C:/Users/Arjun/OneDrive/Duke/Research/RL/cleanrl/sweeps/results/PPO_AlphaGridNorm_multicap_eval_20260330/raw_metrics`
- Aggregated outputs: `C:/Users/Arjun/OneDrive/Duke/Research/RL/cleanrl/sweeps/results/PPO_AlphaGridNorm_multicap_eval_20260330/outputs`
- Plots: `C:/Users/Arjun/OneDrive/Duke/Research/RL/cleanrl/sweeps/results/PPO_AlphaGridNorm_multicap_eval_20260330/plots`

## Evaluation protocol

- Models are compared on the same perturbation grid for `friction`, `mass`, and `damping`.
- Nominal reference within each axis is the `factor=1.0` point.
- Curves show mean return across seeds with `95% CI` shading.
- Robust gain is defined as `vanilla_drop - model_drop`; positive is better.

## Model labels

- `vanilla` -> Vanilla
- `a1e9` -> No-op (1e9)
- `a2p95` -> TV cap=2.95
- `a3p05` -> TV cap=3.05

## Nominal returns by axis

| Axis | Vanilla | No-op (1e9) | TV cap=2.95 | TV cap=3.05 |
| --- | --- | --- | --- | --- |
| damping | 2861.93 +/- 703.87 | 2814.89 +/- 540.19 | 3094.57 +/- 953.27 | 2400.00 +/- 571.46 |
| friction | 2984.94 +/- 810.65 | 3042.52 +/- 729.30 | 2674.42 +/- 749.14 | 2773.93 +/- 698.21 |
| mass | 2900.30 +/- 469.84 | 2932.82 +/- 676.08 | 2586.72 +/- 677.98 | 2844.77 +/- 538.81 |

## Axis overview

| Axis | Model | Nominal return | Mean gain over perturbed scenarios | Positive gain scenarios |
| --- | --- | --- | --- | --- |
| damping | No-op (1e9) | 2814.89 +/- 540.19 | -36.29 | 4/10 |
| damping | TV cap=2.95 | 3094.57 +/- 953.27 | -225.92 | 0/10 |
| damping | TV cap=3.05 | 2400.00 +/- 571.46 | +212.36 | 10/10 |
| friction | No-op (1e9) | 3042.52 +/- 729.30 | -90.13 | 2/10 |
| friction | TV cap=2.95 | 2674.42 +/- 749.14 | +728.87 | 10/10 |
| friction | TV cap=3.05 | 2773.93 +/- 698.21 | +243.26 | 10/10 |
| mass | No-op (1e9) | 2932.82 +/- 676.08 | +9.81 | 6/10 |
| mass | TV cap=2.95 | 2586.72 +/- 677.98 | +406.44 | 10/10 |
| mass | TV cap=3.05 | 2844.77 +/- 538.81 | -118.25 | 1/10 |

## Plot files

- `plots/return_curves_panel.png`
- `plots/gain_curves_panel.png`
- `plots/damping_return_curve.png`
- `plots/damping_gain_curve.png`
- `plots/friction_return_curve.png`
- `plots/friction_gain_curve.png`
- `plots/mass_return_curve.png`
- `plots/mass_gain_curve.png`

## Output CSV files

- `outputs/eval_metrics_final.csv`: latest merged per-seed eval rows.
- `outputs/summary_by_scenario.csv`: per-model, per-axis, per-factor aggregate return table.
- `outputs/drop_summary.csv`: nominal-minus-perturbed drop table.
- `outputs/gain_summary.csv`: paired robust-gain table against vanilla.
- `outputs/axis_overview.csv`: compact axis-level overview.
- `outputs/curve_points.csv`: same data used for return plots.
- `outputs/gain_curve_points.csv`: same data used for gain plots.
