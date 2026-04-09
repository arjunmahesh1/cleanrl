# PPO Alpha Grid - Final Pinned Robustness Evaluation

Date: 2026-04-08

This folder packages the final pinned robustness evaluation for normalized PPO.

## Contents

- Raw metrics: `/Users/arjunmahesh/Library/CloudStorage/OneDrive-Personal/Duke/Research/RL/cleanrl/sweeps/results/PPO_AlphaGridNorm_final_eval_20260329/raw_metrics`
- Aggregated outputs: `/Users/arjunmahesh/Library/CloudStorage/OneDrive-Personal/Duke/Research/RL/cleanrl/sweeps/results/PPO_AlphaGridNorm_final_eval_20260329/outputs`
- Plots: `/Users/arjunmahesh/Library/CloudStorage/OneDrive-Personal/Duke/Research/RL/cleanrl/sweeps/results/PPO_AlphaGridNorm_final_eval_20260329/plots`

## Evaluation protocol

- Models are compared on the same perturbation grid for the configured axes.
- Nominal reference within each axis is the `factor=1.0` point.
- Curves show mean return across seeds with `95% CI` shading.
- Robust gain is defined as `vanilla_drop - model_drop`; positive is better.

## Model labels

- `vanilla` -> Vanilla
- `a1e9` -> No-op (1e9)
- `a3p05` -> TV cap=3.05

## Nominal returns by axis

| Axis | Vanilla | No-op (1e9) | TV cap=3.05 |
| --- | --- | --- | --- |
| damping | 2830.42 +/- 392.26 | 2740.62 +/- 356.00 | 2671.87 +/- 710.80 |
| friction | 2801.26 +/- 463.56 | 2858.93 +/- 424.57 | 2692.18 +/- 705.66 |
| mass | 2932.26 +/- 680.85 | 2894.05 +/- 540.81 | 2568.53 +/- 546.45 |

## Axis overview

| Axis | Model | Nominal return | Mean gain over perturbed scenarios | Positive gain scenarios |
| --- | --- | --- | --- | --- |
| damping | No-op (1e9) | 2740.62 +/- 356.00 | +161.51 | 8/10 |
| damping | TV cap=3.05 | 2671.87 +/- 710.80 | +13.44 | 5/10 |
| friction | No-op (1e9) | 2858.93 +/- 424.57 | -42.87 | 2/10 |
| friction | TV cap=3.05 | 2692.18 +/- 705.66 | +89.99 | 6/10 |
| mass | No-op (1e9) | 2894.05 +/- 540.81 | +51.08 | 6/10 |
| mass | TV cap=3.05 | 2568.53 +/- 546.45 | +286.18 | 9/10 |

## Plot files

- `plots/return_curves_panel.png`
- `plots/gain_curves_panel.png`
- `plots/<axis>_variance_boxplot.png`: per-seed return spread across caps at selected factors.
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
