# PPO_HalfCheetah_lower_bound_signal_20260429

Date: 2026-04-29

This folder packages the final pinned robustness evaluation for normalized PPO.

## Contents

- Raw metrics: `/Users/arjunmahesh/Library/CloudStorage/OneDrive-Personal/Duke/Research/RL/cleanrl/sweeps/results/PPO_HalfCheetah_lower_bound_signal_20260429/raw_metrics`
- Aggregated outputs: `/Users/arjunmahesh/Library/CloudStorage/OneDrive-Personal/Duke/Research/RL/cleanrl/sweeps/results/PPO_HalfCheetah_lower_bound_signal_20260429/outputs`
- Plots: `/Users/arjunmahesh/Library/CloudStorage/OneDrive-Personal/Duke/Research/RL/cleanrl/sweeps/results/PPO_HalfCheetah_lower_bound_signal_20260429/plots`

## Evaluation protocol

- Models are compared on the same perturbation grid for the configured axes.
- Nominal reference within each axis is the `factor=0.0` point.
- Curves show mean return across seeds with `95% CI` shading.
- Robust gain is defined as `vanilla_drop - model_drop`; positive is better.

## Model labels

- `vanilla` -> Vanilla

## Nominal returns by axis

| Axis | Vanilla |
| --- | --- |
| action_noise | 1826.48 +/- 733.58 |
| action_noise_bernoulli | 1844.17 +/- 819.74 |
| state_noise | 1845.90 +/- 787.76 |

## Axis overview

| Axis | Model | Nominal return | Mean gain over perturbed scenarios | Positive gain scenarios |
| --- | --- | --- | --- | --- |

## Plot files

- `plots/with_variance/`: full plot set with variance whiskers.
- `plots/without_variance/`: matching plot set without variance whiskers.
- `plots/with_variance/return_curves_panel.png`
- `plots/with_variance/gain_curves_panel.png`
- `plots/without_variance/return_curves_panel.png`
- `plots/without_variance/gain_curves_panel.png`
- `plots/with_variance/action_noise_return_curve.png`
- `plots/with_variance/action_noise_gain_curve.png`
- `plots/without_variance/action_noise_return_curve.png`
- `plots/without_variance/action_noise_gain_curve.png`
- `plots/with_variance/action_noise_bernoulli_return_curve.png`
- `plots/with_variance/action_noise_bernoulli_gain_curve.png`
- `plots/without_variance/action_noise_bernoulli_return_curve.png`
- `plots/without_variance/action_noise_bernoulli_gain_curve.png`
- `plots/with_variance/state_noise_return_curve.png`
- `plots/with_variance/state_noise_gain_curve.png`
- `plots/without_variance/state_noise_return_curve.png`
- `plots/without_variance/state_noise_gain_curve.png`

## Output CSV files

- `outputs/eval_metrics_final.csv`: latest merged per-seed eval rows.
- `outputs/summary_by_scenario.csv`: per-model, per-axis, per-factor aggregate return table.
- `outputs/drop_summary.csv`: nominal-minus-perturbed drop table.
- `outputs/gain_summary.csv`: paired robust-gain table against vanilla.
- `outputs/axis_overview.csv`: compact axis-level overview.
- `outputs/curve_points.csv`: same data used for return plots.
- `outputs/gain_curve_points.csv`: same data used for gain plots.
