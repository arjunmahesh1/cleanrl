# PPO_HalfCheetah_signal_20260426_v2

Date: 2026-04-27

This folder packages the final pinned robustness evaluation for normalized PPO.

## Contents

- Raw metrics: `/home/users/am1015/cleanrl/sweeps/results/PPO_HalfCheetah_signal_20260426_v2/raw_metrics`
- Aggregated outputs: `/home/users/am1015/cleanrl/sweeps/results/PPO_HalfCheetah_signal_20260426_v2/outputs`
- Plots: `/home/users/am1015/cleanrl/sweeps/results/PPO_HalfCheetah_signal_20260426_v2/plots`

## Evaluation protocol

- Models are compared on the same perturbation grid for the configured axes.
- Nominal reference within each axis is the `factor=1.0` point.
- Curves show mean return across seeds with `95% CI` shading.
- Robust gain is defined as `vanilla_drop - model_drop`; positive is better.

## Model labels

- `vanilla` -> Vanilla
- `a1e9` -> No-op (1e9)
- `a2p70` -> TV cap=2.70
- `a3p00` -> TV cap=3.00
- `a3p05` -> TV cap=3.05
- `a3p10` -> TV cap=3.10
- `a3p20` -> TV cap=3.20
- `a3p40` -> TV cap=3.40
- `a3p70` -> TV cap=3.70

## Nominal returns by axis

| Axis | Vanilla | No-op (1e9) | TV cap=2.70 | TV cap=3.00 | TV cap=3.05 | TV cap=3.10 | TV cap=3.20 | TV cap=3.40 | TV cap=3.70 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| action_noise | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| action_noise_bernoulli | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| state_noise | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

## Axis overview

| Axis | Model | Nominal return | Mean gain over perturbed scenarios | Positive gain scenarios |
| --- | --- | --- | --- | --- |

## Plot files

- `plots/return_curves_panel.png`
- `plots/gain_curves_panel.png`
- `plots/action_noise_return_curve.png`
- `plots/action_noise_gain_curve.png`
- `plots/action_noise_bernoulli_return_curve.png`
- `plots/action_noise_bernoulli_gain_curve.png`
- `plots/state_noise_return_curve.png`
- `plots/state_noise_gain_curve.png`

## Output CSV files

- `outputs/eval_metrics_final.csv`: latest merged per-seed eval rows.
- `outputs/summary_by_scenario.csv`: per-model, per-axis, per-factor aggregate return table.
- `outputs/drop_summary.csv`: nominal-minus-perturbed drop table.
- `outputs/gain_summary.csv`: paired robust-gain table against vanilla.
- `outputs/axis_overview.csv`: compact axis-level overview.
- `outputs/curve_points.csv`: same data used for return plots.
- `outputs/gain_curve_points.csv`: same data used for gain plots.
