# Walker2d PPO Full 30-Seed No-Noop: gaussian_action_noise

Date: 2026-06-10

This folder packages the final pinned robustness evaluation for normalized PPO.

## Contents

- Raw metrics: `/home/users/am1015/cleanrl/sweeps/results/PPO_full30_allcaps_no_noop_20260524/Walker2d/gaussian_action_noise/raw_metrics`
- Aggregated outputs: `/home/users/am1015/cleanrl/sweeps/results/PPO_full30_allcaps_no_noop_20260524/Walker2d/gaussian_action_noise/outputs`
- Plots: `/home/users/am1015/cleanrl/sweeps/results/PPO_full30_allcaps_no_noop_20260524/Walker2d/gaussian_action_noise/plots`

## Evaluation protocol

- Models are compared on the same perturbation grid for the configured axes.
- Nominal reference within each axis is the `factor=0.0` point.
- Curves show mean return across seeds with `95% CI` shading.
- Robust gain is defined as `vanilla_drop - model_drop`; positive is better.
- Plot files are exported in both `PNG` and vector `PDF` format.

## Model labels

- `vanilla` -> Vanilla
- `a2p85` -> a2p85
- `a2p95` -> a2p95
- `a3p00` -> a3p00
- `a3p05` -> TV cap=3.05
- `a3p10` -> a3p10
- `a3p20` -> a3p20
- `a3p50` -> a3p50
- `a3p70` -> a3p70
- `a4p00` -> a4p00

## Nominal returns by axis

| Axis | Vanilla | a2p85 | a2p95 | a3p00 | TV cap=3.05 | a3p10 | a3p20 | a3p50 | a3p70 | a4p00 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| action_noise | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

## Axis overview

| Axis | Model | Nominal return | Mean gain over perturbed scenarios | Positive gain scenarios |
| --- | --- | --- | --- | --- |

## Plot files

- `plots/with_variance/`: full plot set with variance whiskers.
- `plots/without_variance/`: matching plot set without variance whiskers.
- `plots/with_variance/return_curves_panel.png`
- `plots/with_variance/return_curves_panel.pdf`
- `plots/with_variance/gain_curves_panel.png`
- `plots/with_variance/gain_curves_panel.pdf`
- `plots/without_variance/return_curves_panel.png`
- `plots/without_variance/return_curves_panel.pdf`
- `plots/without_variance/gain_curves_panel.png`
- `plots/without_variance/gain_curves_panel.pdf`
- `plots/with_variance/action_noise_return_curve.png`
- `plots/with_variance/action_noise_return_curve.pdf`
- `plots/with_variance/action_noise_gain_curve.png`
- `plots/with_variance/action_noise_gain_curve.pdf`
- `plots/without_variance/action_noise_return_curve.png`
- `plots/without_variance/action_noise_return_curve.pdf`
- `plots/without_variance/action_noise_gain_curve.png`
- `plots/without_variance/action_noise_gain_curve.pdf`

## Output CSV files

- `outputs/eval_metrics_final.csv`: latest merged per-seed eval rows.
- `outputs/summary_by_scenario.csv`: per-model, per-axis, per-factor aggregate return table.
- `outputs/drop_summary.csv`: nominal-minus-perturbed drop table.
- `outputs/gain_summary.csv`: paired robust-gain table against vanilla.
- `outputs/axis_overview.csv`: compact axis-level overview.
- `outputs/curve_points.csv`: same data used for return plots.
- `outputs/gain_curve_points.csv`: same data used for gain plots.
