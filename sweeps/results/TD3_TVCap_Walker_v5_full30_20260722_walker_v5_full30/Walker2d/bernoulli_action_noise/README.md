# Walker2d TD3 TV-Cap 30-Seed: Bernoulli Action Noise

Date: 2026-07-25

This folder packages the final pinned robustness evaluation for normalized PPO.

## Contents

- Raw metrics: `/home/users/am1015/cleanrl/sweeps/results/TD3_TVCap_Walker_v5_full30_20260722_walker_v5_full30/Walker2d/bernoulli_action_noise/raw_metrics`
- Aggregated outputs: `/home/users/am1015/cleanrl/sweeps/results/TD3_TVCap_Walker_v5_full30_20260722_walker_v5_full30/Walker2d/bernoulli_action_noise/outputs`
- Plots: `/home/users/am1015/cleanrl/sweeps/results/TD3_TVCap_Walker_v5_full30_20260722_walker_v5_full30/Walker2d/bernoulli_action_noise/plots`

## Evaluation protocol

- Models are compared on the same perturbation grid for the configured axes.
- Nominal reference within each axis is the `factor=0.0` point.
- Curves show mean return across seeds with `95% CI` shading.
- Robust gain is defined as `vanilla_drop - model_drop`; positive is better.
- Plot files are exported in both `PNG` and vector `PDF` format.

## Model labels

- `vanilla` -> Vanilla
- `tvc100` -> TV c=100
- `tvc150` -> TV c=150
- `tvc200` -> TV c=200
- `tvc225` -> TV c=225
- `tvc250` -> TV c=250
- `tvc275` -> TV c=275
- `tvc300` -> TV c=300
- `tvc400` -> TV c=400
- `tvc500` -> TV c=500

## Nominal returns by axis

| Axis | Vanilla | TV c=100 | TV c=150 | TV c=200 | TV c=225 | TV c=250 | TV c=275 | TV c=300 | TV c=400 | TV c=500 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| action_replace | 3981.98 +/- 258.76 | 598.09 +/- 34.33 | 839.87 +/- 83.53 | 1594.87 +/- 196.62 | 2108.59 +/- 306.52 | 2955.20 +/- 277.81 | 3334.76 +/- 211.70 | 3635.51 +/- 121.71 | 4020.71 +/- 254.35 | 3722.82 +/- 300.79 |

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
- `plots/with_variance/action_replace_return_curve.png`
- `plots/with_variance/action_replace_return_curve.pdf`
- `plots/with_variance/action_replace_gain_curve.png`
- `plots/with_variance/action_replace_gain_curve.pdf`
- `plots/without_variance/action_replace_return_curve.png`
- `plots/without_variance/action_replace_return_curve.pdf`
- `plots/without_variance/action_replace_gain_curve.png`
- `plots/without_variance/action_replace_gain_curve.pdf`

## Output CSV files

- `outputs/eval_metrics_final.csv`: latest merged per-seed eval rows.
- `outputs/summary_by_scenario.csv`: per-model, per-axis, per-factor aggregate return table.
- `outputs/drop_summary.csv`: nominal-minus-perturbed drop table.
- `outputs/gain_summary.csv`: paired robust-gain table against vanilla.
- `outputs/axis_overview.csv`: compact axis-level overview.
- `outputs/curve_points.csv`: same data used for return plots.
- `outputs/gain_curve_points.csv`: same data used for gain plots.
