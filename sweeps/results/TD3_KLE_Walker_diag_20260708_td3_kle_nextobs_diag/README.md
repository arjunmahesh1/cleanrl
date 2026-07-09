# Walker2d TD3 KLE Next-State Ensemble Diagnostic

Date: 2026-07-09

This folder packages the final pinned robustness evaluation for normalized PPO.

## Contents

- Raw metrics: `/home/users/am1015/cleanrl/sweeps/results/TD3_KLE_Walker_diag_20260708_td3_kle_nextobs_diag/raw_metrics`
- Aggregated outputs: `/home/users/am1015/cleanrl/sweeps/results/TD3_KLE_Walker_diag_20260708_td3_kle_nextobs_diag/outputs`
- Plots: `/home/users/am1015/cleanrl/sweeps/results/TD3_KLE_Walker_diag_20260708_td3_kle_nextobs_diag/plots`

## Evaluation protocol

- Models are compared on the same perturbation grid for the configured axes.
- Nominal reference within each axis is the `factor=1.0` point.
- Curves show mean return across seeds with `95% CI` shading.
- Robust gain is defined as `vanilla_drop - model_drop`; positive is better.
- Plot files are exported in both `PNG` and vector `PDF` format.

## Model labels

- `vanilla` -> Vanilla
- `kle2` -> kle2
- `kle20` -> kle20
- `kle100` -> kle100

## Nominal returns by axis

| Axis | Vanilla | kle2 | kle20 | kle100 |
| --- | --- | --- | --- | --- |
| action_replace | n/a | n/a | n/a | n/a |
| actuator_gain | 2200.47 +/- 1154.86 | 482.79 +/- 101.71 | 635.51 +/- 838.97 | 785.06 +/- 237.81 |
| mass | 2188.07 +/- 1151.58 | 467.69 +/- 91.37 | 635.51 +/- 838.97 | 796.95 +/- 258.93 |

## Axis overview

| Axis | Model | Nominal return | Mean gain over perturbed scenarios | Positive gain scenarios |
| --- | --- | --- | --- | --- |
| actuator_gain | kle2 | 482.79 +/- 101.71 | +1192.79 | 15/15 |
| actuator_gain | kle20 | 635.51 +/- 838.97 | +1059.71 | 14/15 |
| actuator_gain | kle100 | 785.06 +/- 237.81 | +1090.35 | 15/15 |
| mass | kle2 | 467.69 +/- 91.37 | +1152.59 | 14/14 |
| mass | kle20 | 635.51 +/- 838.97 | +1013.95 | 14/14 |
| mass | kle100 | 796.95 +/- 258.93 | +1064.28 | 14/14 |

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
- `plots/with_variance/actuator_gain_return_curve.png`
- `plots/with_variance/actuator_gain_return_curve.pdf`
- `plots/with_variance/actuator_gain_gain_curve.png`
- `plots/with_variance/actuator_gain_gain_curve.pdf`
- `plots/without_variance/actuator_gain_return_curve.png`
- `plots/without_variance/actuator_gain_return_curve.pdf`
- `plots/without_variance/actuator_gain_gain_curve.png`
- `plots/without_variance/actuator_gain_gain_curve.pdf`
- `plots/with_variance/mass_return_curve.png`
- `plots/with_variance/mass_return_curve.pdf`
- `plots/with_variance/mass_gain_curve.png`
- `plots/with_variance/mass_gain_curve.pdf`
- `plots/without_variance/mass_return_curve.png`
- `plots/without_variance/mass_return_curve.pdf`
- `plots/without_variance/mass_gain_curve.png`
- `plots/without_variance/mass_gain_curve.pdf`

## Output CSV files

- `outputs/eval_metrics_final.csv`: latest merged per-seed eval rows.
- `outputs/summary_by_scenario.csv`: per-model, per-axis, per-factor aggregate return table.
- `outputs/drop_summary.csv`: nominal-minus-perturbed drop table.
- `outputs/gain_summary.csv`: paired robust-gain table against vanilla.
- `outputs/axis_overview.csv`: compact axis-level overview.
- `outputs/curve_points.csv`: same data used for return plots.
- `outputs/gain_curve_points.csv`: same data used for gain plots.
