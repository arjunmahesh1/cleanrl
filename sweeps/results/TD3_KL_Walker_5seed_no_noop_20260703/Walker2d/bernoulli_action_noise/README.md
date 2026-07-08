# Walker2d TD3-KL 5-Seed: Bernoulli Action Replacement

Date: 2026-07-08

This folder packages the final pinned robustness evaluation for normalized PPO.

## Contents

- Raw metrics: `/home/users/am1015/cleanrl/sweeps/results/TD3_KL_Walker_5seed_no_noop_20260703/Walker2d/bernoulli_action_noise/raw_metrics`
- Aggregated outputs: `/home/users/am1015/cleanrl/sweeps/results/TD3_KL_Walker_5seed_no_noop_20260703/Walker2d/bernoulli_action_noise/outputs`
- Plots: `/home/users/am1015/cleanrl/sweeps/results/TD3_KL_Walker_5seed_no_noop_20260703/Walker2d/bernoulli_action_noise/plots`

## Evaluation protocol

- Models are compared on the same perturbation grid for the configured axes.
- Nominal reference within each axis is the `factor=0.0` point.
- Curves show mean return across seeds with `95% CI` shading.
- Robust gain is defined as `vanilla_drop - model_drop`; positive is better.
- Plot files are exported in both `PNG` and vector `PDF` format.

## Model labels

- `vanilla` -> Vanilla
- `klb0p5` -> klb0p5
- `klb1` -> klb1
- `klb2` -> klb2
- `klb5` -> klb5
- `klb10` -> klb10
- `klb20` -> klb20
- `klb50` -> klb50
- `klb100` -> klb100

## Nominal returns by axis

| Axis | Vanilla | klb0p5 | klb1 | klb2 | klb5 | klb10 | klb20 | klb50 | klb100 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| action_replace | 2151.41 +/- 1174.00 | 370.45 +/- 58.73 | 444.65 +/- 154.29 | 821.70 +/- 156.26 | 386.18 +/- 222.62 | 486.68 +/- 80.90 | 658.89 +/- 265.09 | 594.19 +/- 135.57 | 596.38 +/- 114.66 |

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
