# Walker2d TD3-KL 5-Seed: Single-Axis Nonmass

Date: 2026-07-08

This folder packages the final pinned robustness evaluation for normalized PPO.

## Contents

- Raw metrics: `/home/users/am1015/cleanrl/sweeps/results/TD3_KL_Walker_5seed_no_noop_20260703/Walker2d/single_axis_perturbations/nonmass/raw_metrics`
- Aggregated outputs: `/home/users/am1015/cleanrl/sweeps/results/TD3_KL_Walker_5seed_no_noop_20260703/Walker2d/single_axis_perturbations/nonmass/outputs`
- Plots: `/home/users/am1015/cleanrl/sweeps/results/TD3_KL_Walker_5seed_no_noop_20260703/Walker2d/single_axis_perturbations/nonmass/plots`

## Evaluation protocol

- Models are compared on the same perturbation grid for the configured axes.
- Nominal reference within each axis is the `factor=1.0` point.
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
| actuator_gain | 2099.77 +/- 1067.34 | 368.94 +/- 53.52 | 444.66 +/- 154.30 | 793.37 +/- 158.92 | 392.05 +/- 225.65 | 486.69 +/- 80.90 | 669.28 +/- 286.53 | 584.61 +/- 113.03 | 603.98 +/- 124.09 |
| damping | 2151.99 +/- 1174.63 | 368.94 +/- 53.52 | 444.66 +/- 154.30 | 793.37 +/- 158.92 | 392.05 +/- 225.65 | 474.71 +/- 93.98 | 669.28 +/- 286.53 | 584.61 +/- 113.03 | 603.98 +/- 124.09 |
| friction | 2099.77 +/- 1067.34 | 368.94 +/- 53.52 | 422.16 +/- 139.58 | 793.37 +/- 158.92 | 392.05 +/- 225.65 | 474.71 +/- 93.98 | 669.28 +/- 286.53 | 584.61 +/- 113.03 | 603.98 +/- 124.09 |

## Axis overview

| Axis | Model | Nominal return | Mean gain over perturbed scenarios | Positive gain scenarios |
| --- | --- | --- | --- | --- |
| actuator_gain | klb0p5 | 368.94 +/- 53.52 | +1102.87 | 15/15 |
| actuator_gain | klb1 | 444.66 +/- 154.30 | +1100.55 | 15/15 |
| actuator_gain | klb2 | 793.37 +/- 158.92 | +855.87 | 13/15 |
| actuator_gain | klb5 | 392.05 +/- 225.65 | +1142.02 | 15/15 |
| actuator_gain | klb10 | 486.69 +/- 80.90 | +1063.67 | 15/15 |
| actuator_gain | klb20 | 669.28 +/- 286.53 | +1051.88 | 15/15 |
| actuator_gain | klb50 | 584.61 +/- 113.03 | +1032.90 | 15/15 |
| actuator_gain | klb100 | 603.98 +/- 124.09 | +998.61 | 15/15 |
| damping | klb0p5 | 368.94 +/- 53.52 | -77.58 | 1/15 |
| damping | klb1 | 444.66 +/- 154.30 | -66.16 | 2/15 |
| damping | klb2 | 793.37 +/- 158.92 | -74.52 | 2/15 |
| damping | klb5 | 392.05 +/- 225.65 | -29.10 | 3/15 |
| damping | klb10 | 474.71 +/- 93.98 | -53.64 | 2/15 |
| damping | klb20 | 669.28 +/- 286.53 | -70.97 | 2/15 |
| damping | klb50 | 584.61 +/- 113.03 | -35.51 | 6/15 |
| damping | klb100 | 603.98 +/- 124.09 | -47.19 | 3/15 |
| friction | klb0p5 | 368.94 +/- 53.52 | +523.69 | 9/15 |
| friction | klb1 | 422.16 +/- 139.58 | +560.72 | 10/15 |
| friction | klb2 | 793.37 +/- 158.92 | +532.36 | 10/15 |
| friction | klb5 | 392.05 +/- 225.65 | +523.38 | 10/15 |
| friction | klb10 | 474.71 +/- 93.98 | +484.83 | 9/15 |
| friction | klb20 | 669.28 +/- 286.53 | +455.35 | 8/15 |
| friction | klb50 | 584.61 +/- 113.03 | +475.94 | 9/15 |
| friction | klb100 | 603.98 +/- 124.09 | +471.28 | 9/15 |

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
- `plots/with_variance/actuator_gain_return_curve.png`
- `plots/with_variance/actuator_gain_return_curve.pdf`
- `plots/with_variance/actuator_gain_gain_curve.png`
- `plots/with_variance/actuator_gain_gain_curve.pdf`
- `plots/without_variance/actuator_gain_return_curve.png`
- `plots/without_variance/actuator_gain_return_curve.pdf`
- `plots/without_variance/actuator_gain_gain_curve.png`
- `plots/without_variance/actuator_gain_gain_curve.pdf`
- `plots/with_variance/damping_return_curve.png`
- `plots/with_variance/damping_return_curve.pdf`
- `plots/with_variance/damping_gain_curve.png`
- `plots/with_variance/damping_gain_curve.pdf`
- `plots/without_variance/damping_return_curve.png`
- `plots/without_variance/damping_return_curve.pdf`
- `plots/without_variance/damping_gain_curve.png`
- `plots/without_variance/damping_gain_curve.pdf`
- `plots/with_variance/friction_return_curve.png`
- `plots/with_variance/friction_return_curve.pdf`
- `plots/with_variance/friction_gain_curve.png`
- `plots/with_variance/friction_gain_curve.pdf`
- `plots/without_variance/friction_return_curve.png`
- `plots/without_variance/friction_return_curve.pdf`
- `plots/without_variance/friction_gain_curve.png`
- `plots/without_variance/friction_gain_curve.pdf`

## Output CSV files

- `outputs/eval_metrics_final.csv`: latest merged per-seed eval rows.
- `outputs/summary_by_scenario.csv`: per-model, per-axis, per-factor aggregate return table.
- `outputs/drop_summary.csv`: nominal-minus-perturbed drop table.
- `outputs/gain_summary.csv`: paired robust-gain table against vanilla.
- `outputs/axis_overview.csv`: compact axis-level overview.
- `outputs/curve_points.csv`: same data used for return plots.
- `outputs/gain_curve_points.csv`: same data used for gain plots.
