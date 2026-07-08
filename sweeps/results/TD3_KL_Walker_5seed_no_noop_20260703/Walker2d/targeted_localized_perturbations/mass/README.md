# Walker2d TD3-KL 5-Seed: Targeted Mass

Date: 2026-07-08

This folder packages the final pinned robustness evaluation for normalized PPO.

## Contents

- Raw metrics: `/home/users/am1015/cleanrl/sweeps/results/TD3_KL_Walker_5seed_no_noop_20260703/Walker2d/targeted_localized_perturbations/mass/raw_metrics`
- Aggregated outputs: `/home/users/am1015/cleanrl/sweeps/results/TD3_KL_Walker_5seed_no_noop_20260703/Walker2d/targeted_localized_perturbations/mass/outputs`
- Plots: `/home/users/am1015/cleanrl/sweeps/results/TD3_KL_Walker_5seed_no_noop_20260703/Walker2d/targeted_localized_perturbations/mass/plots`

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
| foot_left_mass | 2099.77 +/- 1067.34 | 368.94 +/- 53.52 | 422.16 +/- 139.58 | 821.98 +/- 156.40 | 392.05 +/- 225.65 | 474.71 +/- 93.98 | 669.28 +/- 286.53 | 584.61 +/- 113.03 | 603.98 +/- 124.09 |
| leg_left_mass | 2099.77 +/- 1067.34 | 368.94 +/- 53.52 | 444.66 +/- 154.30 | 821.98 +/- 156.40 | 392.05 +/- 225.65 | 486.69 +/- 80.90 | 669.28 +/- 286.53 | 584.61 +/- 113.03 | 603.98 +/- 124.09 |
| thigh_left_mass | 2151.99 +/- 1174.63 | 368.94 +/- 53.52 | 422.16 +/- 139.58 | 793.37 +/- 158.92 | 392.05 +/- 225.65 | 486.69 +/- 80.90 | 669.28 +/- 286.53 | 594.19 +/- 135.57 | 603.98 +/- 124.09 |

## Axis overview

| Axis | Model | Nominal return | Mean gain over perturbed scenarios | Positive gain scenarios |
| --- | --- | --- | --- | --- |
| foot_left_mass | klb0p5 | 368.94 +/- 53.52 | +394.48 | 13/14 |
| foot_left_mass | klb1 | 422.16 +/- 139.58 | +419.78 | 13/14 |
| foot_left_mass | klb2 | 821.98 +/- 156.40 | +341.39 | 10/14 |
| foot_left_mass | klb5 | 392.05 +/- 225.65 | +480.88 | 13/14 |
| foot_left_mass | klb10 | 474.71 +/- 93.98 | +370.31 | 13/14 |
| foot_left_mass | klb20 | 669.28 +/- 286.53 | +422.49 | 13/14 |
| foot_left_mass | klb50 | 584.61 +/- 113.03 | +342.60 | 12/14 |
| foot_left_mass | klb100 | 603.98 +/- 124.09 | +405.70 | 13/14 |
| leg_left_mass | klb0p5 | 368.94 +/- 53.52 | +51.03 | 10/14 |
| leg_left_mass | klb1 | 444.66 +/- 154.30 | +50.93 | 10/14 |
| leg_left_mass | klb2 | 821.98 +/- 156.40 | +15.62 | 7/14 |
| leg_left_mass | klb5 | 392.05 +/- 225.65 | +102.99 | 12/14 |
| leg_left_mass | klb10 | 486.69 +/- 80.90 | +50.21 | 10/14 |
| leg_left_mass | klb20 | 669.28 +/- 286.53 | +84.80 | 11/14 |
| leg_left_mass | klb50 | 584.61 +/- 113.03 | +81.29 | 11/14 |
| leg_left_mass | klb100 | 603.98 +/- 124.09 | +49.73 | 10/14 |
| thigh_left_mass | klb0p5 | 368.94 +/- 53.52 | -81.60 | 4/14 |
| thigh_left_mass | klb1 | 422.16 +/- 139.58 | -45.79 | 5/14 |
| thigh_left_mass | klb2 | 793.37 +/- 158.92 | -68.56 | 4/14 |
| thigh_left_mass | klb5 | 392.05 +/- 225.65 | -7.73 | 5/14 |
| thigh_left_mass | klb10 | 486.69 +/- 80.90 | -91.35 | 4/14 |
| thigh_left_mass | klb20 | 669.28 +/- 286.53 | +18.01 | 6/14 |
| thigh_left_mass | klb50 | 594.19 +/- 135.57 | -91.96 | 4/14 |
| thigh_left_mass | klb100 | 603.98 +/- 124.09 | -83.65 | 4/14 |

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
- `plots/with_variance/foot_left_mass_return_curve.png`
- `plots/with_variance/foot_left_mass_return_curve.pdf`
- `plots/with_variance/foot_left_mass_gain_curve.png`
- `plots/with_variance/foot_left_mass_gain_curve.pdf`
- `plots/without_variance/foot_left_mass_return_curve.png`
- `plots/without_variance/foot_left_mass_return_curve.pdf`
- `plots/without_variance/foot_left_mass_gain_curve.png`
- `plots/without_variance/foot_left_mass_gain_curve.pdf`
- `plots/with_variance/leg_left_mass_return_curve.png`
- `plots/with_variance/leg_left_mass_return_curve.pdf`
- `plots/with_variance/leg_left_mass_gain_curve.png`
- `plots/with_variance/leg_left_mass_gain_curve.pdf`
- `plots/without_variance/leg_left_mass_return_curve.png`
- `plots/without_variance/leg_left_mass_return_curve.pdf`
- `plots/without_variance/leg_left_mass_gain_curve.png`
- `plots/without_variance/leg_left_mass_gain_curve.pdf`
- `plots/with_variance/thigh_left_mass_return_curve.png`
- `plots/with_variance/thigh_left_mass_return_curve.pdf`
- `plots/with_variance/thigh_left_mass_gain_curve.png`
- `plots/with_variance/thigh_left_mass_gain_curve.pdf`
- `plots/without_variance/thigh_left_mass_return_curve.png`
- `plots/without_variance/thigh_left_mass_return_curve.pdf`
- `plots/without_variance/thigh_left_mass_gain_curve.png`
- `plots/without_variance/thigh_left_mass_gain_curve.pdf`

## Output CSV files

- `outputs/eval_metrics_final.csv`: latest merged per-seed eval rows.
- `outputs/summary_by_scenario.csv`: per-model, per-axis, per-factor aggregate return table.
- `outputs/drop_summary.csv`: nominal-minus-perturbed drop table.
- `outputs/gain_summary.csv`: paired robust-gain table against vanilla.
- `outputs/axis_overview.csv`: compact axis-level overview.
- `outputs/curve_points.csv`: same data used for return plots.
- `outputs/gain_curve_points.csv`: same data used for gain plots.
