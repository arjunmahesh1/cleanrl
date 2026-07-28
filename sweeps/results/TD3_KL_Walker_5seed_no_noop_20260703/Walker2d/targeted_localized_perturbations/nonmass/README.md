# Walker2d TD3-KL 5-Seed: Targeted Nonmass

Date: 2026-07-08

This folder packages the final pinned robustness evaluation for normalized PPO.

## Contents

- Raw metrics: `/home/users/am1015/cleanrl/sweeps/results/TD3_KL_Walker_5seed_no_noop_20260703/Walker2d/targeted_localized_perturbations/nonmass/raw_metrics`
- Aggregated outputs: `/home/users/am1015/cleanrl/sweeps/results/TD3_KL_Walker_5seed_no_noop_20260703/Walker2d/targeted_localized_perturbations/nonmass/outputs`
- Plots: `/home/users/am1015/cleanrl/sweeps/results/TD3_KL_Walker_5seed_no_noop_20260703/Walker2d/targeted_localized_perturbations/nonmass/plots`

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
| foot_left_actuator_gain | 2099.77 +/- 1067.34 | 370.45 +/- 58.73 | 444.66 +/- 154.30 | 821.98 +/- 156.40 | 392.05 +/- 225.65 | 486.69 +/- 80.90 | 658.89 +/- 265.09 | 584.61 +/- 113.03 | 596.38 +/- 114.66 |
| foot_left_damping | 2099.77 +/- 1067.34 | 368.94 +/- 53.52 | 444.66 +/- 154.30 | 821.98 +/- 156.40 | 386.18 +/- 222.62 | 486.69 +/- 80.90 | 658.89 +/- 265.09 | 584.61 +/- 113.03 | 603.98 +/- 124.09 |
| foot_left_friction | 2151.99 +/- 1174.63 | 368.94 +/- 53.52 | 422.16 +/- 139.58 | 793.37 +/- 158.92 | 386.18 +/- 222.62 | 486.69 +/- 80.90 | 658.89 +/- 265.09 | 584.61 +/- 113.03 | 603.98 +/- 124.09 |
| leg_left_actuator_gain | 2151.99 +/- 1174.63 | 368.94 +/- 53.52 | 422.16 +/- 139.58 | 821.98 +/- 156.40 | 386.18 +/- 222.62 | 474.71 +/- 93.98 | 658.89 +/- 265.09 | 584.61 +/- 113.03 | 603.98 +/- 124.09 |
| leg_left_damping | 2151.99 +/- 1174.63 | 370.45 +/- 58.73 | 422.16 +/- 139.58 | 821.98 +/- 156.40 | 386.18 +/- 222.62 | 474.71 +/- 93.98 | 669.28 +/- 286.53 | 584.61 +/- 113.03 | 603.98 +/- 124.09 |
| thigh_left_actuator_gain | 2151.99 +/- 1174.63 | 368.94 +/- 53.52 | 422.16 +/- 139.58 | 793.37 +/- 158.92 | 386.18 +/- 222.62 | 474.71 +/- 93.98 | 669.28 +/- 286.53 | 584.61 +/- 113.03 | 596.38 +/- 114.66 |
| thigh_left_damping | 2151.99 +/- 1174.63 | 368.94 +/- 53.52 | 422.16 +/- 139.58 | 821.98 +/- 156.40 | 386.18 +/- 222.62 | 474.71 +/- 93.98 | 669.28 +/- 286.53 | 584.61 +/- 113.03 | 596.38 +/- 114.66 |

## Axis overview

| Axis | Model | Nominal return | Mean gain over perturbed scenarios | Positive gain scenarios |
| --- | --- | --- | --- | --- |
| foot_left_actuator_gain | klb0p5 | 370.45 +/- 58.73 | +945.37 | 13/15 |
| foot_left_actuator_gain | klb1 | 444.66 +/- 154.30 | +949.32 | 14/15 |
| foot_left_actuator_gain | klb2 | 821.98 +/- 156.40 | +874.93 | 12/15 |
| foot_left_actuator_gain | klb5 | 392.05 +/- 225.65 | +1001.74 | 15/15 |
| foot_left_actuator_gain | klb10 | 486.69 +/- 80.90 | +934.19 | 15/15 |
| foot_left_actuator_gain | klb20 | 658.89 +/- 265.09 | +1038.51 | 14/15 |
| foot_left_actuator_gain | klb50 | 584.61 +/- 113.03 | +930.39 | 13/15 |
| foot_left_actuator_gain | klb100 | 596.38 +/- 114.66 | +935.18 | 15/15 |
| foot_left_damping | klb0p5 | 368.94 +/- 53.52 | -47.45 | 5/15 |
| foot_left_damping | klb1 | 444.66 +/- 154.30 | -53.85 | 3/15 |
| foot_left_damping | klb2 | 821.98 +/- 156.40 | -86.76 | 0/15 |
| foot_left_damping | klb5 | 386.18 +/- 222.62 | -5.09 | 8/15 |
| foot_left_damping | klb10 | 486.69 +/- 80.90 | -51.81 | 3/15 |
| foot_left_damping | klb20 | 658.89 +/- 265.09 | -30.97 | 5/15 |
| foot_left_damping | klb50 | 584.61 +/- 113.03 | -11.39 | 5/15 |
| foot_left_damping | klb100 | 603.98 +/- 124.09 | -15.17 | 7/15 |
| foot_left_friction | klb0p5 | 368.94 +/- 53.52 | +355.56 | 11/15 |
| foot_left_friction | klb1 | 422.16 +/- 139.58 | +396.30 | 12/15 |
| foot_left_friction | klb2 | 793.37 +/- 158.92 | +352.02 | 10/13 |
| foot_left_friction | klb5 | 386.18 +/- 222.62 | +398.27 | 12/15 |
| foot_left_friction | klb10 | 486.69 +/- 80.90 | +320.40 | 11/15 |
| foot_left_friction | klb20 | 658.89 +/- 265.09 | +418.08 | 12/15 |
| foot_left_friction | klb50 | 584.61 +/- 113.03 | +389.42 | 12/15 |
| foot_left_friction | klb100 | 603.98 +/- 124.09 | +354.53 | 11/15 |
| leg_left_actuator_gain | klb0p5 | 368.94 +/- 53.52 | +401.79 | 10/15 |
| leg_left_actuator_gain | klb1 | 422.16 +/- 139.58 | +447.55 | 10/15 |
| leg_left_actuator_gain | klb2 | 821.98 +/- 156.40 | +261.58 | 9/15 |
| leg_left_actuator_gain | klb5 | 386.18 +/- 222.62 | +487.75 | 10/15 |
| leg_left_actuator_gain | klb10 | 474.71 +/- 93.98 | +360.38 | 9/15 |
| leg_left_actuator_gain | klb20 | 658.89 +/- 265.09 | +380.91 | 9/15 |
| leg_left_actuator_gain | klb50 | 584.61 +/- 113.03 | +424.37 | 9/15 |
| leg_left_actuator_gain | klb100 | 603.98 +/- 124.09 | +380.91 | 9/15 |
| leg_left_damping | klb0p5 | 370.45 +/- 58.73 | -45.66 | 2/15 |
| leg_left_damping | klb1 | 422.16 +/- 139.58 | -8.54 | 4/15 |
| leg_left_damping | klb2 | 821.98 +/- 156.40 | -69.04 | 0/15 |
| leg_left_damping | klb5 | 386.18 +/- 222.62 | -13.23 | 4/15 |
| leg_left_damping | klb10 | 474.71 +/- 93.98 | -28.74 | 1/15 |
| leg_left_damping | klb20 | 669.28 +/- 286.53 | -49.34 | 1/15 |
| leg_left_damping | klb50 | 584.61 +/- 113.03 | -14.37 | 4/15 |
| leg_left_damping | klb100 | 603.98 +/- 124.09 | -24.18 | 4/15 |
| thigh_left_actuator_gain | klb0p5 | 368.94 +/- 53.52 | +460.02 | 11/15 |
| thigh_left_actuator_gain | klb1 | 422.16 +/- 139.58 | +468.49 | 11/15 |
| thigh_left_actuator_gain | klb2 | 793.37 +/- 158.92 | +319.73 | 11/15 |
| thigh_left_actuator_gain | klb5 | 386.18 +/- 222.62 | +538.76 | 12/15 |
| thigh_left_actuator_gain | klb10 | 474.71 +/- 93.98 | +431.69 | 10/15 |
| thigh_left_actuator_gain | klb20 | 669.28 +/- 286.53 | +468.66 | 12/15 |
| thigh_left_actuator_gain | klb50 | 584.61 +/- 113.03 | +523.16 | 11/15 |
| thigh_left_actuator_gain | klb100 | 596.38 +/- 114.66 | +432.38 | 11/15 |
| thigh_left_damping | klb0p5 | 368.94 +/- 53.52 | -41.78 | 3/15 |
| thigh_left_damping | klb1 | 422.16 +/- 139.58 | -8.77 | 7/15 |
| thigh_left_damping | klb2 | 821.98 +/- 156.40 | -71.33 | 1/15 |
| thigh_left_damping | klb5 | 386.18 +/- 222.62 | -0.82 | 7/15 |
| thigh_left_damping | klb10 | 474.71 +/- 93.98 | -30.70 | 3/15 |
| thigh_left_damping | klb20 | 669.28 +/- 286.53 | -33.79 | 5/15 |
| thigh_left_damping | klb50 | 584.61 +/- 113.03 | -6.65 | 8/15 |
| thigh_left_damping | klb100 | 596.38 +/- 114.66 | -25.78 | 7/15 |

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
- `plots/with_variance/foot_left_actuator_gain_return_curve.png`
- `plots/with_variance/foot_left_actuator_gain_return_curve.pdf`
- `plots/with_variance/foot_left_actuator_gain_gain_curve.png`
- `plots/with_variance/foot_left_actuator_gain_gain_curve.pdf`
- `plots/without_variance/foot_left_actuator_gain_return_curve.png`
- `plots/without_variance/foot_left_actuator_gain_return_curve.pdf`
- `plots/without_variance/foot_left_actuator_gain_gain_curve.png`
- `plots/without_variance/foot_left_actuator_gain_gain_curve.pdf`
- `plots/with_variance/foot_left_damping_return_curve.png`
- `plots/with_variance/foot_left_damping_return_curve.pdf`
- `plots/with_variance/foot_left_damping_gain_curve.png`
- `plots/with_variance/foot_left_damping_gain_curve.pdf`
- `plots/without_variance/foot_left_damping_return_curve.png`
- `plots/without_variance/foot_left_damping_return_curve.pdf`
- `plots/without_variance/foot_left_damping_gain_curve.png`
- `plots/without_variance/foot_left_damping_gain_curve.pdf`
- `plots/with_variance/foot_left_friction_return_curve.png`
- `plots/with_variance/foot_left_friction_return_curve.pdf`
- `plots/with_variance/foot_left_friction_gain_curve.png`
- `plots/with_variance/foot_left_friction_gain_curve.pdf`
- `plots/without_variance/foot_left_friction_return_curve.png`
- `plots/without_variance/foot_left_friction_return_curve.pdf`
- `plots/without_variance/foot_left_friction_gain_curve.png`
- `plots/without_variance/foot_left_friction_gain_curve.pdf`
- `plots/with_variance/leg_left_actuator_gain_return_curve.png`
- `plots/with_variance/leg_left_actuator_gain_return_curve.pdf`
- `plots/with_variance/leg_left_actuator_gain_gain_curve.png`
- `plots/with_variance/leg_left_actuator_gain_gain_curve.pdf`
- `plots/without_variance/leg_left_actuator_gain_return_curve.png`
- `plots/without_variance/leg_left_actuator_gain_return_curve.pdf`
- `plots/without_variance/leg_left_actuator_gain_gain_curve.png`
- `plots/without_variance/leg_left_actuator_gain_gain_curve.pdf`
- `plots/with_variance/leg_left_damping_return_curve.png`
- `plots/with_variance/leg_left_damping_return_curve.pdf`
- `plots/with_variance/leg_left_damping_gain_curve.png`
- `plots/with_variance/leg_left_damping_gain_curve.pdf`
- `plots/without_variance/leg_left_damping_return_curve.png`
- `plots/without_variance/leg_left_damping_return_curve.pdf`
- `plots/without_variance/leg_left_damping_gain_curve.png`
- `plots/without_variance/leg_left_damping_gain_curve.pdf`
- `plots/with_variance/thigh_left_actuator_gain_return_curve.png`
- `plots/with_variance/thigh_left_actuator_gain_return_curve.pdf`
- `plots/with_variance/thigh_left_actuator_gain_gain_curve.png`
- `plots/with_variance/thigh_left_actuator_gain_gain_curve.pdf`
- `plots/without_variance/thigh_left_actuator_gain_return_curve.png`
- `plots/without_variance/thigh_left_actuator_gain_return_curve.pdf`
- `plots/without_variance/thigh_left_actuator_gain_gain_curve.png`
- `plots/without_variance/thigh_left_actuator_gain_gain_curve.pdf`
- `plots/with_variance/thigh_left_damping_return_curve.png`
- `plots/with_variance/thigh_left_damping_return_curve.pdf`
- `plots/with_variance/thigh_left_damping_gain_curve.png`
- `plots/with_variance/thigh_left_damping_gain_curve.pdf`
- `plots/without_variance/thigh_left_damping_return_curve.png`
- `plots/without_variance/thigh_left_damping_return_curve.pdf`
- `plots/without_variance/thigh_left_damping_gain_curve.png`
- `plots/without_variance/thigh_left_damping_gain_curve.pdf`

## Output CSV files

- `outputs/eval_metrics_final.csv`: latest merged per-seed eval rows.
- `outputs/summary_by_scenario.csv`: per-model, per-axis, per-factor aggregate return table.
- `outputs/drop_summary.csv`: nominal-minus-perturbed drop table.
- `outputs/gain_summary.csv`: paired robust-gain table against vanilla.
- `outputs/axis_overview.csv`: compact axis-level overview.
- `outputs/curve_points.csv`: same data used for return plots.
- `outputs/gain_curve_points.csv`: same data used for gain plots.
