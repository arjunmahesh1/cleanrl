# Walker2d TD3 Physical KL Radius 30-Seed: Targeted Localized Perturbations

Date: 2026-07-25

This folder packages the final pinned robustness evaluation for normalized PPO.

## Contents

- Raw metrics: `/home/users/am1015/cleanrl/sweeps/results/TD3_KLPhysical_Walker_full30_20260724_td3_klrho_phys5_full30_confirm_compat/Walker2d/targeted_localized_perturbations/raw_metrics`
- Aggregated outputs: `/home/users/am1015/cleanrl/sweeps/results/TD3_KLPhysical_Walker_full30_20260724_td3_klrho_phys5_full30_confirm_compat/Walker2d/targeted_localized_perturbations/outputs`
- Plots: `/home/users/am1015/cleanrl/sweeps/results/TD3_KLPhysical_Walker_full30_20260724_td3_klrho_phys5_full30_confirm_compat/Walker2d/targeted_localized_perturbations/plots`

## Evaluation protocol

- Models are compared on the same perturbation grid for the configured axes.
- Nominal reference within each axis is the `factor=1.0` point.
- Curves show mean return across seeds with `95% CI` shading.
- Robust gain is defined as `vanilla_drop - model_drop`; positive is better.
- Plot files are exported in both `PNG` and vector `PDF` format.

## Model labels

- `vanilla` -> Vanilla
- `klprho0` -> KL rho=0
- `klprho0p05` -> KL rho=0.05

## Nominal returns by axis

| Axis | Vanilla | KL rho=0 | KL rho=0.05 |
| --- | --- | --- | --- |
| foot_left_actuator_gain | 3955.51 +/- 151.23 | 4134.92 +/- 221.21 | 3604.27 +/- 381.79 |
| foot_left_damping | 3944.42 +/- 146.21 | 4131.94 +/- 226.62 | 3604.27 +/- 381.79 |
| foot_left_friction | 3944.42 +/- 146.21 | 4131.94 +/- 226.62 | 3604.18 +/- 385.17 |
| foot_left_mass | 3955.51 +/- 151.23 | 4134.92 +/- 221.21 | 3604.18 +/- 385.17 |
| leg_left_actuator_gain | 3955.51 +/- 151.23 | 4134.92 +/- 221.21 | 3604.27 +/- 381.79 |
| leg_left_damping | 3957.08 +/- 150.63 | 4131.94 +/- 226.62 | 3604.18 +/- 385.17 |
| leg_left_mass | 3944.42 +/- 146.21 | 4134.92 +/- 221.21 | 3604.18 +/- 385.17 |
| thigh_left_actuator_gain | 3955.51 +/- 151.23 | 4131.94 +/- 226.62 | 3604.18 +/- 385.17 |
| thigh_left_damping | 3955.51 +/- 151.23 | 4131.94 +/- 226.62 | 3604.18 +/- 385.17 |
| thigh_left_mass | 3955.51 +/- 151.23 | 4134.92 +/- 221.21 | 3604.18 +/- 385.17 |

## Axis overview

| Axis | Model | Nominal return | Mean gain over perturbed scenarios | Positive gain scenarios |
| --- | --- | --- | --- | --- |
| foot_left_actuator_gain | KL rho=0 | 4134.92 +/- 221.21 | +121.64 | 13/15 |
| foot_left_actuator_gain | KL rho=0.05 | 3604.27 +/- 381.79 | +402.02 | 15/15 |
| foot_left_damping | KL rho=0 | 4131.94 +/- 226.62 | -9.42 | 6/15 |
| foot_left_damping | KL rho=0.05 | 3604.27 +/- 381.79 | -19.65 | 2/15 |
| foot_left_friction | KL rho=0 | 4131.94 +/- 226.62 | +57.30 | 9/15 |
| foot_left_friction | KL rho=0.05 | 3604.18 +/- 385.17 | -56.19 | 4/15 |
| foot_left_mass | KL rho=0 | 4134.92 +/- 221.21 | -52.02 | 6/14 |
| foot_left_mass | KL rho=0.05 | 3604.18 +/- 385.17 | -43.57 | 5/14 |
| leg_left_actuator_gain | KL rho=0 | 4134.92 +/- 221.21 | +57.69 | 13/15 |
| leg_left_actuator_gain | KL rho=0.05 | 3604.27 +/- 381.79 | +110.64 | 11/15 |
| leg_left_damping | KL rho=0 | 4131.94 +/- 226.62 | +18.33 | 12/15 |
| leg_left_damping | KL rho=0.05 | 3604.18 +/- 385.17 | +1.21 | 8/15 |
| leg_left_mass | KL rho=0 | 4134.92 +/- 221.21 | -53.41 | 2/14 |
| leg_left_mass | KL rho=0.05 | 3604.18 +/- 385.17 | -56.73 | 4/14 |
| thigh_left_actuator_gain | KL rho=0 | 4131.94 +/- 226.62 | -109.07 | 5/15 |
| thigh_left_actuator_gain | KL rho=0.05 | 3604.18 +/- 385.17 | +177.02 | 9/15 |
| thigh_left_damping | KL rho=0 | 4131.94 +/- 226.62 | +17.13 | 13/15 |
| thigh_left_damping | KL rho=0.05 | 3604.18 +/- 385.17 | -8.54 | 6/15 |
| thigh_left_mass | KL rho=0 | 4134.92 +/- 221.21 | -64.75 | 3/14 |
| thigh_left_mass | KL rho=0.05 | 3604.18 +/- 385.17 | +51.47 | 12/14 |

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
- `plots/with_variance/foot_left_mass_return_curve.png`
- `plots/with_variance/foot_left_mass_return_curve.pdf`
- `plots/with_variance/foot_left_mass_gain_curve.png`
- `plots/with_variance/foot_left_mass_gain_curve.pdf`
- `plots/without_variance/foot_left_mass_return_curve.png`
- `plots/without_variance/foot_left_mass_return_curve.pdf`
- `plots/without_variance/foot_left_mass_gain_curve.png`
- `plots/without_variance/foot_left_mass_gain_curve.pdf`
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
- `plots/with_variance/leg_left_mass_return_curve.png`
- `plots/with_variance/leg_left_mass_return_curve.pdf`
- `plots/with_variance/leg_left_mass_gain_curve.png`
- `plots/with_variance/leg_left_mass_gain_curve.pdf`
- `plots/without_variance/leg_left_mass_return_curve.png`
- `plots/without_variance/leg_left_mass_return_curve.pdf`
- `plots/without_variance/leg_left_mass_gain_curve.png`
- `plots/without_variance/leg_left_mass_gain_curve.pdf`
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
