# Walker2d TD3 Physical KL Radius 30-Seed: Single Axis Perturbations

Date: 2026-07-25

This folder packages the final pinned robustness evaluation for normalized PPO.

## Contents

- Raw metrics: `/home/users/am1015/cleanrl/sweeps/results/TD3_KLPhysical_Walker_full30_20260724_td3_klrho_phys5_full30_confirm_compat/Walker2d/single_axis_perturbations/raw_metrics`
- Aggregated outputs: `/home/users/am1015/cleanrl/sweeps/results/TD3_KLPhysical_Walker_full30_20260724_td3_klrho_phys5_full30_confirm_compat/Walker2d/single_axis_perturbations/outputs`
- Plots: `/home/users/am1015/cleanrl/sweeps/results/TD3_KLPhysical_Walker_full30_20260724_td3_klrho_phys5_full30_confirm_compat/Walker2d/single_axis_perturbations/plots`

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
| actuator_gain | 3955.51 +/- 151.23 | 4131.94 +/- 226.62 | 3604.18 +/- 385.17 |
| damping | 3955.51 +/- 151.23 | 4134.92 +/- 221.21 | 3604.18 +/- 385.17 |
| friction | 3955.51 +/- 151.23 | 4134.92 +/- 221.21 | 3604.18 +/- 385.17 |
| mass | 3942.85 +/- 146.80 | 4131.94 +/- 226.62 | 3604.18 +/- 385.17 |

## Axis overview

| Axis | Model | Nominal return | Mean gain over perturbed scenarios | Positive gain scenarios |
| --- | --- | --- | --- | --- |
| actuator_gain | KL rho=0 | 4131.94 +/- 226.62 | -46.33 | 4/15 |
| actuator_gain | KL rho=0.05 | 3604.18 +/- 385.17 | +258.73 | 13/15 |
| damping | KL rho=0 | 4134.92 +/- 221.21 | +19.76 | 12/15 |
| damping | KL rho=0.05 | 3604.18 +/- 385.17 | -0.78 | 7/15 |
| friction | KL rho=0 | 4134.92 +/- 221.21 | -178.60 | 5/15 |
| friction | KL rho=0.05 | 3604.18 +/- 385.17 | -30.84 | 9/15 |
| mass | KL rho=0 | 4131.94 +/- 226.62 | -2.17 | 6/14 |
| mass | KL rho=0.05 | 3604.18 +/- 385.17 | +200.31 | 11/14 |

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
