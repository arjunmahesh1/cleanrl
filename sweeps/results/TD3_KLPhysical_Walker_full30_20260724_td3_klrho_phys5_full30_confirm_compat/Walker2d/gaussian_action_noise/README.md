# Walker2d TD3 Physical KL Radius 30-Seed: Gaussian Action Noise

Date: 2026-07-25

This folder packages the final pinned robustness evaluation for normalized PPO.

## Contents

- Raw metrics: `/home/users/am1015/cleanrl/sweeps/results/TD3_KLPhysical_Walker_full30_20260724_td3_klrho_phys5_full30_confirm_compat/Walker2d/gaussian_action_noise/raw_metrics`
- Aggregated outputs: `/home/users/am1015/cleanrl/sweeps/results/TD3_KLPhysical_Walker_full30_20260724_td3_klrho_phys5_full30_confirm_compat/Walker2d/gaussian_action_noise/outputs`
- Plots: `/home/users/am1015/cleanrl/sweeps/results/TD3_KLPhysical_Walker_full30_20260724_td3_klrho_phys5_full30_confirm_compat/Walker2d/gaussian_action_noise/plots`

## Evaluation protocol

- Models are compared on the same perturbation grid for the configured axes.
- Nominal reference within each axis is the `factor=0.0` point.
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
| action_noise | 3928.91 +/- 137.29 | 4237.03 +/- 197.22 | 3740.81 +/- 381.98 |

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
