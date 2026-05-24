# Walker2d PPO full-cap repack fixed: targeted_nonmass

Date: 2026-05-20

This folder packages the final pinned robustness evaluation for normalized PPO.

## Contents

- Raw metrics: `/Users/arjunmahesh/Library/CloudStorage/OneDrive-Personal/Duke/Research/RL/cleanrl/sweeps/results/PPO_Walker_final_presentable_evalfix_20260511/regeneration_sources/targeted_nonmass/raw_metrics`
- Aggregated outputs: `/Users/arjunmahesh/Library/CloudStorage/OneDrive-Personal/Duke/Research/RL/cleanrl/sweeps/results/PPO_Walker_fullcap_repack_20260520_fixed/targeted_nonmass/outputs`
- Plots: `/Users/arjunmahesh/Library/CloudStorage/OneDrive-Personal/Duke/Research/RL/cleanrl/sweeps/results/PPO_Walker_fullcap_repack_20260520_fixed/targeted_nonmass/plots`

## Evaluation protocol

- Models are compared on the same perturbation grid for the configured axes.
- Nominal reference within each axis is the `factor=1.0` point.
- Curves show mean return across seeds with `95% CI` shading.
- Robust gain is defined as `vanilla_drop - model_drop`; positive is better.
- Plot files are exported in both `PNG` and vector `PDF` format.

## Model labels

- `vanilla` -> Vanilla
- `a2p85` -> TV cap=2.85
- `a2p95` -> TV cap=2.95
- `a3p00` -> TV cap=3.00
- `a3p05` -> TV cap=3.05
- `a3p10` -> TV cap=3.10
- `a3p20` -> TV cap=3.20
- `a3p50` -> TV cap=3.50
- `a3p70` -> TV cap=3.70
- `a4p00` -> TV cap=4.00

## Nominal returns by axis

| Axis | Vanilla | TV cap=2.85 | TV cap=2.95 | TV cap=3.00 | TV cap=3.05 | TV cap=3.10 | TV cap=3.20 | TV cap=3.50 | TV cap=3.70 | TV cap=4.00 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| foot_left_actuator_gain | 3030.54 +/- 790.55 | 2728.36 +/- 1101.71 | 3522.26 +/- 977.51 | 3108.41 +/- 876.52 | 3675.46 +/- 565.48 | 3396.70 +/- 302.75 | 3095.00 +/- 977.59 | 2853.47 +/- 1024.29 | 2723.84 +/- 823.11 | 3671.75 +/- 324.52 |
| foot_left_damping | 3030.54 +/- 790.55 | 2728.36 +/- 1101.71 | 3522.26 +/- 977.51 | 3108.41 +/- 876.52 | 3675.46 +/- 565.48 | 3396.70 +/- 302.75 | 3095.00 +/- 977.59 | 2853.47 +/- 1024.29 | 2723.84 +/- 823.11 | 3671.75 +/- 324.52 |
| foot_left_friction | 3030.54 +/- 790.55 | 2728.36 +/- 1101.71 | 3522.26 +/- 977.51 | 3108.41 +/- 876.52 | 3675.46 +/- 565.48 | 3396.70 +/- 302.75 | 3095.00 +/- 977.59 | 2853.47 +/- 1024.29 | 2723.84 +/- 823.11 | 3671.75 +/- 324.52 |
| leg_left_actuator_gain | 3030.54 +/- 790.55 | 2728.36 +/- 1101.71 | 3522.26 +/- 977.51 | 3108.41 +/- 876.52 | 3675.46 +/- 565.48 | 3396.70 +/- 302.75 | 3095.00 +/- 977.59 | 2853.47 +/- 1024.29 | 2723.84 +/- 823.11 | 3671.75 +/- 324.52 |
| leg_left_damping | 3030.54 +/- 790.55 | 2728.36 +/- 1101.71 | 3522.26 +/- 977.51 | 3108.41 +/- 876.52 | 3675.46 +/- 565.48 | 3396.70 +/- 302.75 | 3095.00 +/- 977.59 | 2853.47 +/- 1024.29 | 2723.84 +/- 823.11 | 3671.75 +/- 324.52 |
| thigh_left_actuator_gain | 3030.54 +/- 790.55 | 2728.36 +/- 1101.71 | 3522.26 +/- 977.51 | 3108.41 +/- 876.52 | 3675.46 +/- 565.48 | 3396.70 +/- 302.75 | 3095.00 +/- 977.59 | 2853.47 +/- 1024.29 | 2723.84 +/- 823.11 | 3671.75 +/- 324.52 |
| thigh_left_damping | 3030.54 +/- 790.55 | 2728.36 +/- 1101.71 | 3522.26 +/- 977.51 | 3108.41 +/- 876.52 | 3675.46 +/- 565.48 | 3396.70 +/- 302.75 | 3095.00 +/- 977.59 | 2853.47 +/- 1024.29 | 2723.84 +/- 823.11 | 3671.75 +/- 324.52 |

## Axis overview

| Axis | Model | Nominal return | Mean gain over perturbed scenarios | Positive gain scenarios |
| --- | --- | --- | --- | --- |
| foot_left_actuator_gain | TV cap=2.85 | 2728.36 +/- 1101.71 | +291.77 | 11/15 |
| foot_left_actuator_gain | TV cap=2.95 | 3522.26 +/- 977.51 | -405.49 | 3/15 |
| foot_left_actuator_gain | TV cap=3.00 | 3108.41 +/- 876.52 | +380.99 | 9/15 |
| foot_left_actuator_gain | TV cap=3.05 | 3675.46 +/- 565.48 | -540.37 | 0/15 |
| foot_left_actuator_gain | TV cap=3.10 | 3396.70 +/- 302.75 | -47.58 | 6/15 |
| foot_left_actuator_gain | TV cap=3.20 | 3095.00 +/- 977.59 | +266.18 | 11/15 |
| foot_left_actuator_gain | TV cap=3.50 | 2853.47 +/- 1024.29 | +354.35 | 10/15 |
| foot_left_actuator_gain | TV cap=3.70 | 2723.84 +/- 823.11 | +291.93 | 13/15 |
| foot_left_actuator_gain | TV cap=4.00 | 3671.75 +/- 324.52 | -399.01 | 2/15 |
| foot_left_damping | TV cap=2.85 | 2728.36 +/- 1101.71 | +24.49 | 10/15 |
| foot_left_damping | TV cap=2.95 | 3522.26 +/- 977.51 | -16.85 | 6/15 |
| foot_left_damping | TV cap=3.00 | 3108.41 +/- 876.52 | +13.67 | 8/15 |
| foot_left_damping | TV cap=3.05 | 3675.46 +/- 565.48 | -52.25 | 6/15 |
| foot_left_damping | TV cap=3.10 | 3396.70 +/- 302.75 | +38.02 | 9/15 |
| foot_left_damping | TV cap=3.20 | 3095.00 +/- 977.59 | -55.72 | 6/15 |
| foot_left_damping | TV cap=3.50 | 2853.47 +/- 1024.29 | +44.79 | 8/15 |
| foot_left_damping | TV cap=3.70 | 2723.84 +/- 823.11 | -84.14 | 4/15 |
| foot_left_damping | TV cap=4.00 | 3671.75 +/- 324.52 | +30.21 | 10/15 |
| foot_left_friction | TV cap=2.85 | 2728.36 +/- 1101.71 | -228.00 | 6/15 |
| foot_left_friction | TV cap=2.95 | 3522.26 +/- 977.51 | -167.98 | 7/15 |
| foot_left_friction | TV cap=3.00 | 3108.41 +/- 876.52 | -92.81 | 6/15 |
| foot_left_friction | TV cap=3.05 | 3675.46 +/- 565.48 | -531.19 | 0/15 |
| foot_left_friction | TV cap=3.10 | 3396.70 +/- 302.75 | +171.55 | 11/15 |
| foot_left_friction | TV cap=3.20 | 3095.00 +/- 977.59 | -301.37 | 2/15 |
| foot_left_friction | TV cap=3.50 | 2853.47 +/- 1024.29 | -335.07 | 2/15 |
| foot_left_friction | TV cap=3.70 | 2723.84 +/- 823.11 | -177.05 | 4/15 |
| foot_left_friction | TV cap=4.00 | 3671.75 +/- 324.52 | -425.77 | 0/15 |
| leg_left_actuator_gain | TV cap=2.85 | 2728.36 +/- 1101.71 | +70.89 | 7/15 |
| leg_left_actuator_gain | TV cap=2.95 | 3522.26 +/- 977.51 | -280.72 | 1/15 |
| leg_left_actuator_gain | TV cap=3.00 | 3108.41 +/- 876.52 | +70.82 | 10/15 |
| leg_left_actuator_gain | TV cap=3.05 | 3675.46 +/- 565.48 | -373.92 | 0/15 |
| leg_left_actuator_gain | TV cap=3.10 | 3396.70 +/- 302.75 | -64.32 | 8/15 |
| leg_left_actuator_gain | TV cap=3.20 | 3095.00 +/- 977.59 | -149.27 | 3/15 |
| leg_left_actuator_gain | TV cap=3.50 | 2853.47 +/- 1024.29 | +211.45 | 13/15 |
| leg_left_actuator_gain | TV cap=3.70 | 2723.84 +/- 823.11 | +61.82 | 7/15 |
| leg_left_actuator_gain | TV cap=4.00 | 3671.75 +/- 324.52 | -162.91 | 7/15 |
| leg_left_damping | TV cap=2.85 | 2728.36 +/- 1101.71 | +10.74 | 7/15 |
| leg_left_damping | TV cap=2.95 | 3522.26 +/- 977.51 | +31.68 | 10/15 |
| leg_left_damping | TV cap=3.00 | 3108.41 +/- 876.52 | +28.97 | 9/15 |
| leg_left_damping | TV cap=3.05 | 3675.46 +/- 565.48 | +25.95 | 9/15 |
| leg_left_damping | TV cap=3.10 | 3396.70 +/- 302.75 | +84.01 | 11/15 |
| leg_left_damping | TV cap=3.20 | 3095.00 +/- 977.59 | -2.12 | 6/15 |
| leg_left_damping | TV cap=3.50 | 2853.47 +/- 1024.29 | +37.06 | 10/15 |
| leg_left_damping | TV cap=3.70 | 2723.84 +/- 823.11 | -4.33 | 7/15 |
| leg_left_damping | TV cap=4.00 | 3671.75 +/- 324.52 | +44.88 | 9/15 |
| thigh_left_actuator_gain | TV cap=2.85 | 2728.36 +/- 1101.71 | -189.25 | 6/15 |
| thigh_left_actuator_gain | TV cap=2.95 | 3522.26 +/- 977.51 | -866.31 | 0/15 |
| thigh_left_actuator_gain | TV cap=3.00 | 3108.41 +/- 876.52 | -140.48 | 6/15 |
| thigh_left_actuator_gain | TV cap=3.05 | 3675.46 +/- 565.48 | -488.99 | 2/15 |
| thigh_left_actuator_gain | TV cap=3.10 | 3396.70 +/- 302.75 | -255.78 | 3/15 |
| thigh_left_actuator_gain | TV cap=3.20 | 3095.00 +/- 977.59 | -279.93 | 4/15 |
| thigh_left_actuator_gain | TV cap=3.50 | 2853.47 +/- 1024.29 | -42.04 | 6/15 |
| thigh_left_actuator_gain | TV cap=3.70 | 2723.84 +/- 823.11 | -111.89 | 5/15 |
| thigh_left_actuator_gain | TV cap=4.00 | 3671.75 +/- 324.52 | -422.11 | 1/15 |
| thigh_left_damping | TV cap=2.85 | 2728.36 +/- 1101.71 | +0.81 | 9/15 |
| thigh_left_damping | TV cap=2.95 | 3522.26 +/- 977.51 | +14.96 | 10/15 |
| thigh_left_damping | TV cap=3.00 | 3108.41 +/- 876.52 | +46.56 | 11/15 |
| thigh_left_damping | TV cap=3.05 | 3675.46 +/- 565.48 | -10.47 | 7/15 |
| thigh_left_damping | TV cap=3.10 | 3396.70 +/- 302.75 | +76.21 | 11/15 |
| thigh_left_damping | TV cap=3.20 | 3095.00 +/- 977.59 | -69.84 | 6/15 |
| thigh_left_damping | TV cap=3.50 | 2853.47 +/- 1024.29 | +43.46 | 9/15 |
| thigh_left_damping | TV cap=3.70 | 2723.84 +/- 823.11 | +5.06 | 9/15 |
| thigh_left_damping | TV cap=4.00 | 3671.75 +/- 324.52 | +88.89 | 13/15 |

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
