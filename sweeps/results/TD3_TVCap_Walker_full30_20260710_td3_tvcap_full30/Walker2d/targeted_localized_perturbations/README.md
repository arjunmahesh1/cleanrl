# Walker2d TD3 TV-Cap Full 30-Seed: Targeted Localized Perturbations

Date: 2026-07-15

This folder packages the final pinned robustness evaluation for normalized PPO.

## Contents

- Raw metrics: `/home/users/am1015/cleanrl/sweeps/results/TD3_TVCap_Walker_full30_20260710_td3_tvcap_full30/Walker2d/targeted_localized_perturbations/raw_metrics`
- Aggregated outputs: `/home/users/am1015/cleanrl/sweeps/results/TD3_TVCap_Walker_full30_20260710_td3_tvcap_full30/Walker2d/targeted_localized_perturbations/outputs`
- Plots: `/home/users/am1015/cleanrl/sweeps/results/TD3_TVCap_Walker_full30_20260710_td3_tvcap_full30/Walker2d/targeted_localized_perturbations/plots`

## Evaluation protocol

- Models are compared on the same perturbation grid for the configured axes.
- Nominal reference within each axis is the `factor=1.0` point.
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

## Nominal returns by axis

| Axis | Vanilla | TV c=100 | TV c=150 | TV c=200 | TV c=225 | TV c=250 | TV c=275 | TV c=300 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| foot_left_actuator_gain | 1239.19 +/- 324.80 | 438.09 +/- 46.05 | 439.62 +/- 29.08 | 651.94 +/- 91.42 | 928.49 +/- 245.03 | 1119.10 +/- 311.40 | 1024.09 +/- 268.71 | 1184.02 +/- 295.62 |
| foot_left_damping | 1239.19 +/- 324.80 | 437.76 +/- 44.76 | 433.81 +/- 27.19 | 641.92 +/- 87.79 | 905.41 +/- 233.60 | 1117.13 +/- 304.12 | 1024.09 +/- 268.71 | 1184.02 +/- 295.62 |
| foot_left_friction | 1254.46 +/- 338.50 | 441.18 +/- 45.71 | 433.81 +/- 27.19 | 651.94 +/- 91.42 | 928.49 +/- 245.03 | 1119.10 +/- 311.40 | 1024.09 +/- 268.71 | 1207.86 +/- 310.45 |
| foot_left_mass | 1254.46 +/- 338.50 | 437.76 +/- 44.76 | 439.62 +/- 29.08 | 645.72 +/- 88.95 | 905.41 +/- 233.60 | 1119.10 +/- 311.40 | 1047.31 +/- 285.52 | 1184.02 +/- 295.62 |
| leg_left_actuator_gain | 1239.19 +/- 324.80 | 441.18 +/- 45.71 | 433.81 +/- 27.19 | 651.94 +/- 91.42 | 905.41 +/- 233.60 | 1119.10 +/- 311.40 | 1024.09 +/- 268.71 | 1184.02 +/- 295.62 |
| leg_left_damping | 1239.19 +/- 324.80 | 441.18 +/- 45.71 | 439.62 +/- 29.08 | 645.72 +/- 88.95 | 906.82 +/- 233.14 | 1119.10 +/- 311.40 | 1024.09 +/- 268.71 | 1184.02 +/- 295.62 |
| leg_left_mass | 1239.19 +/- 324.80 | 441.18 +/- 45.71 | 439.62 +/- 29.08 | 651.94 +/- 91.42 | 905.41 +/- 233.60 | 1117.13 +/- 304.12 | 1024.09 +/- 268.71 | 1184.02 +/- 295.62 |
| thigh_left_actuator_gain | 1254.46 +/- 338.50 | 441.18 +/- 45.71 | 439.62 +/- 29.08 | 651.94 +/- 91.42 | 905.41 +/- 233.60 | 1119.10 +/- 311.40 | 1024.09 +/- 268.71 | 1184.02 +/- 295.62 |
| thigh_left_damping | 1239.19 +/- 324.80 | 441.18 +/- 45.71 | 433.81 +/- 27.19 | 641.92 +/- 87.79 | 905.41 +/- 233.60 | 1117.13 +/- 304.12 | 1024.09 +/- 268.71 | 1184.02 +/- 295.62 |
| thigh_left_mass | 1254.46 +/- 338.50 | 441.18 +/- 45.71 | 433.81 +/- 27.19 | 645.72 +/- 88.95 | 905.41 +/- 233.60 | 1119.10 +/- 311.40 | 1047.31 +/- 285.52 | 1184.02 +/- 295.62 |

## Axis overview

| Axis | Model | Nominal return | Mean gain over perturbed scenarios | Positive gain scenarios |
| --- | --- | --- | --- | --- |
| foot_left_actuator_gain | TV c=100 | 438.09 +/- 46.05 | +376.32 | 14/15 |
| foot_left_actuator_gain | TV c=150 | 439.62 +/- 29.08 | +348.47 | 14/15 |
| foot_left_actuator_gain | TV c=200 | 651.94 +/- 91.42 | +251.72 | 13/15 |
| foot_left_actuator_gain | TV c=225 | 928.49 +/- 245.03 | +199.20 | 14/15 |
| foot_left_actuator_gain | TV c=250 | 1119.10 +/- 311.40 | +125.29 | 13/15 |
| foot_left_actuator_gain | TV c=275 | 1024.09 +/- 268.71 | +76.42 | 12/15 |
| foot_left_actuator_gain | TV c=300 | 1184.02 +/- 295.62 | +27.06 | 12/15 |
| foot_left_damping | TV c=100 | 437.76 +/- 44.76 | +13.63 | 10/15 |
| foot_left_damping | TV c=150 | 433.81 +/- 27.19 | +20.01 | 13/15 |
| foot_left_damping | TV c=200 | 641.92 +/- 87.79 | +28.92 | 13/15 |
| foot_left_damping | TV c=225 | 905.41 +/- 233.60 | +29.62 | 13/15 |
| foot_left_damping | TV c=250 | 1117.13 +/- 304.12 | +34.65 | 14/15 |
| foot_left_damping | TV c=275 | 1024.09 +/- 268.71 | +52.31 | 14/15 |
| foot_left_damping | TV c=300 | 1184.02 +/- 295.62 | +8.34 | 9/15 |
| foot_left_friction | TV c=100 | 441.18 +/- 45.71 | +140.42 | 13/15 |
| foot_left_friction | TV c=150 | 433.81 +/- 27.19 | +135.30 | 13/15 |
| foot_left_friction | TV c=200 | 651.94 +/- 91.42 | +109.90 | 13/15 |
| foot_left_friction | TV c=225 | 928.49 +/- 245.03 | +51.32 | 12/15 |
| foot_left_friction | TV c=250 | 1119.10 +/- 311.40 | +16.72 | 11/15 |
| foot_left_friction | TV c=275 | 1024.09 +/- 268.71 | +57.22 | 15/15 |
| foot_left_friction | TV c=300 | 1207.86 +/- 310.45 | +0.39 | 6/15 |
| foot_left_mass | TV c=100 | 437.76 +/- 44.76 | +136.69 | 14/14 |
| foot_left_mass | TV c=150 | 439.62 +/- 29.08 | +148.44 | 14/14 |
| foot_left_mass | TV c=200 | 645.72 +/- 88.95 | +123.23 | 14/14 |
| foot_left_mass | TV c=225 | 905.41 +/- 233.60 | +47.08 | 9/14 |
| foot_left_mass | TV c=250 | 1119.10 +/- 311.40 | +99.80 | 11/14 |
| foot_left_mass | TV c=275 | 1047.31 +/- 285.52 | +109.35 | 14/14 |
| foot_left_mass | TV c=300 | 1184.02 +/- 295.62 | +22.80 | 11/14 |
| leg_left_actuator_gain | TV c=100 | 441.18 +/- 45.71 | +102.74 | 11/15 |
| leg_left_actuator_gain | TV c=150 | 433.81 +/- 27.19 | +114.30 | 14/15 |
| leg_left_actuator_gain | TV c=200 | 651.94 +/- 91.42 | +76.98 | 14/15 |
| leg_left_actuator_gain | TV c=225 | 905.41 +/- 233.60 | +89.31 | 12/15 |
| leg_left_actuator_gain | TV c=250 | 1119.10 +/- 311.40 | +28.24 | 11/15 |
| leg_left_actuator_gain | TV c=275 | 1024.09 +/- 268.71 | +43.44 | 11/15 |
| leg_left_actuator_gain | TV c=300 | 1184.02 +/- 295.62 | +22.31 | 8/15 |
| leg_left_damping | TV c=100 | 441.18 +/- 45.71 | -0.14 | 5/15 |
| leg_left_damping | TV c=150 | 439.62 +/- 29.08 | +6.25 | 9/15 |
| leg_left_damping | TV c=200 | 645.72 +/- 88.95 | +17.89 | 12/15 |
| leg_left_damping | TV c=225 | 906.82 +/- 233.14 | +14.21 | 12/15 |
| leg_left_damping | TV c=250 | 1119.10 +/- 311.40 | +4.10 | 9/15 |
| leg_left_damping | TV c=275 | 1024.09 +/- 268.71 | +29.82 | 13/15 |
| leg_left_damping | TV c=300 | 1184.02 +/- 295.62 | -1.99 | 5/15 |
| leg_left_mass | TV c=100 | 441.18 +/- 45.71 | +31.55 | 14/14 |
| leg_left_mass | TV c=150 | 439.62 +/- 29.08 | +41.98 | 13/14 |
| leg_left_mass | TV c=200 | 651.94 +/- 91.42 | +44.65 | 13/14 |
| leg_left_mass | TV c=225 | 905.41 +/- 233.60 | +6.04 | 9/14 |
| leg_left_mass | TV c=250 | 1117.13 +/- 304.12 | +23.66 | 12/14 |
| leg_left_mass | TV c=275 | 1024.09 +/- 268.71 | +64.84 | 13/14 |
| leg_left_mass | TV c=300 | 1184.02 +/- 295.62 | +0.98 | 3/14 |
| thigh_left_actuator_gain | TV c=100 | 441.18 +/- 45.71 | +232.61 | 15/15 |
| thigh_left_actuator_gain | TV c=150 | 439.62 +/- 29.08 | +217.45 | 15/15 |
| thigh_left_actuator_gain | TV c=200 | 651.94 +/- 91.42 | +167.99 | 15/15 |
| thigh_left_actuator_gain | TV c=225 | 905.41 +/- 233.60 | +96.98 | 11/15 |
| thigh_left_actuator_gain | TV c=250 | 1119.10 +/- 311.40 | +70.32 | 13/15 |
| thigh_left_actuator_gain | TV c=275 | 1024.09 +/- 268.71 | +110.31 | 15/15 |
| thigh_left_actuator_gain | TV c=300 | 1184.02 +/- 295.62 | +83.42 | 13/15 |
| thigh_left_damping | TV c=100 | 441.18 +/- 45.71 | +0.22 | 7/15 |
| thigh_left_damping | TV c=150 | 433.81 +/- 27.19 | +9.39 | 10/15 |
| thigh_left_damping | TV c=200 | 641.92 +/- 87.79 | +20.51 | 12/15 |
| thigh_left_damping | TV c=225 | 905.41 +/- 233.60 | +23.54 | 12/15 |
| thigh_left_damping | TV c=250 | 1117.13 +/- 304.12 | +9.49 | 11/15 |
| thigh_left_damping | TV c=275 | 1024.09 +/- 268.71 | +33.88 | 12/15 |
| thigh_left_damping | TV c=300 | 1184.02 +/- 295.62 | -5.42 | 5/15 |
| thigh_left_mass | TV c=100 | 441.18 +/- 45.71 | +27.24 | 6/14 |
| thigh_left_mass | TV c=150 | 433.81 +/- 27.19 | +57.19 | 9/14 |
| thigh_left_mass | TV c=200 | 645.72 +/- 88.95 | +60.40 | 11/14 |
| thigh_left_mass | TV c=225 | 905.41 +/- 233.60 | +64.40 | 13/14 |
| thigh_left_mass | TV c=250 | 1119.10 +/- 311.40 | +17.14 | 6/14 |
| thigh_left_mass | TV c=275 | 1047.31 +/- 285.52 | +53.60 | 10/14 |
| thigh_left_mass | TV c=300 | 1184.02 +/- 295.62 | +12.79 | 8/14 |

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
