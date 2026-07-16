# Walker2d TD3 TV-Cap Full 30-Seed: Single Axis Perturbations

Date: 2026-07-15

This folder packages the final pinned robustness evaluation for normalized PPO.

## Contents

- Raw metrics: `/home/users/am1015/cleanrl/sweeps/results/TD3_TVCap_Walker_full30_20260710_td3_tvcap_full30/Walker2d/single_axis_perturbations/raw_metrics`
- Aggregated outputs: `/home/users/am1015/cleanrl/sweeps/results/TD3_TVCap_Walker_full30_20260710_td3_tvcap_full30/Walker2d/single_axis_perturbations/outputs`
- Plots: `/home/users/am1015/cleanrl/sweeps/results/TD3_TVCap_Walker_full30_20260710_td3_tvcap_full30/Walker2d/single_axis_perturbations/plots`

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
| actuator_gain | 1239.19 +/- 324.80 | 437.76 +/- 44.76 | 439.62 +/- 29.08 | 651.94 +/- 91.42 | 905.41 +/- 233.60 | 1117.13 +/- 304.12 | 1024.09 +/- 268.71 | 1184.02 +/- 295.62 |
| damping | 1254.46 +/- 338.50 | 437.76 +/- 44.76 | 439.62 +/- 29.08 | 651.94 +/- 91.42 | 905.41 +/- 233.60 | 1117.13 +/- 304.12 | 1024.09 +/- 268.71 | 1184.02 +/- 295.62 |
| friction | 1239.19 +/- 324.80 | 441.18 +/- 45.71 | 439.62 +/- 29.08 | 651.94 +/- 91.42 | 905.41 +/- 233.60 | 1117.13 +/- 304.12 | 1047.31 +/- 285.52 | 1184.02 +/- 295.62 |
| mass | 1254.46 +/- 338.50 | 437.76 +/- 44.76 | 439.62 +/- 29.08 | 651.94 +/- 91.42 | 928.49 +/- 245.03 | 1119.10 +/- 311.40 | 1024.09 +/- 268.71 | 1184.02 +/- 295.62 |

## Axis overview

| Axis | Model | Nominal return | Mean gain over perturbed scenarios | Positive gain scenarios |
| --- | --- | --- | --- | --- |
| actuator_gain | TV c=100 | 437.76 +/- 44.76 | +540.21 | 13/15 |
| actuator_gain | TV c=150 | 439.62 +/- 29.08 | +515.25 | 14/15 |
| actuator_gain | TV c=200 | 651.94 +/- 91.42 | +382.67 | 13/15 |
| actuator_gain | TV c=225 | 905.41 +/- 233.60 | +240.64 | 13/15 |
| actuator_gain | TV c=250 | 1117.13 +/- 304.12 | +110.27 | 12/15 |
| actuator_gain | TV c=275 | 1024.09 +/- 268.71 | +147.34 | 14/15 |
| actuator_gain | TV c=300 | 1184.02 +/- 295.62 | +47.40 | 12/15 |
| damping | TV c=100 | 437.76 +/- 44.76 | +31.45 | 13/15 |
| damping | TV c=150 | 439.62 +/- 29.08 | +35.83 | 13/15 |
| damping | TV c=200 | 651.94 +/- 91.42 | +40.59 | 13/15 |
| damping | TV c=225 | 905.41 +/- 233.60 | +58.31 | 15/15 |
| damping | TV c=250 | 1117.13 +/- 304.12 | +36.52 | 12/15 |
| damping | TV c=275 | 1024.09 +/- 268.71 | +68.06 | 15/15 |
| damping | TV c=300 | 1184.02 +/- 295.62 | +28.59 | 12/15 |
| friction | TV c=100 | 441.18 +/- 45.71 | +351.79 | 14/15 |
| friction | TV c=150 | 439.62 +/- 29.08 | +333.35 | 14/15 |
| friction | TV c=200 | 651.94 +/- 91.42 | +253.12 | 14/15 |
| friction | TV c=225 | 905.41 +/- 233.60 | +121.61 | 15/15 |
| friction | TV c=250 | 1117.13 +/- 304.12 | +67.22 | 12/15 |
| friction | TV c=275 | 1047.31 +/- 285.52 | +109.19 | 15/15 |
| friction | TV c=300 | 1184.02 +/- 295.62 | +5.72 | 7/15 |
| mass | TV c=100 | 437.76 +/- 44.76 | +475.37 | 11/14 |
| mass | TV c=150 | 439.62 +/- 29.08 | +479.66 | 14/14 |
| mass | TV c=200 | 651.94 +/- 91.42 | +370.09 | 14/14 |
| mass | TV c=225 | 928.49 +/- 245.03 | +206.58 | 11/14 |
| mass | TV c=250 | 1119.10 +/- 311.40 | +136.83 | 13/14 |
| mass | TV c=275 | 1024.09 +/- 268.71 | +134.23 | 12/14 |
| mass | TV c=300 | 1184.02 +/- 295.62 | +55.15 | 10/14 |

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
