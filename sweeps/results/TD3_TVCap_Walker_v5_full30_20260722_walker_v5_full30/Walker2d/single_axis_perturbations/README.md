# Walker2d TD3 TV-Cap 30-Seed: Single Axis Perturbations

Date: 2026-07-25

This folder packages the final pinned robustness evaluation for normalized PPO.

## Contents

- Raw metrics: `/home/users/am1015/cleanrl/sweeps/results/TD3_TVCap_Walker_v5_full30_20260722_walker_v5_full30/Walker2d/single_axis_perturbations/raw_metrics`
- Aggregated outputs: `/home/users/am1015/cleanrl/sweeps/results/TD3_TVCap_Walker_v5_full30_20260722_walker_v5_full30/Walker2d/single_axis_perturbations/outputs`
- Plots: `/home/users/am1015/cleanrl/sweeps/results/TD3_TVCap_Walker_v5_full30_20260722_walker_v5_full30/Walker2d/single_axis_perturbations/plots`

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
- `tvc400` -> TV c=400
- `tvc500` -> TV c=500

## Nominal returns by axis

| Axis | Vanilla | TV c=100 | TV c=150 | TV c=200 | TV c=225 | TV c=250 | TV c=275 | TV c=300 | TV c=400 | TV c=500 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| actuator_gain | 3948.17 +/- 249.58 | 553.94 +/- 39.49 | 782.17 +/- 90.12 | 1435.52 +/- 198.96 | 1885.68 +/- 283.08 | 2802.90 +/- 293.63 | 3163.91 +/- 286.07 | 3561.48 +/- 126.66 | 4007.94 +/- 258.84 | 3704.09 +/- 312.19 |
| damping | 3956.08 +/- 251.25 | 553.94 +/- 39.49 | 782.17 +/- 90.12 | 1435.52 +/- 198.96 | 1885.68 +/- 283.08 | 2802.90 +/- 293.63 | 3141.05 +/- 286.24 | 3561.48 +/- 126.66 | 4007.94 +/- 258.84 | 3704.09 +/- 312.19 |
| friction | 3948.17 +/- 249.58 | 559.06 +/- 38.56 | 782.17 +/- 90.12 | 1435.52 +/- 198.96 | 1869.44 +/- 287.64 | 2799.87 +/- 307.09 | 3141.05 +/- 286.24 | 3565.15 +/- 133.60 | 4007.94 +/- 258.84 | 3659.87 +/- 314.15 |
| mass | 3948.17 +/- 249.58 | 553.94 +/- 39.49 | 782.17 +/- 90.12 | 1413.80 +/- 178.24 | 1885.68 +/- 283.08 | 2802.90 +/- 293.63 | 3141.05 +/- 286.24 | 3561.48 +/- 126.66 | 4007.94 +/- 258.84 | 3704.09 +/- 312.19 |

## Axis overview

| Axis | Model | Nominal return | Mean gain over perturbed scenarios | Positive gain scenarios |
| --- | --- | --- | --- | --- |
| actuator_gain | TV c=100 | 553.94 +/- 39.49 | +2134.19 | 14/15 |
| actuator_gain | TV c=150 | 782.17 +/- 90.12 | +1976.01 | 14/15 |
| actuator_gain | TV c=200 | 1435.52 +/- 198.96 | +1592.25 | 14/15 |
| actuator_gain | TV c=225 | 1885.68 +/- 283.08 | +1298.34 | 14/15 |
| actuator_gain | TV c=250 | 2802.90 +/- 293.63 | +692.45 | 13/15 |
| actuator_gain | TV c=275 | 3163.91 +/- 286.07 | +414.33 | 13/15 |
| actuator_gain | TV c=300 | 3561.48 +/- 126.66 | +235.67 | 14/15 |
| actuator_gain | TV c=400 | 4007.94 +/- 258.84 | -12.08 | 6/15 |
| actuator_gain | TV c=500 | 3704.09 +/- 312.19 | +100.91 | 12/15 |
| damping | TV c=100 | 553.94 +/- 39.49 | +1.72 | 7/15 |
| damping | TV c=150 | 782.17 +/- 90.12 | -9.73 | 5/15 |
| damping | TV c=200 | 1435.52 +/- 198.96 | +1.56 | 9/15 |
| damping | TV c=225 | 1885.68 +/- 283.08 | +5.48 | 8/15 |
| damping | TV c=250 | 2802.90 +/- 293.63 | +31.82 | 12/15 |
| damping | TV c=275 | 3141.05 +/- 286.24 | +4.91 | 8/15 |
| damping | TV c=300 | 3561.48 +/- 126.66 | -14.45 | 7/15 |
| damping | TV c=400 | 4007.94 +/- 258.84 | -12.39 | 6/15 |
| damping | TV c=500 | 3704.09 +/- 312.19 | -6.72 | 5/15 |
| friction | TV c=100 | 559.06 +/- 38.56 | +1256.41 | 15/15 |
| friction | TV c=150 | 782.17 +/- 90.12 | +1137.32 | 13/15 |
| friction | TV c=200 | 1435.52 +/- 198.96 | +903.07 | 14/15 |
| friction | TV c=225 | 1869.44 +/- 287.64 | +757.68 | 12/15 |
| friction | TV c=250 | 2799.87 +/- 307.09 | +491.71 | 13/15 |
| friction | TV c=275 | 3141.05 +/- 286.24 | +322.69 | 12/15 |
| friction | TV c=300 | 3565.15 +/- 133.60 | +272.04 | 10/15 |
| friction | TV c=400 | 4007.94 +/- 258.84 | +39.99 | 10/15 |
| friction | TV c=500 | 3659.87 +/- 314.15 | +120.05 | 11/15 |
| mass | TV c=100 | 553.94 +/- 39.49 | +1871.41 | 13/14 |
| mass | TV c=150 | 782.17 +/- 90.12 | +1772.36 | 13/14 |
| mass | TV c=200 | 1413.80 +/- 178.24 | +1526.86 | 13/14 |
| mass | TV c=225 | 1885.68 +/- 283.08 | +1222.03 | 13/14 |
| mass | TV c=250 | 2802.90 +/- 293.63 | +652.66 | 12/14 |
| mass | TV c=275 | 3141.05 +/- 286.24 | +404.13 | 12/14 |
| mass | TV c=300 | 3561.48 +/- 126.66 | +225.55 | 13/14 |
| mass | TV c=400 | 4007.94 +/- 258.84 | -35.95 | 4/14 |
| mass | TV c=500 | 3704.09 +/- 312.19 | +70.63 | 11/14 |

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
