# Walker2d TD3 TV-Cap 30-Seed: Targeted Localized Perturbations

Date: 2026-07-25

This folder packages the final pinned robustness evaluation for normalized PPO.

## Contents

- Raw metrics: `/home/users/am1015/cleanrl/sweeps/results/TD3_TVCap_Walker_v5_full30_20260722_walker_v5_full30/Walker2d/targeted_localized_perturbations/raw_metrics`
- Aggregated outputs: `/home/users/am1015/cleanrl/sweeps/results/TD3_TVCap_Walker_v5_full30_20260722_walker_v5_full30/Walker2d/targeted_localized_perturbations/outputs`
- Plots: `/home/users/am1015/cleanrl/sweeps/results/TD3_TVCap_Walker_v5_full30_20260722_walker_v5_full30/Walker2d/targeted_localized_perturbations/plots`

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
| foot_left_actuator_gain | 3948.17 +/- 249.58 | 553.94 +/- 39.49 | 782.17 +/- 90.12 | 1435.52 +/- 198.96 | 1885.68 +/- 283.08 | 2802.90 +/- 293.63 | 3163.91 +/- 286.07 | 3603.16 +/- 133.92 | 4007.66 +/- 258.21 | 3704.09 +/- 312.19 |
| foot_left_damping | 3962.23 +/- 251.19 | 553.94 +/- 39.49 | 782.17 +/- 90.12 | 1435.52 +/- 198.96 | 1885.68 +/- 283.08 | 2786.60 +/- 297.84 | 3163.91 +/- 286.07 | 3561.48 +/- 126.66 | 4000.08 +/- 258.81 | 3704.09 +/- 312.19 |
| foot_left_friction | 3956.08 +/- 251.25 | 559.06 +/- 38.56 | 782.17 +/- 90.12 | 1435.52 +/- 198.96 | 1869.44 +/- 287.64 | 2802.90 +/- 293.63 | 3141.05 +/- 286.24 | 3603.16 +/- 133.92 | 4000.08 +/- 258.81 | 3704.09 +/- 312.19 |
| foot_left_mass | 3948.17 +/- 249.58 | 553.94 +/- 39.49 | 782.17 +/- 90.12 | 1435.52 +/- 198.96 | 1869.44 +/- 287.64 | 2802.90 +/- 293.63 | 3141.05 +/- 286.24 | 3561.48 +/- 126.66 | 4007.94 +/- 258.84 | 3659.87 +/- 314.15 |
| leg_left_actuator_gain | 3948.17 +/- 249.58 | 559.06 +/- 38.56 | 782.17 +/- 90.12 | 1435.52 +/- 198.96 | 1885.68 +/- 283.08 | 2802.90 +/- 293.63 | 3141.05 +/- 286.24 | 3603.16 +/- 133.92 | 4007.94 +/- 258.84 | 3674.58 +/- 316.69 |
| leg_left_damping | 3956.08 +/- 251.25 | 559.06 +/- 38.56 | 775.35 +/- 86.96 | 1435.52 +/- 198.96 | 1885.68 +/- 283.08 | 2802.90 +/- 293.63 | 3161.18 +/- 290.95 | 3561.48 +/- 126.66 | 4000.08 +/- 258.81 | 3659.87 +/- 314.15 |
| leg_left_mass | 3956.08 +/- 251.25 | 559.06 +/- 38.56 | 782.17 +/- 90.12 | 1413.80 +/- 178.24 | 1885.68 +/- 283.08 | 2799.87 +/- 307.09 | 3141.05 +/- 286.24 | 3561.48 +/- 126.66 | 4007.94 +/- 258.84 | 3659.87 +/- 314.15 |
| thigh_left_actuator_gain | 3948.17 +/- 249.58 | 559.06 +/- 38.56 | 782.17 +/- 90.12 | 1435.52 +/- 198.96 | 1869.44 +/- 287.64 | 2802.90 +/- 293.63 | 3163.91 +/- 286.07 | 3561.48 +/- 126.66 | 4007.94 +/- 258.84 | 3659.87 +/- 314.15 |
| thigh_left_damping | 3948.17 +/- 249.58 | 559.06 +/- 38.56 | 782.17 +/- 90.12 | 1435.52 +/- 198.96 | 1885.68 +/- 283.08 | 2802.90 +/- 293.63 | 3163.91 +/- 286.07 | 3561.48 +/- 126.66 | 4000.08 +/- 258.81 | 3659.87 +/- 314.15 |
| thigh_left_mass | 3948.17 +/- 249.58 | 553.94 +/- 39.49 | 775.35 +/- 86.96 | 1413.80 +/- 178.24 | 1885.68 +/- 283.08 | 2799.87 +/- 307.09 | 3141.05 +/- 286.24 | 3561.48 +/- 126.66 | 4000.08 +/- 258.81 | 3704.09 +/- 312.19 |

## Axis overview

| Axis | Model | Nominal return | Mean gain over perturbed scenarios | Positive gain scenarios |
| --- | --- | --- | --- | --- |
| foot_left_actuator_gain | TV c=100 | 553.94 +/- 39.49 | +1326.37 | 14/15 |
| foot_left_actuator_gain | TV c=150 | 782.17 +/- 90.12 | +1211.22 | 14/15 |
| foot_left_actuator_gain | TV c=200 | 1435.52 +/- 198.96 | +936.03 | 13/15 |
| foot_left_actuator_gain | TV c=225 | 1885.68 +/- 283.08 | +703.74 | 12/15 |
| foot_left_actuator_gain | TV c=250 | 2802.90 +/- 293.63 | +360.79 | 12/15 |
| foot_left_actuator_gain | TV c=275 | 3163.91 +/- 286.07 | +79.78 | 9/15 |
| foot_left_actuator_gain | TV c=300 | 3603.16 +/- 133.92 | -27.64 | 7/15 |
| foot_left_actuator_gain | TV c=400 | 4007.66 +/- 258.21 | -94.38 | 2/15 |
| foot_left_actuator_gain | TV c=500 | 3704.09 +/- 312.19 | +76.60 | 9/15 |
| foot_left_damping | TV c=100 | 553.94 +/- 39.49 | +9.26 | 10/15 |
| foot_left_damping | TV c=150 | 782.17 +/- 90.12 | -3.55 | 5/15 |
| foot_left_damping | TV c=200 | 1435.52 +/- 198.96 | -0.46 | 8/15 |
| foot_left_damping | TV c=225 | 1885.68 +/- 283.08 | +16.72 | 12/15 |
| foot_left_damping | TV c=250 | 2786.60 +/- 297.84 | +38.48 | 14/15 |
| foot_left_damping | TV c=275 | 3163.91 +/- 286.07 | -1.70 | 7/15 |
| foot_left_damping | TV c=300 | 3561.48 +/- 126.66 | +12.27 | 11/15 |
| foot_left_damping | TV c=400 | 4000.08 +/- 258.81 | +8.59 | 9/15 |
| foot_left_damping | TV c=500 | 3704.09 +/- 312.19 | +5.43 | 8/15 |
| foot_left_friction | TV c=100 | 559.06 +/- 38.56 | +344.87 | 13/15 |
| foot_left_friction | TV c=150 | 782.17 +/- 90.12 | +286.51 | 10/15 |
| foot_left_friction | TV c=200 | 1435.52 +/- 198.96 | +181.54 | 6/15 |
| foot_left_friction | TV c=225 | 1869.44 +/- 287.64 | +181.93 | 7/15 |
| foot_left_friction | TV c=250 | 2802.90 +/- 293.63 | +86.84 | 9/15 |
| foot_left_friction | TV c=275 | 3141.05 +/- 286.24 | +6.85 | 5/15 |
| foot_left_friction | TV c=300 | 3603.16 +/- 133.92 | +3.87 | 8/15 |
| foot_left_friction | TV c=400 | 4000.08 +/- 258.81 | +4.47 | 6/15 |
| foot_left_friction | TV c=500 | 3704.09 +/- 312.19 | +53.96 | 12/15 |
| foot_left_mass | TV c=100 | 553.94 +/- 39.49 | +351.05 | 13/14 |
| foot_left_mass | TV c=150 | 782.17 +/- 90.12 | +368.15 | 12/14 |
| foot_left_mass | TV c=200 | 1435.52 +/- 198.96 | +332.52 | 14/14 |
| foot_left_mass | TV c=225 | 1869.44 +/- 287.64 | +258.52 | 11/14 |
| foot_left_mass | TV c=250 | 2802.90 +/- 293.63 | +170.80 | 13/14 |
| foot_left_mass | TV c=275 | 3141.05 +/- 286.24 | +126.71 | 14/14 |
| foot_left_mass | TV c=300 | 3561.48 +/- 126.66 | +89.43 | 8/14 |
| foot_left_mass | TV c=400 | 4007.94 +/- 258.84 | +12.28 | 6/14 |
| foot_left_mass | TV c=500 | 3659.87 +/- 314.15 | +35.72 | 8/14 |
| leg_left_actuator_gain | TV c=100 | 559.06 +/- 38.56 | +562.73 | 12/15 |
| leg_left_actuator_gain | TV c=150 | 782.17 +/- 90.12 | +479.15 | 12/15 |
| leg_left_actuator_gain | TV c=200 | 1435.52 +/- 198.96 | +344.83 | 15/15 |
| leg_left_actuator_gain | TV c=225 | 1885.68 +/- 283.08 | +273.89 | 12/15 |
| leg_left_actuator_gain | TV c=250 | 2802.90 +/- 293.63 | -5.97 | 6/15 |
| leg_left_actuator_gain | TV c=275 | 3141.05 +/- 286.24 | +10.58 | 6/15 |
| leg_left_actuator_gain | TV c=300 | 3603.16 +/- 133.92 | -173.97 | 4/15 |
| leg_left_actuator_gain | TV c=400 | 4007.94 +/- 258.84 | -48.15 | 4/15 |
| leg_left_actuator_gain | TV c=500 | 3674.58 +/- 316.69 | -6.56 | 9/15 |
| leg_left_damping | TV c=100 | 559.06 +/- 38.56 | -3.56 | 5/15 |
| leg_left_damping | TV c=150 | 775.35 +/- 86.96 | -10.26 | 3/15 |
| leg_left_damping | TV c=200 | 1435.52 +/- 198.96 | +0.25 | 9/15 |
| leg_left_damping | TV c=225 | 1885.68 +/- 283.08 | +2.76 | 8/15 |
| leg_left_damping | TV c=250 | 2802.90 +/- 293.63 | +12.95 | 9/15 |
| leg_left_damping | TV c=275 | 3161.18 +/- 290.95 | -17.07 | 5/15 |
| leg_left_damping | TV c=300 | 3561.48 +/- 126.66 | -7.04 | 7/15 |
| leg_left_damping | TV c=400 | 4000.08 +/- 258.81 | +4.77 | 10/15 |
| leg_left_damping | TV c=500 | 3659.87 +/- 314.15 | +46.84 | 15/15 |
| leg_left_mass | TV c=100 | 559.06 +/- 38.56 | +40.48 | 9/14 |
| leg_left_mass | TV c=150 | 782.17 +/- 90.12 | +45.45 | 9/14 |
| leg_left_mass | TV c=200 | 1413.80 +/- 178.24 | +95.39 | 10/14 |
| leg_left_mass | TV c=225 | 1885.68 +/- 283.08 | +51.16 | 10/14 |
| leg_left_mass | TV c=250 | 2799.87 +/- 307.09 | +35.83 | 10/14 |
| leg_left_mass | TV c=275 | 3141.05 +/- 286.24 | -29.11 | 7/14 |
| leg_left_mass | TV c=300 | 3561.48 +/- 126.66 | -22.58 | 6/14 |
| leg_left_mass | TV c=400 | 4007.94 +/- 258.84 | -25.15 | 2/14 |
| leg_left_mass | TV c=500 | 3659.87 +/- 314.15 | +8.86 | 11/14 |
| thigh_left_actuator_gain | TV c=100 | 559.06 +/- 38.56 | +926.94 | 15/15 |
| thigh_left_actuator_gain | TV c=150 | 782.17 +/- 90.12 | +797.42 | 14/15 |
| thigh_left_actuator_gain | TV c=200 | 1435.52 +/- 198.96 | +500.76 | 11/15 |
| thigh_left_actuator_gain | TV c=225 | 1869.44 +/- 287.64 | +382.19 | 11/15 |
| thigh_left_actuator_gain | TV c=250 | 2802.90 +/- 293.63 | +70.31 | 6/15 |
| thigh_left_actuator_gain | TV c=275 | 3163.91 +/- 286.07 | +2.67 | 7/15 |
| thigh_left_actuator_gain | TV c=300 | 3561.48 +/- 126.66 | -58.42 | 5/15 |
| thigh_left_actuator_gain | TV c=400 | 4007.94 +/- 258.84 | +84.38 | 14/15 |
| thigh_left_actuator_gain | TV c=500 | 3659.87 +/- 314.15 | +141.25 | 15/15 |
| thigh_left_damping | TV c=100 | 559.06 +/- 38.56 | +2.46 | 6/15 |
| thigh_left_damping | TV c=150 | 782.17 +/- 90.12 | -11.15 | 4/15 |
| thigh_left_damping | TV c=200 | 1435.52 +/- 198.96 | +7.28 | 9/15 |
| thigh_left_damping | TV c=225 | 1885.68 +/- 283.08 | +24.96 | 11/15 |
| thigh_left_damping | TV c=250 | 2802.90 +/- 293.63 | +25.71 | 9/15 |
| thigh_left_damping | TV c=275 | 3163.91 +/- 286.07 | -10.21 | 5/15 |
| thigh_left_damping | TV c=300 | 3561.48 +/- 126.66 | +7.71 | 11/15 |
| thigh_left_damping | TV c=400 | 4000.08 +/- 258.81 | +4.00 | 9/15 |
| thigh_left_damping | TV c=500 | 3659.87 +/- 314.15 | +41.85 | 15/15 |
| thigh_left_mass | TV c=100 | 553.94 +/- 39.49 | +190.78 | 10/14 |
| thigh_left_mass | TV c=150 | 775.35 +/- 86.96 | +206.77 | 10/14 |
| thigh_left_mass | TV c=200 | 1413.80 +/- 178.24 | +317.82 | 9/14 |
| thigh_left_mass | TV c=225 | 1885.68 +/- 283.08 | +234.03 | 9/14 |
| thigh_left_mass | TV c=250 | 2799.87 +/- 307.09 | +64.38 | 9/14 |
| thigh_left_mass | TV c=275 | 3141.05 +/- 286.24 | +12.02 | 9/14 |
| thigh_left_mass | TV c=300 | 3561.48 +/- 126.66 | +17.33 | 9/14 |
| thigh_left_mass | TV c=400 | 4000.08 +/- 258.81 | -0.84 | 8/14 |
| thigh_left_mass | TV c=500 | 3704.09 +/- 312.19 | -41.66 | 4/14 |

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
