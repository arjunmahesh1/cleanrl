# Walker2d TD3 TV-Cap 30-Seed: Combos

Date: 2026-07-25

This folder packages the final pinned robustness evaluation for normalized PPO.

## Contents

- Raw metrics: `/home/users/am1015/cleanrl/sweeps/results/TD3_TVCap_Walker_v5_full30_20260722_walker_v5_full30/Walker2d/combos/raw_metrics`
- Aggregated outputs: `/home/users/am1015/cleanrl/sweeps/results/TD3_TVCap_Walker_v5_full30_20260722_walker_v5_full30/Walker2d/combos/outputs`
- Plots: `/home/users/am1015/cleanrl/sweeps/results/TD3_TVCap_Walker_v5_full30_20260722_walker_v5_full30/Walker2d/combos/plots`

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
| friction_damping | 3956.08 +/- 251.25 | 559.06 +/- 38.56 | 775.35 +/- 86.96 | 1435.52 +/- 198.96 | 1869.44 +/- 287.64 | 2802.90 +/- 293.63 | 3163.91 +/- 286.07 | 3561.48 +/- 126.66 | 4000.08 +/- 258.81 | 3659.87 +/- 314.15 |
| friction_mass | 3956.08 +/- 251.25 | 559.06 +/- 38.56 | 775.35 +/- 86.96 | 1413.80 +/- 178.24 | 1885.68 +/- 283.08 | 2799.87 +/- 307.09 | 3163.91 +/- 286.07 | 3561.48 +/- 126.66 | 4000.08 +/- 258.81 | 3659.87 +/- 314.15 |
| friction_mass_damping | 3948.17 +/- 249.58 | 553.94 +/- 39.49 | 782.17 +/- 90.12 | 1435.52 +/- 198.96 | 1869.44 +/- 287.64 | 2799.87 +/- 307.09 | 3163.91 +/- 286.07 | 3603.16 +/- 133.92 | 4007.94 +/- 258.84 | 3659.87 +/- 314.15 |
| mass_damping | 3948.17 +/- 249.58 | 553.94 +/- 39.49 | 782.17 +/- 90.12 | 1413.80 +/- 178.24 | 1869.44 +/- 287.64 | 2802.90 +/- 293.63 | 3141.05 +/- 286.24 | 3561.48 +/- 126.66 | 4000.08 +/- 258.81 | 3659.87 +/- 314.15 |

## Axis overview

| Axis | Model | Nominal return | Mean gain over perturbed scenarios | Positive gain scenarios |
| --- | --- | --- | --- | --- |
| friction_damping | TV c=100 | 559.06 +/- 38.56 | +1256.67 | 15/15 |
| friction_damping | TV c=150 | 775.35 +/- 86.96 | +1147.93 | 15/15 |
| friction_damping | TV c=200 | 1435.52 +/- 198.96 | +898.07 | 14/15 |
| friction_damping | TV c=225 | 1869.44 +/- 287.64 | +755.96 | 12/15 |
| friction_damping | TV c=250 | 2802.90 +/- 293.63 | +480.33 | 14/15 |
| friction_damping | TV c=275 | 3163.91 +/- 286.07 | +310.95 | 13/15 |
| friction_damping | TV c=300 | 3561.48 +/- 126.66 | +267.63 | 12/15 |
| friction_damping | TV c=400 | 4000.08 +/- 258.81 | +54.06 | 12/15 |
| friction_damping | TV c=500 | 3659.87 +/- 314.15 | +123.58 | 12/15 |
| friction_mass | TV c=100 | 559.06 +/- 38.56 | +1964.37 | 14/14 |
| friction_mass | TV c=150 | 775.35 +/- 86.96 | +1854.59 | 13/14 |
| friction_mass | TV c=200 | 1413.80 +/- 178.24 | +1599.57 | 13/14 |
| friction_mass | TV c=225 | 1885.68 +/- 283.08 | +1294.31 | 13/14 |
| friction_mass | TV c=250 | 2799.87 +/- 307.09 | +709.39 | 12/14 |
| friction_mass | TV c=275 | 3163.91 +/- 286.07 | +455.08 | 12/14 |
| friction_mass | TV c=300 | 3561.48 +/- 126.66 | +282.87 | 12/14 |
| friction_mass | TV c=400 | 4000.08 +/- 258.81 | +3.52 | 4/14 |
| friction_mass | TV c=500 | 3659.87 +/- 314.15 | +144.68 | 12/14 |
| friction_mass_damping | TV c=100 | 553.94 +/- 39.49 | +1971.09 | 14/14 |
| friction_mass_damping | TV c=150 | 782.17 +/- 90.12 | +1843.64 | 13/14 |
| friction_mass_damping | TV c=200 | 1435.52 +/- 198.96 | +1567.59 | 13/14 |
| friction_mass_damping | TV c=225 | 1869.44 +/- 287.64 | +1280.22 | 13/14 |
| friction_mass_damping | TV c=250 | 2799.87 +/- 307.09 | +686.27 | 12/14 |
| friction_mass_damping | TV c=275 | 3163.91 +/- 286.07 | +420.06 | 12/14 |
| friction_mass_damping | TV c=300 | 3603.16 +/- 133.92 | +225.41 | 12/14 |
| friction_mass_damping | TV c=400 | 4007.94 +/- 258.84 | -10.66 | 6/14 |
| friction_mass_damping | TV c=500 | 3659.87 +/- 314.15 | +131.42 | 12/14 |
| mass_damping | TV c=100 | 553.94 +/- 39.49 | +1885.39 | 13/14 |
| mass_damping | TV c=150 | 782.17 +/- 90.12 | +1782.27 | 13/14 |
| mass_damping | TV c=200 | 1413.80 +/- 178.24 | +1538.14 | 13/14 |
| mass_damping | TV c=225 | 1869.44 +/- 287.64 | +1244.70 | 13/14 |
| mass_damping | TV c=250 | 2802.90 +/- 293.63 | +643.46 | 13/14 |
| mass_damping | TV c=275 | 3141.05 +/- 286.24 | +404.39 | 12/14 |
| mass_damping | TV c=300 | 3561.48 +/- 126.66 | +231.11 | 13/14 |
| mass_damping | TV c=400 | 4000.08 +/- 258.81 | -32.22 | 5/14 |
| mass_damping | TV c=500 | 3659.87 +/- 314.15 | +118.55 | 11/14 |

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
- `plots/with_variance/friction_damping_return_curve.png`
- `plots/with_variance/friction_damping_return_curve.pdf`
- `plots/with_variance/friction_damping_gain_curve.png`
- `plots/with_variance/friction_damping_gain_curve.pdf`
- `plots/without_variance/friction_damping_return_curve.png`
- `plots/without_variance/friction_damping_return_curve.pdf`
- `plots/without_variance/friction_damping_gain_curve.png`
- `plots/without_variance/friction_damping_gain_curve.pdf`
- `plots/with_variance/friction_mass_return_curve.png`
- `plots/with_variance/friction_mass_return_curve.pdf`
- `plots/with_variance/friction_mass_gain_curve.png`
- `plots/with_variance/friction_mass_gain_curve.pdf`
- `plots/without_variance/friction_mass_return_curve.png`
- `plots/without_variance/friction_mass_return_curve.pdf`
- `plots/without_variance/friction_mass_gain_curve.png`
- `plots/without_variance/friction_mass_gain_curve.pdf`
- `plots/with_variance/friction_mass_damping_return_curve.png`
- `plots/with_variance/friction_mass_damping_return_curve.pdf`
- `plots/with_variance/friction_mass_damping_gain_curve.png`
- `plots/with_variance/friction_mass_damping_gain_curve.pdf`
- `plots/without_variance/friction_mass_damping_return_curve.png`
- `plots/without_variance/friction_mass_damping_return_curve.pdf`
- `plots/without_variance/friction_mass_damping_gain_curve.png`
- `plots/without_variance/friction_mass_damping_gain_curve.pdf`
- `plots/with_variance/mass_damping_return_curve.png`
- `plots/with_variance/mass_damping_return_curve.pdf`
- `plots/with_variance/mass_damping_gain_curve.png`
- `plots/with_variance/mass_damping_gain_curve.pdf`
- `plots/without_variance/mass_damping_return_curve.png`
- `plots/without_variance/mass_damping_return_curve.pdf`
- `plots/without_variance/mass_damping_gain_curve.png`
- `plots/without_variance/mass_damping_gain_curve.pdf`

## Output CSV files

- `outputs/eval_metrics_final.csv`: latest merged per-seed eval rows.
- `outputs/summary_by_scenario.csv`: per-model, per-axis, per-factor aggregate return table.
- `outputs/drop_summary.csv`: nominal-minus-perturbed drop table.
- `outputs/gain_summary.csv`: paired robust-gain table against vanilla.
- `outputs/axis_overview.csv`: compact axis-level overview.
- `outputs/curve_points.csv`: same data used for return plots.
- `outputs/gain_curve_points.csv`: same data used for gain plots.
