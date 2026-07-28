# Walker2d TD3 TV-Cap Full 30-Seed: Combos

Date: 2026-07-15

This folder packages the final pinned robustness evaluation for normalized PPO.

## Contents

- Raw metrics: `/home/users/am1015/cleanrl/sweeps/results/TD3_TVCap_Walker_full30_20260710_td3_tvcap_full30/Walker2d/combos/raw_metrics`
- Aggregated outputs: `/home/users/am1015/cleanrl/sweeps/results/TD3_TVCap_Walker_full30_20260710_td3_tvcap_full30/Walker2d/combos/outputs`
- Plots: `/home/users/am1015/cleanrl/sweeps/results/TD3_TVCap_Walker_full30_20260710_td3_tvcap_full30/Walker2d/combos/plots`

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
| friction_damping | 1239.19 +/- 324.80 | 441.18 +/- 45.71 | 439.62 +/- 29.08 | 651.94 +/- 91.42 | 905.41 +/- 233.60 | 1117.13 +/- 304.12 | 1030.22 +/- 269.83 | 1184.02 +/- 295.62 |
| friction_mass | 1239.19 +/- 324.80 | 441.18 +/- 45.71 | 433.81 +/- 27.19 | 645.72 +/- 88.95 | 905.41 +/- 233.60 | 1117.13 +/- 304.12 | 1047.31 +/- 285.52 | 1184.02 +/- 295.62 |
| friction_mass_damping | 1239.19 +/- 324.80 | 441.18 +/- 45.71 | 439.62 +/- 29.08 | 651.94 +/- 91.42 | 928.49 +/- 245.03 | 1117.13 +/- 304.12 | 1024.09 +/- 268.71 | 1184.02 +/- 295.62 |
| mass_damping | 1239.19 +/- 324.80 | 441.18 +/- 45.71 | 439.62 +/- 29.08 | 645.72 +/- 88.95 | 905.41 +/- 233.60 | 1119.10 +/- 311.40 | 1047.31 +/- 285.52 | 1184.02 +/- 295.62 |

## Axis overview

| Axis | Model | Nominal return | Mean gain over perturbed scenarios | Positive gain scenarios |
| --- | --- | --- | --- | --- |
| friction_damping | TV c=100 | 441.18 +/- 45.71 | +342.32 | 14/15 |
| friction_damping | TV c=150 | 439.62 +/- 29.08 | +321.29 | 14/15 |
| friction_damping | TV c=200 | 651.94 +/- 91.42 | +242.08 | 14/15 |
| friction_damping | TV c=225 | 905.41 +/- 233.60 | +115.52 | 13/15 |
| friction_damping | TV c=250 | 1117.13 +/- 304.12 | +58.19 | 12/15 |
| friction_damping | TV c=275 | 1030.22 +/- 269.83 | +120.33 | 14/15 |
| friction_damping | TV c=300 | 1184.02 +/- 295.62 | +2.10 | 7/15 |
| friction_mass | TV c=100 | 441.18 +/- 45.71 | +484.02 | 12/14 |
| friction_mass | TV c=150 | 433.81 +/- 27.19 | +495.51 | 13/14 |
| friction_mass | TV c=200 | 645.72 +/- 88.95 | +394.22 | 13/14 |
| friction_mass | TV c=225 | 905.41 +/- 233.60 | +261.62 | 14/14 |
| friction_mass | TV c=250 | 1117.13 +/- 304.12 | +147.62 | 12/14 |
| friction_mass | TV c=275 | 1047.31 +/- 285.52 | +111.13 | 12/14 |
| friction_mass | TV c=300 | 1184.02 +/- 295.62 | +62.11 | 10/14 |
| friction_mass_damping | TV c=100 | 441.18 +/- 45.71 | +488.75 | 11/14 |
| friction_mass_damping | TV c=150 | 439.62 +/- 29.08 | +492.22 | 13/14 |
| friction_mass_damping | TV c=200 | 651.94 +/- 91.42 | +393.23 | 13/14 |
| friction_mass_damping | TV c=225 | 928.49 +/- 245.03 | +240.72 | 13/14 |
| friction_mass_damping | TV c=250 | 1117.13 +/- 304.12 | +147.73 | 12/14 |
| friction_mass_damping | TV c=275 | 1024.09 +/- 268.71 | +133.97 | 11/14 |
| friction_mass_damping | TV c=300 | 1184.02 +/- 295.62 | +64.81 | 11/14 |
| mass_damping | TV c=100 | 441.18 +/- 45.71 | +457.37 | 11/14 |
| mass_damping | TV c=150 | 439.62 +/- 29.08 | +462.87 | 13/14 |
| mass_damping | TV c=200 | 645.72 +/- 88.95 | +359.54 | 11/14 |
| mass_damping | TV c=225 | 905.41 +/- 233.60 | +211.07 | 10/14 |
| mass_damping | TV c=250 | 1119.10 +/- 311.40 | +110.77 | 12/14 |
| mass_damping | TV c=275 | 1047.31 +/- 285.52 | +88.84 | 10/14 |
| mass_damping | TV c=300 | 1184.02 +/- 295.62 | +31.59 | 10/14 |

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
