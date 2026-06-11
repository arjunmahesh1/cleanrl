# Walker2d PPO Full 30-Seed No-Noop: single_axis_perturbations

Date: 2026-06-10

This folder packages the final pinned robustness evaluation for normalized PPO.

## Contents

- Raw metrics: `/home/users/am1015/cleanrl/sweeps/results/PPO_full30_allcaps_no_noop_20260524/Walker2d/single_axis_perturbations/raw_metrics`
- Aggregated outputs: `/home/users/am1015/cleanrl/sweeps/results/PPO_full30_allcaps_no_noop_20260524/Walker2d/single_axis_perturbations/outputs`
- Plots: `/home/users/am1015/cleanrl/sweeps/results/PPO_full30_allcaps_no_noop_20260524/Walker2d/single_axis_perturbations/plots`

## Evaluation protocol

- Models are compared on the same perturbation grid for the configured axes.
- Nominal reference within each axis is the `factor=1.0` point.
- Curves show mean return across seeds with `95% CI` shading.
- Robust gain is defined as `vanilla_drop - model_drop`; positive is better.
- Plot files are exported in both `PNG` and vector `PDF` format.

## Model labels

- `vanilla` -> Vanilla
- `a2p85` -> a2p85
- `a2p95` -> a2p95
- `a3p00` -> a3p00
- `a3p05` -> TV cap=3.05
- `a3p10` -> a3p10
- `a3p20` -> a3p20
- `a3p50` -> a3p50
- `a3p70` -> a3p70
- `a4p00` -> a4p00

## Nominal returns by axis

| Axis | Vanilla | a2p85 | a2p95 | a3p00 | TV cap=3.05 | a3p10 | a3p20 | a3p50 | a3p70 | a4p00 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| actuator_gain | 3513.60 +/- 343.01 | 3258.71 +/- 378.06 | 3208.90 +/- 404.12 | 3508.89 +/- 293.95 | 3492.91 +/- 245.96 | 3753.62 +/- 408.31 | 3466.75 +/- 319.14 | 3780.53 +/- 343.67 | 3557.00 +/- 332.48 | 3850.41 +/- 330.96 |
| damping | 3513.60 +/- 343.01 | 3155.56 +/- 368.31 | 3208.90 +/- 404.12 | 3508.89 +/- 293.95 | 3523.74 +/- 243.10 | 3753.62 +/- 408.31 | 3466.75 +/- 319.14 | 3780.53 +/- 343.67 | 3557.00 +/- 332.48 | 3850.41 +/- 330.96 |
| friction | 3513.60 +/- 343.01 | 3258.71 +/- 378.06 | 3208.90 +/- 404.12 | 3508.89 +/- 293.95 | 3492.91 +/- 245.96 | 3753.62 +/- 408.31 | 3466.75 +/- 319.14 | 3780.53 +/- 343.67 | 3557.00 +/- 332.48 | 3850.41 +/- 330.96 |
| mass | 3513.60 +/- 343.01 | 3258.71 +/- 378.06 | 3208.90 +/- 404.12 | 3500.66 +/- 288.98 | 3492.91 +/- 245.96 | 3742.31 +/- 412.06 | 3466.75 +/- 319.14 | 3780.53 +/- 343.67 | 3557.00 +/- 332.48 | 3850.41 +/- 330.96 |

## Axis overview

| Axis | Model | Nominal return | Mean gain over perturbed scenarios | Positive gain scenarios |
| --- | --- | --- | --- | --- |
| actuator_gain | Vanilla | 3513.60 +/- 343.01 | +0.00 | 0/0 |
| actuator_gain | Vanilla | 3513.60 +/- 343.01 | +0.00 | 0/15 |
| actuator_gain | a2p85 | 3258.71 +/- 378.06 | +166.28 | 13/15 |
| actuator_gain | a2p95 | 3208.90 +/- 404.12 | +321.74 | 14/15 |
| actuator_gain | a3p00 | 3508.89 +/- 293.95 | -29.25 | 6/15 |
| actuator_gain | TV cap=3.05 | 3492.91 +/- 245.96 | +29.46 | 12/15 |
| actuator_gain | a3p10 | 3753.62 +/- 408.31 | -215.08 | 1/15 |
| actuator_gain | a3p20 | 3466.75 +/- 319.14 | +96.34 | 14/15 |
| actuator_gain | a3p50 | 3780.53 +/- 343.67 | -260.40 | 1/15 |
| actuator_gain | a3p70 | 3557.00 +/- 332.48 | +1.53 | 5/15 |
| actuator_gain | a4p00 | 3850.41 +/- 330.96 | -246.85 | 1/15 |
| damping | Vanilla | 3513.60 +/- 343.01 | +0.00 | 0/0 |
| damping | Vanilla | 3513.60 +/- 343.01 | +0.00 | 0/15 |
| damping | a2p85 | 3155.56 +/- 368.31 | +77.09 | 12/15 |
| damping | a2p95 | 3208.90 +/- 404.12 | +69.17 | 15/15 |
| damping | a3p00 | 3508.89 +/- 293.95 | +52.36 | 13/15 |
| damping | TV cap=3.05 | 3523.74 +/- 243.10 | +41.53 | 13/15 |
| damping | a3p10 | 3753.62 +/- 408.31 | -7.90 | 5/15 |
| damping | a3p20 | 3466.75 +/- 319.14 | +27.30 | 11/15 |
| damping | a3p50 | 3780.53 +/- 343.67 | +66.68 | 15/15 |
| damping | a3p70 | 3557.00 +/- 332.48 | +77.02 | 9/15 |
| damping | a4p00 | 3850.41 +/- 330.96 | +30.13 | 9/15 |
| friction | Vanilla | 3513.60 +/- 343.01 | +0.00 | 0/0 |
| friction | Vanilla | 3513.60 +/- 343.01 | +0.00 | 0/15 |
| friction | a2p85 | 3258.71 +/- 378.06 | +189.06 | 11/15 |
| friction | a2p95 | 3208.90 +/- 404.12 | +344.69 | 15/15 |
| friction | a3p00 | 3508.89 +/- 293.95 | +157.64 | 11/15 |
| friction | TV cap=3.05 | 3492.91 +/- 245.96 | -51.26 | 6/15 |
| friction | a3p10 | 3753.62 +/- 408.31 | +33.54 | 7/15 |
| friction | a3p20 | 3466.75 +/- 319.14 | +17.46 | 9/15 |
| friction | a3p50 | 3780.53 +/- 343.67 | -12.81 | 8/15 |
| friction | a3p70 | 3557.00 +/- 332.48 | +8.90 | 6/15 |
| friction | a4p00 | 3850.41 +/- 330.96 | -168.26 | 5/15 |
| mass | Vanilla | 3513.60 +/- 343.01 | +0.00 | 0/0 |
| mass | Vanilla | 3513.60 +/- 343.01 | +0.00 | 0/14 |
| mass | a2p85 | 3258.71 +/- 378.06 | +143.36 | 11/14 |
| mass | a2p95 | 3208.90 +/- 404.12 | +292.69 | 13/14 |
| mass | a3p00 | 3500.66 +/- 288.98 | -52.81 | 5/14 |
| mass | TV cap=3.05 | 3492.91 +/- 245.96 | +32.67 | 11/14 |
| mass | a3p10 | 3742.31 +/- 412.06 | -208.15 | 1/14 |
| mass | a3p20 | 3466.75 +/- 319.14 | +91.58 | 13/14 |
| mass | a3p50 | 3780.53 +/- 343.67 | -273.29 | 1/14 |
| mass | a3p70 | 3557.00 +/- 332.48 | -38.09 | 4/14 |
| mass | a4p00 | 3850.41 +/- 330.96 | -242.29 | 1/14 |

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
