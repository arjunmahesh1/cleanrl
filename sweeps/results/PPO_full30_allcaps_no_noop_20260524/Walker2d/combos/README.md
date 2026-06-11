# Walker2d PPO Full 30-Seed No-Noop: combos

Date: 2026-06-10

This folder packages the final pinned robustness evaluation for normalized PPO.

## Contents

- Raw metrics: `/home/users/am1015/cleanrl/sweeps/results/PPO_full30_allcaps_no_noop_20260524/Walker2d/combos/raw_metrics`
- Aggregated outputs: `/home/users/am1015/cleanrl/sweeps/results/PPO_full30_allcaps_no_noop_20260524/Walker2d/combos/outputs`
- Plots: `/home/users/am1015/cleanrl/sweeps/results/PPO_full30_allcaps_no_noop_20260524/Walker2d/combos/plots`

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
| friction_damping | 3513.60 +/- 343.01 | 3258.71 +/- 378.06 | 3208.90 +/- 404.12 | 3508.89 +/- 293.95 | 3492.91 +/- 245.96 | 3753.62 +/- 408.31 | 3466.75 +/- 319.14 | 3780.53 +/- 343.67 | 3560.97 +/- 337.02 | 3850.41 +/- 330.96 |
| friction_mass | 3513.60 +/- 343.01 | 3258.71 +/- 378.06 | 3208.90 +/- 404.12 | 3508.89 +/- 293.95 | 3492.91 +/- 245.96 | 3753.62 +/- 408.31 | 3373.08 +/- 342.70 | 3780.53 +/- 343.67 | 3557.00 +/- 332.48 | 3850.41 +/- 330.96 |
| friction_mass_damping | 3513.60 +/- 343.01 | 3258.71 +/- 378.06 | 3208.90 +/- 404.12 | 3508.89 +/- 293.95 | 3492.91 +/- 245.96 | 3753.62 +/- 408.31 | 3466.75 +/- 319.14 | 3780.53 +/- 343.67 | 3557.00 +/- 332.48 | 3850.41 +/- 330.96 |
| mass_damping | 3513.60 +/- 343.01 | 3258.71 +/- 378.06 | 3208.90 +/- 404.12 | 3508.89 +/- 293.95 | 3492.91 +/- 245.96 | 3753.62 +/- 408.31 | 3466.75 +/- 319.14 | 3780.53 +/- 343.67 | 3557.00 +/- 332.48 | 3850.41 +/- 330.96 |

## Axis overview

| Axis | Model | Nominal return | Mean gain over perturbed scenarios | Positive gain scenarios |
| --- | --- | --- | --- | --- |
| friction_damping | Vanilla | 3513.60 +/- 343.01 | +0.00 | 0/0 |
| friction_damping | Vanilla | 3513.60 +/- 343.01 | +0.00 | 0/15 |
| friction_damping | a2p85 | 3258.71 +/- 378.06 | +179.21 | 12/15 |
| friction_damping | a2p95 | 3208.90 +/- 404.12 | +339.43 | 15/15 |
| friction_damping | a3p00 | 3508.89 +/- 293.95 | +142.09 | 11/15 |
| friction_damping | TV cap=3.05 | 3492.91 +/- 245.96 | -72.63 | 6/15 |
| friction_damping | a3p10 | 3753.62 +/- 408.31 | +40.57 | 7/15 |
| friction_damping | a3p20 | 3466.75 +/- 319.14 | +19.09 | 10/15 |
| friction_damping | a3p50 | 3780.53 +/- 343.67 | -9.46 | 8/15 |
| friction_damping | a3p70 | 3560.97 +/- 337.02 | +2.95 | 8/15 |
| friction_damping | a4p00 | 3850.41 +/- 330.96 | -176.79 | 5/15 |
| friction_mass | Vanilla | 3513.60 +/- 343.01 | +0.00 | 0/0 |
| friction_mass | Vanilla | 3513.60 +/- 343.01 | +0.00 | 0/14 |
| friction_mass | a2p85 | 3258.71 +/- 378.06 | +126.05 | 11/14 |
| friction_mass | a2p95 | 3208.90 +/- 404.12 | +206.61 | 13/14 |
| friction_mass | a3p00 | 3508.89 +/- 293.95 | -32.19 | 6/14 |
| friction_mass | TV cap=3.05 | 3492.91 +/- 245.96 | -29.67 | 8/14 |
| friction_mass | a3p10 | 3753.62 +/- 408.31 | -208.04 | 0/14 |
| friction_mass | a3p20 | 3373.08 +/- 342.70 | +180.63 | 13/14 |
| friction_mass | a3p50 | 3780.53 +/- 343.67 | -307.06 | 1/14 |
| friction_mass | a3p70 | 3557.00 +/- 332.48 | -58.38 | 3/14 |
| friction_mass | a4p00 | 3850.41 +/- 330.96 | -352.24 | 0/14 |
| friction_mass_damping | Vanilla | 3513.60 +/- 343.01 | +0.00 | 0/0 |
| friction_mass_damping | Vanilla | 3513.60 +/- 343.01 | +0.00 | 0/14 |
| friction_mass_damping | a2p85 | 3258.71 +/- 378.06 | +132.17 | 11/14 |
| friction_mass_damping | a2p95 | 3208.90 +/- 404.12 | +202.14 | 13/14 |
| friction_mass_damping | a3p00 | 3508.89 +/- 293.95 | -36.02 | 6/14 |
| friction_mass_damping | TV cap=3.05 | 3492.91 +/- 245.96 | -39.04 | 9/14 |
| friction_mass_damping | a3p10 | 3753.62 +/- 408.31 | -215.34 | 1/14 |
| friction_mass_damping | a3p20 | 3466.75 +/- 319.14 | +80.71 | 12/14 |
| friction_mass_damping | a3p50 | 3780.53 +/- 343.67 | -310.39 | 0/14 |
| friction_mass_damping | a3p70 | 3557.00 +/- 332.48 | -59.29 | 4/14 |
| friction_mass_damping | a4p00 | 3850.41 +/- 330.96 | -361.30 | 0/14 |
| mass_damping | Vanilla | 3513.60 +/- 343.01 | +0.00 | 0/0 |
| mass_damping | Vanilla | 3513.60 +/- 343.01 | +0.00 | 0/14 |
| mass_damping | a2p85 | 3258.71 +/- 378.06 | +151.68 | 11/14 |
| mass_damping | a2p95 | 3208.90 +/- 404.12 | +301.31 | 13/14 |
| mass_damping | a3p00 | 3508.89 +/- 293.95 | -54.57 | 5/14 |
| mass_damping | TV cap=3.05 | 3492.91 +/- 245.96 | +36.50 | 11/14 |
| mass_damping | a3p10 | 3753.62 +/- 408.31 | -216.26 | 0/14 |
| mass_damping | a3p20 | 3466.75 +/- 319.14 | +104.56 | 13/14 |
| mass_damping | a3p50 | 3780.53 +/- 343.67 | -264.63 | 1/14 |
| mass_damping | a3p70 | 3557.00 +/- 332.48 | -31.85 | 4/14 |
| mass_damping | a4p00 | 3850.41 +/- 330.96 | -223.22 | 2/14 |

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
