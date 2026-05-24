# PPO_Walker_single_axis_mass_0p1_2p0_20260430

Date: 2026-05-15

This folder packages the final pinned robustness evaluation for normalized PPO.

## Contents

- Raw metrics: `/Users/arjunmahesh/Library/CloudStorage/OneDrive-Personal/Duke/Research/RL/cleanrl/sweeps/results/PPO_Walker_final_presentable_20260430/raw_metrics/single_axis_mass`
- Aggregated outputs: `/Users/arjunmahesh/Library/CloudStorage/OneDrive-Personal/Duke/Research/RL/cleanrl/sweeps/results/PPO_Walker_single_axis_mass_0p1_2p0_20260430/outputs`
- Plots: `/Users/arjunmahesh/Library/CloudStorage/OneDrive-Personal/Duke/Research/RL/cleanrl/sweeps/results/PPO_Walker_single_axis_mass_0p1_2p0_20260430/plots`

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
| mass | 2946.33 +/- 835.60 | 2020.63 +/- 891.50 | 3056.41 +/- 854.73 | 2662.37 +/- 250.93 | 2713.15 +/- 571.11 | 3101.67 +/- 515.09 | 2123.78 +/- 776.01 | 2390.23 +/- 778.98 | 2629.46 +/- 548.22 | 2778.34 +/- 677.82 |

## Axis overview

| Axis | Model | Nominal return | Mean gain over perturbed scenarios | Positive gain scenarios |
| --- | --- | --- | --- | --- |
| mass | TV cap=2.85 | 2020.63 +/- 891.50 | +907.21 | 14/14 |
| mass | TV cap=2.95 | 3056.41 +/- 854.73 | +112.02 | 9/14 |
| mass | TV cap=3.00 | 2662.37 +/- 250.93 | +330.36 | 13/14 |
| mass | TV cap=3.05 | 2713.15 +/- 571.11 | +458.58 | 14/14 |
| mass | TV cap=3.10 | 3101.67 +/- 515.09 | -54.89 | 4/14 |
| mass | TV cap=3.20 | 2123.78 +/- 776.01 | +753.41 | 14/14 |
| mass | TV cap=3.50 | 2390.23 +/- 778.98 | +409.81 | 13/14 |
| mass | TV cap=3.70 | 2629.46 +/- 548.22 | +404.18 | 13/14 |
| mass | TV cap=4.00 | 2778.34 +/- 677.82 | +316.61 | 14/14 |

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
