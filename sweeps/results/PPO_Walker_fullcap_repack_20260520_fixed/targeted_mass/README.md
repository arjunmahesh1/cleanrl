# Walker2d PPO full-cap repack fixed: targeted_mass

Date: 2026-05-20

This folder packages the final pinned robustness evaluation for normalized PPO.

## Contents

- Raw metrics: `/Users/arjunmahesh/Library/CloudStorage/OneDrive-Personal/Duke/Research/RL/cleanrl/sweeps/results/PPO_Walker_final_presentable_evalfix_20260511/regeneration_sources/targeted_mass/raw_metrics`
- Aggregated outputs: `/Users/arjunmahesh/Library/CloudStorage/OneDrive-Personal/Duke/Research/RL/cleanrl/sweeps/results/PPO_Walker_fullcap_repack_20260520_fixed/targeted_mass/outputs`
- Plots: `/Users/arjunmahesh/Library/CloudStorage/OneDrive-Personal/Duke/Research/RL/cleanrl/sweeps/results/PPO_Walker_fullcap_repack_20260520_fixed/targeted_mass/plots`

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
| foot_left_mass | 3030.54 +/- 790.55 | 2728.36 +/- 1101.71 | 3522.26 +/- 977.51 | 3108.41 +/- 876.52 | 3675.46 +/- 565.48 | 3396.70 +/- 302.75 | 3095.00 +/- 977.59 | 2853.47 +/- 1024.29 | 2723.84 +/- 823.11 | 3671.75 +/- 324.52 |
| leg_left_mass | 3030.54 +/- 790.55 | 2728.36 +/- 1101.71 | 3522.26 +/- 977.51 | 3108.41 +/- 876.52 | 3675.46 +/- 565.48 | 3396.70 +/- 302.75 | 3095.00 +/- 977.59 | 2853.47 +/- 1024.29 | 2723.84 +/- 823.11 | 3671.75 +/- 324.52 |
| thigh_left_mass | 3030.54 +/- 790.55 | 2728.36 +/- 1101.71 | 3522.26 +/- 977.51 | 3108.41 +/- 876.52 | 3675.46 +/- 565.48 | 3396.70 +/- 302.75 | 3095.00 +/- 977.59 | 2853.47 +/- 1024.29 | 2723.84 +/- 823.11 | 3671.75 +/- 324.52 |

## Axis overview

| Axis | Model | Nominal return | Mean gain over perturbed scenarios | Positive gain scenarios |
| --- | --- | --- | --- | --- |
| foot_left_mass | TV cap=2.85 | 2728.36 +/- 1101.71 | -67.47 | 7/14 |
| foot_left_mass | TV cap=2.95 | 3522.26 +/- 977.51 | -157.50 | 6/14 |
| foot_left_mass | TV cap=3.00 | 3108.41 +/- 876.52 | +108.16 | 9/14 |
| foot_left_mass | TV cap=3.05 | 3675.46 +/- 565.48 | -392.67 | 4/14 |
| foot_left_mass | TV cap=3.10 | 3396.70 +/- 302.75 | +147.73 | 8/14 |
| foot_left_mass | TV cap=3.20 | 3095.00 +/- 977.59 | -299.28 | 5/14 |
| foot_left_mass | TV cap=3.50 | 2853.47 +/- 1024.29 | +55.10 | 7/14 |
| foot_left_mass | TV cap=3.70 | 2723.84 +/- 823.11 | -20.26 | 6/14 |
| foot_left_mass | TV cap=4.00 | 3671.75 +/- 324.52 | -656.74 | 3/14 |
| leg_left_mass | TV cap=2.85 | 2728.36 +/- 1101.71 | +415.46 | 10/14 |
| leg_left_mass | TV cap=2.95 | 3522.26 +/- 977.51 | +286.54 | 8/14 |
| leg_left_mass | TV cap=3.00 | 3108.41 +/- 876.52 | +452.54 | 9/14 |
| leg_left_mass | TV cap=3.05 | 3675.46 +/- 565.48 | +260.90 | 9/14 |
| leg_left_mass | TV cap=3.10 | 3396.70 +/- 302.75 | +316.77 | 8/14 |
| leg_left_mass | TV cap=3.20 | 3095.00 +/- 977.59 | +84.71 | 8/14 |
| leg_left_mass | TV cap=3.50 | 2853.47 +/- 1024.29 | +319.28 | 12/14 |
| leg_left_mass | TV cap=3.70 | 2723.84 +/- 823.11 | +418.31 | 8/14 |
| leg_left_mass | TV cap=4.00 | 3671.75 +/- 324.52 | +206.44 | 9/14 |
| thigh_left_mass | TV cap=2.85 | 2728.36 +/- 1101.71 | -245.06 | 2/14 |
| thigh_left_mass | TV cap=2.95 | 3522.26 +/- 977.51 | -397.12 | 2/14 |
| thigh_left_mass | TV cap=3.00 | 3108.41 +/- 876.52 | -86.39 | 8/14 |
| thigh_left_mass | TV cap=3.05 | 3675.46 +/- 565.48 | -386.80 | 1/14 |
| thigh_left_mass | TV cap=3.10 | 3396.70 +/- 302.75 | -43.52 | 9/14 |
| thigh_left_mass | TV cap=3.20 | 3095.00 +/- 977.59 | -341.70 | 0/14 |
| thigh_left_mass | TV cap=3.50 | 2853.47 +/- 1024.29 | +162.78 | 11/14 |
| thigh_left_mass | TV cap=3.70 | 2723.84 +/- 823.11 | +193.84 | 9/14 |
| thigh_left_mass | TV cap=4.00 | 3671.75 +/- 324.52 | -430.43 | 0/14 |

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
- `plots/with_variance/foot_left_mass_return_curve.png`
- `plots/with_variance/foot_left_mass_return_curve.pdf`
- `plots/with_variance/foot_left_mass_gain_curve.png`
- `plots/with_variance/foot_left_mass_gain_curve.pdf`
- `plots/without_variance/foot_left_mass_return_curve.png`
- `plots/without_variance/foot_left_mass_return_curve.pdf`
- `plots/without_variance/foot_left_mass_gain_curve.png`
- `plots/without_variance/foot_left_mass_gain_curve.pdf`
- `plots/with_variance/leg_left_mass_return_curve.png`
- `plots/with_variance/leg_left_mass_return_curve.pdf`
- `plots/with_variance/leg_left_mass_gain_curve.png`
- `plots/with_variance/leg_left_mass_gain_curve.pdf`
- `plots/without_variance/leg_left_mass_return_curve.png`
- `plots/without_variance/leg_left_mass_return_curve.pdf`
- `plots/without_variance/leg_left_mass_gain_curve.png`
- `plots/without_variance/leg_left_mass_gain_curve.pdf`
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
