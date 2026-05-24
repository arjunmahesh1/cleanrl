# PPO_Walker_targeted_mass_0p1_2p0_20260430

Date: 2026-05-15

This folder packages the final pinned robustness evaluation for normalized PPO.

## Contents

- Raw metrics: `/Users/arjunmahesh/Library/CloudStorage/OneDrive-Personal/Duke/Research/RL/cleanrl/sweeps/results/PPO_Walker_final_presentable_20260430/raw_metrics/targeted_mass`
- Aggregated outputs: `/Users/arjunmahesh/Library/CloudStorage/OneDrive-Personal/Duke/Research/RL/cleanrl/sweeps/results/PPO_Walker_targeted_mass_0p1_2p0_20260430/outputs`
- Plots: `/Users/arjunmahesh/Library/CloudStorage/OneDrive-Personal/Duke/Research/RL/cleanrl/sweeps/results/PPO_Walker_targeted_mass_0p1_2p0_20260430/plots`

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

## Nominal returns by axis

| Axis | Vanilla | TV cap=2.85 | TV cap=2.95 | TV cap=3.00 |
| --- | --- | --- | --- | --- |
| foot_left_mass | 2499.90 +/- 640.73 | 2138.24 +/- 1006.29 | 2872.49 +/- 833.79 | n/a |
| leg_left_mass | 2783.88 +/- 329.82 | 2274.81 +/- 1014.57 | 2966.29 +/- 807.71 | n/a |
| thigh_left_mass | 2787.54 +/- 561.70 | 2228.11 +/- 1020.62 | 2997.97 +/- 1010.38 | 2735.59 +/- 477.37 |

## Axis overview

| Axis | Model | Nominal return | Mean gain over perturbed scenarios | Positive gain scenarios |
| --- | --- | --- | --- | --- |
| foot_left_mass | TV cap=2.85 | 2138.24 +/- 1006.29 | -20.54 | 5/14 |
| foot_left_mass | TV cap=2.95 | 2872.49 +/- 833.79 | -126.98 | 4/14 |
| leg_left_mass | TV cap=2.85 | 2274.81 +/- 1014.57 | +58.55 | 8/14 |
| leg_left_mass | TV cap=2.95 | 2966.29 +/- 807.71 | +43.67 | 7/14 |
| thigh_left_mass | TV cap=2.85 | 2228.11 +/- 1020.62 | +133.06 | 9/14 |
| thigh_left_mass | TV cap=2.95 | 2997.97 +/- 1010.38 | -217.86 | 2/14 |
| thigh_left_mass | TV cap=3.00 | 2735.59 +/- 477.37 | +25.95 | 7/14 |

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
