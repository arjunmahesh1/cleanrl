# PPO_Walker_targeted_nonmass_0p0_2p0_20260430

Date: 2026-05-15

This folder packages the final pinned robustness evaluation for normalized PPO.

## Contents

- Raw metrics: `/Users/arjunmahesh/Library/CloudStorage/OneDrive-Personal/Duke/Research/RL/cleanrl/sweeps/results/PPO_Walker_final_presentable_20260430/raw_metrics/targeted_nonmass`
- Aggregated outputs: `/Users/arjunmahesh/Library/CloudStorage/OneDrive-Personal/Duke/Research/RL/cleanrl/sweeps/results/PPO_Walker_targeted_nonmass_0p0_2p0_20260430/outputs`
- Plots: `/Users/arjunmahesh/Library/CloudStorage/OneDrive-Personal/Duke/Research/RL/cleanrl/sweeps/results/PPO_Walker_targeted_nonmass_0p0_2p0_20260430/plots`

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

## Nominal returns by axis

| Axis | Vanilla | TV cap=2.85 | TV cap=2.95 | TV cap=3.00 | TV cap=3.05 |
| --- | --- | --- | --- | --- | --- |
| foot_left_actuator_gain | 2853.12 +/- 509.93 | 2240.01 +/- 1055.07 | 2857.79 +/- 870.42 | 2910.95 +/- 325.50 | n/a |
| foot_left_damping | 2975.67 +/- 570.98 | 2016.94 +/- 910.32 | 2951.50 +/- 933.51 | 2679.19 +/- 329.07 | n/a |
| foot_left_friction | 2822.36 +/- 701.19 | 2017.21 +/- 859.70 | 3057.39 +/- 924.02 | 2559.16 +/- 296.64 | n/a |
| leg_left_actuator_gain | 2898.94 +/- 601.16 | 2038.20 +/- 870.57 | 2929.70 +/- 828.01 | 2752.01 +/- 336.16 | n/a |
| leg_left_damping | 2763.58 +/- 609.16 | 2177.06 +/- 1084.45 | 2716.68 +/- 650.84 | 2646.11 +/- 289.49 | n/a |
| thigh_left_actuator_gain | 2816.25 +/- 537.65 | 2037.93 +/- 887.33 | 2897.25 +/- 893.74 | 2500.18 +/- 197.96 | n/a |
| thigh_left_damping | 2795.41 +/- 447.00 | 2217.13 +/- 1078.17 | 2988.85 +/- 904.67 | 2430.02 +/- 450.07 | 2553.61 +/- 732.53 |

## Axis overview

| Axis | Model | Nominal return | Mean gain over perturbed scenarios | Positive gain scenarios |
| --- | --- | --- | --- | --- |
| foot_left_actuator_gain | TV cap=2.85 | 2240.01 +/- 1055.07 | +559.90 | 12/15 |
| foot_left_actuator_gain | TV cap=2.95 | 2857.79 +/- 870.42 | -33.84 | 8/15 |
| foot_left_actuator_gain | TV cap=3.00 | 2910.95 +/- 325.50 | +219.28 | 9/15 |
| foot_left_damping | TV cap=2.85 | 2016.94 +/- 910.32 | +256.93 | 15/15 |
| foot_left_damping | TV cap=2.95 | 2951.50 +/- 933.51 | +114.94 | 13/15 |
| foot_left_damping | TV cap=3.00 | 2679.19 +/- 329.07 | +161.99 | 12/15 |
| foot_left_friction | TV cap=2.85 | 2017.21 +/- 859.70 | +712.66 | 12/15 |
| foot_left_friction | TV cap=2.95 | 3057.39 +/- 924.02 | +316.39 | 14/15 |
| foot_left_friction | TV cap=3.00 | 2559.16 +/- 296.64 | +723.09 | 14/15 |
| leg_left_actuator_gain | TV cap=2.85 | 2038.20 +/- 870.57 | +540.99 | 15/15 |
| leg_left_actuator_gain | TV cap=2.95 | 2929.70 +/- 828.01 | +182.10 | 13/15 |
| leg_left_actuator_gain | TV cap=3.00 | 2752.01 +/- 336.16 | +304.78 | 13/15 |
| leg_left_damping | TV cap=2.85 | 2177.06 +/- 1084.45 | -111.98 | 2/15 |
| leg_left_damping | TV cap=2.95 | 2716.68 +/- 650.84 | +120.00 | 12/15 |
| leg_left_damping | TV cap=3.00 | 2646.11 +/- 289.49 | -73.15 | 6/15 |
| thigh_left_actuator_gain | TV cap=2.85 | 2037.93 +/- 887.33 | +164.31 | 8/15 |
| thigh_left_actuator_gain | TV cap=2.95 | 2897.25 +/- 893.74 | -453.67 | 0/15 |
| thigh_left_actuator_gain | TV cap=3.00 | 2500.18 +/- 197.96 | +351.80 | 13/15 |
| thigh_left_damping | TV cap=2.85 | 2217.13 +/- 1078.17 | -99.81 | 2/15 |
| thigh_left_damping | TV cap=2.95 | 2988.85 +/- 904.67 | -87.39 | 4/15 |
| thigh_left_damping | TV cap=3.00 | 2430.02 +/- 450.07 | +257.92 | 14/15 |
| thigh_left_damping | TV cap=3.05 | 2553.61 +/- 732.53 | +195.75 | 14/15 |

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

## Output CSV files

- `outputs/eval_metrics_final.csv`: latest merged per-seed eval rows.
- `outputs/summary_by_scenario.csv`: per-model, per-axis, per-factor aggregate return table.
- `outputs/drop_summary.csv`: nominal-minus-perturbed drop table.
- `outputs/gain_summary.csv`: paired robust-gain table against vanilla.
- `outputs/axis_overview.csv`: compact axis-level overview.
- `outputs/curve_points.csv`: same data used for return plots.
- `outputs/gain_curve_points.csv`: same data used for gain plots.
