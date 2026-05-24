# PPO_Walker_single_axis_physical_nonmass_0p0_2p0_20260430

Date: 2026-05-15

This folder packages the final pinned robustness evaluation for normalized PPO.

## Contents

- Raw metrics: `/Users/arjunmahesh/Library/CloudStorage/OneDrive-Personal/Duke/Research/RL/cleanrl/sweeps/results/PPO_Walker_final_presentable_20260430/raw_metrics/single_axis_nonmass`
- Aggregated outputs: `/Users/arjunmahesh/Library/CloudStorage/OneDrive-Personal/Duke/Research/RL/cleanrl/sweeps/results/PPO_Walker_single_axis_physical_nonmass_0p0_2p0_20260430/outputs`
- Plots: `/Users/arjunmahesh/Library/CloudStorage/OneDrive-Personal/Duke/Research/RL/cleanrl/sweeps/results/PPO_Walker_single_axis_physical_nonmass_0p0_2p0_20260430/plots`

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
| actuator_gain | 2899.94 +/- 635.61 | 2157.10 +/- 940.79 | 2992.57 +/- 813.21 | 2604.13 +/- 308.91 | 2827.30 +/- 665.12 | 3185.52 +/- 328.32 | 2080.49 +/- 630.99 | 2375.83 +/- 808.70 | 2308.53 +/- 474.50 | 2777.34 +/- 594.61 |
| damping | 2875.61 +/- 726.72 | 2122.66 +/- 903.09 | 2818.40 +/- 840.04 | 2679.87 +/- 441.61 | 2502.75 +/- 501.17 | 3251.24 +/- 455.99 | 2192.57 +/- 830.14 | 2454.42 +/- 1015.88 | 2571.38 +/- 727.41 | 2965.52 +/- 429.99 |
| friction | 3080.16 +/- 619.67 | 2082.35 +/- 1023.48 | 2845.46 +/- 861.80 | 2708.46 +/- 321.72 | 2661.14 +/- 542.03 | 3083.33 +/- 539.40 | 2082.95 +/- 866.33 | 2448.42 +/- 873.95 | 2449.34 +/- 513.69 | 3025.32 +/- 513.11 |

## Axis overview

| Axis | Model | Nominal return | Mean gain over perturbed scenarios | Positive gain scenarios |
| --- | --- | --- | --- | --- |
| actuator_gain | TV cap=2.85 | 2157.10 +/- 940.79 | +681.60 | 15/15 |
| actuator_gain | TV cap=2.95 | 2992.57 +/- 813.21 | -3.96 | 5/15 |
| actuator_gain | TV cap=3.00 | 2604.13 +/- 308.91 | +263.30 | 13/15 |
| actuator_gain | TV cap=3.05 | 2827.30 +/- 665.12 | +189.95 | 13/15 |
| actuator_gain | TV cap=3.10 | 3185.52 +/- 328.32 | -244.27 | 1/15 |
| actuator_gain | TV cap=3.20 | 2080.49 +/- 630.99 | +710.69 | 15/15 |
| actuator_gain | TV cap=3.50 | 2375.83 +/- 808.70 | +377.01 | 12/15 |
| actuator_gain | TV cap=3.70 | 2308.53 +/- 474.50 | +575.46 | 15/15 |
| actuator_gain | TV cap=4.00 | 2777.34 +/- 594.61 | +226.43 | 15/15 |
| damping | TV cap=2.85 | 2122.66 +/- 903.09 | -10.58 | 9/15 |
| damping | TV cap=2.95 | 2818.40 +/- 840.04 | +73.31 | 11/15 |
| damping | TV cap=3.00 | 2679.87 +/- 441.61 | -12.99 | 5/15 |
| damping | TV cap=3.05 | 2502.75 +/- 501.17 | +161.15 | 13/15 |
| damping | TV cap=3.10 | 3251.24 +/- 455.99 | -276.39 | 0/15 |
| damping | TV cap=3.20 | 2192.57 +/- 830.14 | +25.60 | 9/15 |
| damping | TV cap=3.50 | 2454.42 +/- 1015.88 | +25.57 | 9/15 |
| damping | TV cap=3.70 | 2571.38 +/- 727.41 | +2.37 | 8/15 |
| damping | TV cap=4.00 | 2965.52 +/- 429.99 | -23.50 | 9/15 |
| friction | TV cap=2.85 | 2082.35 +/- 1023.48 | +781.43 | 13/15 |
| friction | TV cap=2.95 | 2845.46 +/- 861.80 | +553.64 | 14/15 |
| friction | TV cap=3.00 | 2708.46 +/- 321.72 | +415.76 | 14/15 |
| friction | TV cap=3.05 | 2661.14 +/- 542.03 | +467.26 | 15/15 |
| friction | TV cap=3.10 | 3083.33 +/- 539.40 | +337.69 | 14/15 |
| friction | TV cap=3.20 | 2082.95 +/- 866.33 | +804.87 | 15/15 |
| friction | TV cap=3.50 | 2448.42 +/- 873.95 | +489.12 | 13/15 |
| friction | TV cap=3.70 | 2449.34 +/- 513.69 | +599.20 | 15/15 |
| friction | TV cap=4.00 | 3025.32 +/- 513.11 | +198.29 | 13/15 |

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

## Output CSV files

- `outputs/eval_metrics_final.csv`: latest merged per-seed eval rows.
- `outputs/summary_by_scenario.csv`: per-model, per-axis, per-factor aggregate return table.
- `outputs/drop_summary.csv`: nominal-minus-perturbed drop table.
- `outputs/gain_summary.csv`: paired robust-gain table against vanilla.
- `outputs/axis_overview.csv`: compact axis-level overview.
- `outputs/curve_points.csv`: same data used for return plots.
- `outputs/gain_curve_points.csv`: same data used for gain plots.
