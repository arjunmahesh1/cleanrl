# PPO_AlphaGridNorm_expanded_eval_20260404

Date: 2026-04-15

This folder packages the final pinned robustness evaluation for normalized PPO.

## Contents

- Raw metrics: `/home/users/am1015/cleanrl/sweeps/results/PPO_AlphaGridNorm_expanded_eval_20260404/raw_metrics`
- Aggregated outputs: `/home/users/am1015/cleanrl/sweeps/results/PPO_AlphaGridNorm_expanded_eval_20260404/outputs`
- Plots: `/home/users/am1015/cleanrl/sweeps/results/PPO_AlphaGridNorm_expanded_eval_20260404/plots`

## Evaluation protocol

- Models are compared on the same perturbation grid for the configured axes.
- Nominal reference within each axis is the `factor=1.0` point.
- Curves show mean return across seeds with `95% CI` shading.
- Robust gain is defined as `vanilla_drop - model_drop`; positive is better.

## Model labels

- `vanilla` -> Vanilla
- `a1e9` -> No-op (1e9)
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

| Axis | Vanilla | No-op (1e9) | TV cap=2.85 | TV cap=2.95 | TV cap=3.00 | TV cap=3.05 | TV cap=3.10 | TV cap=3.20 | TV cap=3.50 | TV cap=3.70 | TV cap=4.00 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| damping | 2861.93 +/- 703.87 | 2814.89 +/- 540.19 | 2256.24 +/- 1100.71 | 3094.57 +/- 953.27 | 2645.71 +/- 325.67 | 2400.00 +/- 571.46 | 2987.04 +/- 367.25 | 2150.07 +/- 705.59 | 2326.56 +/- 904.14 | 2456.24 +/- 540.31 | 3034.12 +/- 434.35 |
| friction | 2984.94 +/- 810.65 | 3042.52 +/- 729.30 | 2126.99 +/- 905.28 | 2674.42 +/- 749.14 | 2606.77 +/- 319.53 | 2773.93 +/- 698.21 | 2967.12 +/- 592.43 | 1957.99 +/- 673.57 | 2368.42 +/- 909.18 | 2562.58 +/- 728.76 | 2986.01 +/- 403.99 |
| mass | 2737.10 +/- 394.27 | 2824.22 +/- 451.24 | 2129.58 +/- 1004.14 | 2875.69 +/- 780.02 | 2483.04 +/- 394.36 | 2822.36 +/- 462.50 | 3300.94 +/- 301.90 | 2188.79 +/- 854.10 | 2574.11 +/- 692.53 | 2412.53 +/- 498.13 | 3063.91 +/- 578.09 |

## Axis overview

| Axis | Model | Nominal return | Mean gain over perturbed scenarios | Positive gain scenarios |
| --- | --- | --- | --- | --- |
| damping | No-op (1e9) | 2814.89 +/- 540.19 | -36.29 | 4/10 |
| damping | TV cap=2.85 | 2256.24 +/- 1100.71 | -183.12 | 1/10 |
| damping | TV cap=2.95 | 3094.57 +/- 953.27 | -225.92 | 0/10 |
| damping | TV cap=3.00 | 2645.71 +/- 325.67 | +7.43 | 6/10 |
| damping | TV cap=3.05 | 2400.00 +/- 571.46 | +212.36 | 10/10 |
| damping | TV cap=3.10 | 2987.04 +/- 367.25 | +5.13 | 5/10 |
| damping | TV cap=3.20 | 2150.07 +/- 705.59 | +38.69 | 7/10 |
| damping | TV cap=3.50 | 2326.56 +/- 904.14 | +139.09 | 9/10 |
| damping | TV cap=3.70 | 2456.24 +/- 540.31 | +107.56 | 7/10 |
| damping | TV cap=4.00 | 3034.12 +/- 434.35 | -76.33 | 3/10 |
| friction | No-op (1e9) | 3042.52 +/- 729.30 | -90.13 | 2/10 |
| friction | TV cap=2.85 | 2126.99 +/- 905.28 | +472.51 | 8/10 |
| friction | TV cap=2.95 | 2674.42 +/- 749.14 | +728.87 | 10/10 |
| friction | TV cap=3.00 | 2606.77 +/- 319.53 | +392.12 | 10/10 |
| friction | TV cap=3.05 | 2773.93 +/- 698.21 | +243.26 | 10/10 |
| friction | TV cap=3.10 | 2967.12 +/- 592.43 | +355.00 | 9/10 |
| friction | TV cap=3.20 | 1957.99 +/- 673.57 | +720.21 | 10/10 |
| friction | TV cap=3.50 | 2368.42 +/- 909.18 | +369.35 | 8/10 |
| friction | TV cap=3.70 | 2562.58 +/- 728.76 | +315.14 | 10/10 |
| friction | TV cap=4.00 | 2986.01 +/- 403.99 | +169.27 | 7/10 |
| mass | No-op (1e9) | 2824.22 +/- 451.24 | -107.65 | 1/10 |
| mass | TV cap=2.85 | 2129.58 +/- 1004.14 | +520.55 | 10/10 |
| mass | TV cap=2.95 | 2875.69 +/- 780.02 | +25.67 | 5/10 |
| mass | TV cap=3.00 | 2483.04 +/- 394.36 | +209.84 | 8/10 |
| mass | TV cap=3.05 | 2822.36 +/- 462.50 | +89.60 | 6/10 |
| mass | TV cap=3.10 | 3300.94 +/- 301.90 | -467.74 | 0/10 |
| mass | TV cap=3.20 | 2188.79 +/- 854.10 | +378.84 | 9/10 |
| mass | TV cap=3.50 | 2574.11 +/- 692.53 | -47.76 | 5/10 |
| mass | TV cap=3.70 | 2412.53 +/- 498.13 | +300.95 | 10/10 |
| mass | TV cap=4.00 | 3063.91 +/- 578.09 | -176.30 | 2/10 |

## Plot files

- `plots/return_curves_panel.png`
- `plots/gain_curves_panel.png`
- `plots/damping_return_curve.png`
- `plots/damping_gain_curve.png`
- `plots/friction_return_curve.png`
- `plots/friction_gain_curve.png`
- `plots/mass_return_curve.png`
- `plots/mass_gain_curve.png`

## Output CSV files

- `outputs/eval_metrics_final.csv`: latest merged per-seed eval rows.
- `outputs/summary_by_scenario.csv`: per-model, per-axis, per-factor aggregate return table.
- `outputs/drop_summary.csv`: nominal-minus-perturbed drop table.
- `outputs/gain_summary.csv`: paired robust-gain table against vanilla.
- `outputs/axis_overview.csv`: compact axis-level overview.
- `outputs/curve_points.csv`: same data used for return plots.
- `outputs/gain_curve_points.csv`: same data used for gain plots.
