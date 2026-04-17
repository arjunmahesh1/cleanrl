# PPO_Adversarial_fmd_combo_20260406

Date: 2026-04-15

This folder packages the final pinned robustness evaluation for normalized PPO.

## Contents

- Raw metrics: `/home/users/am1015/cleanrl/sweeps/results/PPO_Adversarial_fmd_combo_20260406/raw_metrics`
- Aggregated outputs: `/home/users/am1015/cleanrl/sweeps/results/PPO_Adversarial_fmd_combo_20260406/outputs`
- Plots: `/home/users/am1015/cleanrl/sweeps/results/PPO_Adversarial_fmd_combo_20260406/plots`

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
| friction_damping | 2748.93 +/- 355.24 | 2763.51 +/- 547.30 | 2041.06 +/- 944.25 | 2746.22 +/- 726.02 | 2592.15 +/- 317.65 | 2527.85 +/- 567.23 | 3093.57 +/- 338.63 | 2111.78 +/- 875.76 | 2323.77 +/- 757.27 | 2656.25 +/- 694.24 | 2949.91 +/- 485.09 |
| friction_mass | 2975.47 +/- 671.99 | 2720.33 +/- 636.52 | 2110.79 +/- 1142.27 | 2841.01 +/- 650.94 | 2751.57 +/- 337.51 | 2656.44 +/- 595.73 | 3043.18 +/- 481.13 | 2086.13 +/- 706.75 | 2306.42 +/- 652.15 | 2530.54 +/- 717.62 | 2860.08 +/- 472.31 |
| friction_mass_damping | 2912.94 +/- 475.98 | 2781.84 +/- 503.43 | 2135.93 +/- 939.98 | 2766.15 +/- 804.58 | 2820.94 +/- 568.51 | 2638.70 +/- 616.03 | 3139.48 +/- 433.66 | 2189.79 +/- 878.60 | 2413.59 +/- 873.91 | 2600.46 +/- 600.09 | 2986.47 +/- 339.50 |
| mass_damping | 2565.32 +/- 510.99 | 2907.31 +/- 443.43 | 2234.95 +/- 1085.62 | 2839.65 +/- 811.31 | 2744.09 +/- 411.33 | 2857.99 +/- 652.32 | 3036.57 +/- 387.21 | 2246.82 +/- 797.25 | 2358.16 +/- 831.42 | 2543.34 +/- 758.69 | 2854.26 +/- 540.57 |

## Axis overview

| Axis | Model | Nominal return | Mean gain over perturbed scenarios | Positive gain scenarios |
| --- | --- | --- | --- | --- |
| friction_damping | No-op (1e9) | 2763.51 +/- 547.30 | +2.86 | 5/10 |
| friction_damping | TV cap=2.85 | 2041.06 +/- 944.25 | +366.30 | 7/10 |
| friction_damping | TV cap=2.95 | 2746.22 +/- 726.02 | +442.08 | 10/10 |
| friction_damping | TV cap=3.00 | 2592.15 +/- 317.65 | +171.99 | 5/10 |
| friction_damping | TV cap=3.05 | 2527.85 +/- 567.23 | +238.15 | 10/10 |
| friction_damping | TV cap=3.10 | 3093.57 +/- 338.63 | +64.44 | 4/10 |
| friction_damping | TV cap=3.20 | 2111.78 +/- 875.76 | +330.79 | 8/10 |
| friction_damping | TV cap=3.50 | 2323.77 +/- 757.27 | +198.16 | 8/10 |
| friction_damping | TV cap=3.70 | 2656.25 +/- 694.24 | -34.04 | 4/10 |
| friction_damping | TV cap=4.00 | 2949.91 +/- 485.09 | -69.15 | 4/10 |
| friction_mass | No-op (1e9) | 2720.33 +/- 636.52 | +246.42 | 10/10 |
| friction_mass | TV cap=2.85 | 2110.79 +/- 1142.27 | +596.10 | 10/10 |
| friction_mass | TV cap=2.95 | 2841.01 +/- 650.94 | +112.24 | 8/10 |
| friction_mass | TV cap=3.00 | 2751.57 +/- 337.51 | +220.71 | 8/10 |
| friction_mass | TV cap=3.05 | 2656.44 +/- 595.73 | +398.72 | 10/10 |
| friction_mass | TV cap=3.10 | 3043.18 +/- 481.13 | -109.73 | 5/10 |
| friction_mass | TV cap=3.20 | 2086.13 +/- 706.75 | +554.18 | 10/10 |
| friction_mass | TV cap=3.50 | 2306.42 +/- 652.15 | +484.21 | 9/10 |
| friction_mass | TV cap=3.70 | 2530.54 +/- 717.62 | +438.95 | 10/10 |
| friction_mass | TV cap=4.00 | 2860.08 +/- 472.31 | +235.18 | 10/10 |
| friction_mass_damping | No-op (1e9) | 2781.84 +/- 503.43 | +125.88 | 9/10 |
| friction_mass_damping | TV cap=2.85 | 2135.93 +/- 939.98 | +581.23 | 10/10 |
| friction_mass_damping | TV cap=2.95 | 2766.15 +/- 804.58 | +185.40 | 8/10 |
| friction_mass_damping | TV cap=3.00 | 2820.94 +/- 568.51 | +181.25 | 9/10 |
| friction_mass_damping | TV cap=3.05 | 2638.70 +/- 616.03 | +420.56 | 10/10 |
| friction_mass_damping | TV cap=3.10 | 3139.48 +/- 433.66 | -153.55 | 2/10 |
| friction_mass_damping | TV cap=3.20 | 2189.79 +/- 878.60 | +546.78 | 10/10 |
| friction_mass_damping | TV cap=3.50 | 2413.59 +/- 873.91 | +371.74 | 9/10 |
| friction_mass_damping | TV cap=3.70 | 2600.46 +/- 600.09 | +417.99 | 10/10 |
| friction_mass_damping | TV cap=4.00 | 2986.47 +/- 339.50 | +89.03 | 5/10 |
| mass_damping | No-op (1e9) | 2907.31 +/- 443.43 | -350.99 | 0/10 |
| mass_damping | TV cap=2.85 | 2234.95 +/- 1085.62 | +238.84 | 9/10 |
| mass_damping | TV cap=2.95 | 2839.65 +/- 811.31 | -135.91 | 3/10 |
| mass_damping | TV cap=3.00 | 2744.09 +/- 411.33 | -290.17 | 0/10 |
| mass_damping | TV cap=3.05 | 2857.99 +/- 652.32 | -177.75 | 0/10 |
| mass_damping | TV cap=3.10 | 3036.57 +/- 387.21 | -373.93 | 1/10 |
| mass_damping | TV cap=3.20 | 2246.82 +/- 797.25 | +149.31 | 7/10 |
| mass_damping | TV cap=3.50 | 2358.16 +/- 831.42 | -35.48 | 6/10 |
| mass_damping | TV cap=3.70 | 2543.34 +/- 758.69 | -3.97 | 7/10 |
| mass_damping | TV cap=4.00 | 2854.26 +/- 540.57 | -130.19 | 2/10 |

## Plot files

- `plots/return_curves_panel.png`
- `plots/gain_curves_panel.png`
- `plots/friction_damping_return_curve.png`
- `plots/friction_damping_gain_curve.png`
- `plots/friction_mass_return_curve.png`
- `plots/friction_mass_gain_curve.png`
- `plots/friction_mass_damping_return_curve.png`
- `plots/friction_mass_damping_gain_curve.png`
- `plots/mass_damping_return_curve.png`
- `plots/mass_damping_gain_curve.png`

## Output CSV files

- `outputs/eval_metrics_final.csv`: latest merged per-seed eval rows.
- `outputs/summary_by_scenario.csv`: per-model, per-axis, per-factor aggregate return table.
- `outputs/drop_summary.csv`: nominal-minus-perturbed drop table.
- `outputs/gain_summary.csv`: paired robust-gain table against vanilla.
- `outputs/axis_overview.csv`: compact axis-level overview.
- `outputs/curve_points.csv`: same data used for return plots.
- `outputs/gain_curve_points.csv`: same data used for gain plots.
