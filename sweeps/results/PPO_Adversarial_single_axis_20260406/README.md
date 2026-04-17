# PPO_Adversarial_single_axis_20260406

Date: 2026-04-15

This folder packages the final pinned robustness evaluation for normalized PPO.

## Contents

- Raw metrics: `/home/users/am1015/cleanrl/sweeps/results/PPO_Adversarial_single_axis_20260406/raw_metrics`
- Aggregated outputs: `/home/users/am1015/cleanrl/sweeps/results/PPO_Adversarial_single_axis_20260406/outputs`
- Plots: `/home/users/am1015/cleanrl/sweeps/results/PPO_Adversarial_single_axis_20260406/plots`

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
| actuator_gain | 2825.29 +/- 561.67 | 2747.35 +/- 488.04 | 2176.90 +/- 1007.55 | 2811.39 +/- 789.95 | 2647.97 +/- 341.31 | 2565.15 +/- 524.34 | 2903.00 +/- 370.40 | 2195.09 +/- 869.62 | 2536.36 +/- 868.29 | 2693.94 +/- 625.49 | 2897.85 +/- 589.94 |
| damping | 3085.79 +/- 724.17 | 2812.81 +/- 466.03 | 2356.86 +/- 1146.43 | 2988.83 +/- 1025.13 | 2767.72 +/- 405.59 | 2792.30 +/- 486.18 | 3002.59 +/- 471.75 | 2221.97 +/- 865.48 | 2499.82 +/- 843.23 | 2562.39 +/- 536.49 | 2775.06 +/- 648.35 |
| friction | 2893.81 +/- 718.92 | 2877.18 +/- 506.02 | 2124.67 +/- 899.31 | 2877.72 +/- 731.06 | 2735.35 +/- 540.63 | 2646.52 +/- 627.31 | 3054.44 +/- 589.88 | 2135.74 +/- 854.21 | 2511.84 +/- 869.72 | 2577.08 +/- 543.26 | 2838.74 +/- 534.53 |
| mass | 2822.59 +/- 670.25 | 2872.82 +/- 667.25 | 2209.60 +/- 1075.53 | 2960.62 +/- 1017.33 | 2621.56 +/- 322.89 | 2544.92 +/- 565.77 | 2850.11 +/- 396.08 | 2044.27 +/- 745.63 | 2613.36 +/- 929.77 | 2474.62 +/- 668.46 | 2922.14 +/- 507.95 |
| obs_noise | 356.87 +/- 37.54 | 356.16 +/- 39.64 | 328.54 +/- 50.35 | 354.62 +/- 100.45 | 345.76 +/- 48.54 | 347.09 +/- 24.12 | 372.58 +/- 44.46 | 354.46 +/- 54.88 | 372.58 +/- 50.99 | 376.08 +/- 39.19 | 363.73 +/- 36.27 |
| reward_noise | 2826.07 +/- 438.41 | 2950.04 +/- 508.22 | 2029.25 +/- 896.77 | 2854.41 +/- 748.33 | 2721.68 +/- 343.72 | 2667.91 +/- 605.35 | 2951.42 +/- 366.48 | 2184.77 +/- 829.66 | 2511.41 +/- 874.45 | 2384.30 +/- 553.01 | 2976.92 +/- 475.02 |

## Axis overview

| Axis | Model | Nominal return | Mean gain over perturbed scenarios | Positive gain scenarios |
| --- | --- | --- | --- | --- |
| actuator_gain | No-op (1e9) | 2747.35 +/- 488.04 | +6.83 | 7/10 |
| actuator_gain | TV cap=2.85 | 2176.90 +/- 1007.55 | +535.24 | 9/10 |
| actuator_gain | TV cap=2.95 | 2811.39 +/- 789.95 | +122.77 | 8/10 |
| actuator_gain | TV cap=3.00 | 2647.97 +/- 341.31 | +103.78 | 7/10 |
| actuator_gain | TV cap=3.05 | 2565.15 +/- 524.34 | +375.00 | 10/10 |
| actuator_gain | TV cap=3.10 | 2903.00 +/- 370.40 | -37.83 | 5/10 |
| actuator_gain | TV cap=3.20 | 2195.09 +/- 869.62 | +428.99 | 9/10 |
| actuator_gain | TV cap=3.50 | 2536.36 +/- 868.29 | -11.78 | 6/10 |
| actuator_gain | TV cap=3.70 | 2693.94 +/- 625.49 | +129.55 | 9/10 |
| actuator_gain | TV cap=4.00 | 2897.85 +/- 589.94 | +42.19 | 4/10 |
| damping | No-op (1e9) | 2812.81 +/- 466.03 | +165.58 | 9/10 |
| damping | TV cap=2.85 | 2356.86 +/- 1146.43 | -28.32 | 3/10 |
| damping | TV cap=2.95 | 2988.83 +/- 1025.13 | +72.47 | 9/10 |
| damping | TV cap=3.00 | 2767.72 +/- 405.59 | +123.22 | 9/10 |
| damping | TV cap=3.05 | 2792.30 +/- 486.18 | +144.66 | 7/10 |
| damping | TV cap=3.10 | 3002.59 +/- 471.75 | +267.79 | 10/10 |
| damping | TV cap=3.20 | 2221.97 +/- 865.48 | +173.93 | 10/10 |
| damping | TV cap=3.50 | 2499.82 +/- 843.23 | +142.97 | 9/10 |
| damping | TV cap=3.70 | 2562.39 +/- 536.49 | +207.20 | 9/10 |
| damping | TV cap=4.00 | 2775.06 +/- 648.35 | +350.76 | 10/10 |
| friction | No-op (1e9) | 2877.18 +/- 506.02 | +9.49 | 6/10 |
| friction | TV cap=2.85 | 2124.67 +/- 899.31 | +396.89 | 7/10 |
| friction | TV cap=2.95 | 2877.72 +/- 731.06 | +524.83 | 10/10 |
| friction | TV cap=3.00 | 2735.35 +/- 540.63 | +217.00 | 7/10 |
| friction | TV cap=3.05 | 2646.52 +/- 627.31 | +298.21 | 9/10 |
| friction | TV cap=3.10 | 3054.44 +/- 589.88 | +338.35 | 10/10 |
| friction | TV cap=3.20 | 2135.74 +/- 854.21 | +489.52 | 10/10 |
| friction | TV cap=3.50 | 2511.84 +/- 869.72 | +190.72 | 7/10 |
| friction | TV cap=3.70 | 2577.08 +/- 543.26 | +226.43 | 10/10 |
| friction | TV cap=4.00 | 2838.74 +/- 534.53 | +225.43 | 8/10 |
| mass | No-op (1e9) | 2872.82 +/- 667.25 | -58.63 | 2/10 |
| mass | TV cap=2.85 | 2209.60 +/- 1075.53 | +489.79 | 9/10 |
| mass | TV cap=2.95 | 2960.62 +/- 1017.33 | -15.61 | 4/10 |
| mass | TV cap=3.00 | 2621.56 +/- 322.89 | +107.90 | 6/10 |
| mass | TV cap=3.05 | 2544.92 +/- 565.77 | +387.60 | 10/10 |
| mass | TV cap=3.10 | 2850.11 +/- 396.08 | +48.06 | 7/10 |
| mass | TV cap=3.20 | 2044.27 +/- 745.63 | +592.99 | 10/10 |
| mass | TV cap=3.50 | 2613.36 +/- 929.77 | -88.68 | 5/10 |
| mass | TV cap=3.70 | 2474.62 +/- 668.46 | +361.06 | 9/10 |
| mass | TV cap=4.00 | 2922.14 +/- 507.95 | +53.63 | 5/10 |
| obs_noise | No-op (1e9) | 356.16 +/- 39.64 | +2.48 | 6/10 |
| obs_noise | TV cap=2.85 | 328.54 +/- 50.35 | +12.60 | 6/10 |
| obs_noise | TV cap=2.95 | 354.62 +/- 100.45 | +26.38 | 8/10 |
| obs_noise | TV cap=3.00 | 345.76 +/- 48.54 | +33.15 | 9/10 |
| obs_noise | TV cap=3.05 | 347.09 +/- 24.12 | +27.57 | 5/10 |
| obs_noise | TV cap=3.10 | 372.58 +/- 44.46 | +21.67 | 6/10 |
| obs_noise | TV cap=3.20 | 354.46 +/- 54.88 | -11.62 | 4/10 |
| obs_noise | TV cap=3.50 | 372.58 +/- 50.99 | +31.80 | 7/10 |
| obs_noise | TV cap=3.70 | 376.08 +/- 39.19 | +25.05 | 6/10 |
| obs_noise | TV cap=4.00 | 363.73 +/- 36.27 | +28.96 | 7/10 |
| reward_noise | No-op (1e9) | 2950.04 +/- 508.22 | -88.44 | 4/10 |
| reward_noise | TV cap=2.85 | 2029.25 +/- 896.77 | +81.67 | 6/10 |
| reward_noise | TV cap=2.95 | 2854.41 +/- 748.33 | +62.69 | 6/10 |
| reward_noise | TV cap=3.00 | 2721.68 +/- 343.72 | -44.90 | 4/10 |
| reward_noise | TV cap=3.05 | 2667.91 +/- 605.35 | +63.12 | 7/10 |
| reward_noise | TV cap=3.10 | 2951.42 +/- 366.48 | +79.90 | 7/10 |
| reward_noise | TV cap=3.20 | 2184.77 +/- 829.66 | +3.84 | 5/10 |
| reward_noise | TV cap=3.50 | 2511.41 +/- 874.45 | -69.63 | 3/10 |
| reward_noise | TV cap=3.70 | 2384.30 +/- 553.01 | +148.05 | 7/10 |
| reward_noise | TV cap=4.00 | 2976.92 +/- 475.02 | -78.28 | 5/10 |

## Plot files

- `plots/return_curves_panel.png`
- `plots/gain_curves_panel.png`
- `plots/actuator_gain_return_curve.png`
- `plots/actuator_gain_gain_curve.png`
- `plots/damping_return_curve.png`
- `plots/damping_gain_curve.png`
- `plots/friction_return_curve.png`
- `plots/friction_gain_curve.png`
- `plots/mass_return_curve.png`
- `plots/mass_gain_curve.png`
- `plots/obs_noise_return_curve.png`
- `plots/obs_noise_gain_curve.png`
- `plots/reward_noise_return_curve.png`
- `plots/reward_noise_gain_curve.png`

## Output CSV files

- `outputs/eval_metrics_final.csv`: latest merged per-seed eval rows.
- `outputs/summary_by_scenario.csv`: per-model, per-axis, per-factor aggregate return table.
- `outputs/drop_summary.csv`: nominal-minus-perturbed drop table.
- `outputs/gain_summary.csv`: paired robust-gain table against vanilla.
- `outputs/axis_overview.csv`: compact axis-level overview.
- `outputs/curve_points.csv`: same data used for return plots.
- `outputs/gain_curve_points.csv`: same data used for gain plots.
