# PPO_Adversarial_single_axis_20260406

Date: 2026-04-08

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
| actuator_bias | 2802.57 +/- 595.28 | 2599.83 +/- 452.74 | 2175.15 +/- 995.98 | 2993.21 +/- 918.91 | 2788.00 +/- 429.27 | 2688.57 +/- 634.97 | 3036.10 +/- 317.02 | 2253.92 +/- 837.34 | 2507.62 +/- 847.31 | 2523.79 +/- 488.52 | 2919.62 +/- 546.39 |
| actuator_gain | 2669.70 +/- 604.93 | 2657.55 +/- 532.70 | 2275.51 +/- 1104.57 | 3028.33 +/- 799.70 | 2718.59 +/- 402.25 | 2681.98 +/- 564.01 | 3034.00 +/- 494.97 | 2183.75 +/- 760.57 | 2555.88 +/- 882.15 | 2654.50 +/- 708.47 | 2970.52 +/- 452.23 |
| damping | 3085.79 +/- 724.17 | 2812.81 +/- 466.03 | 2356.86 +/- 1146.43 | 2988.83 +/- 1025.13 | 2767.72 +/- 405.59 | 2792.30 +/- 486.18 | 3002.59 +/- 471.75 | 2221.97 +/- 865.48 | 2499.82 +/- 843.23 | 2562.39 +/- 536.49 | 2775.06 +/- 648.35 |
| friction | 2893.81 +/- 718.92 | 2877.18 +/- 506.02 | 2124.67 +/- 899.31 | 2877.72 +/- 731.06 | 2735.35 +/- 540.63 | 2646.52 +/- 627.31 | 3054.44 +/- 589.88 | 2135.74 +/- 854.21 | 2511.84 +/- 869.72 | 2577.08 +/- 543.26 | 2838.74 +/- 534.53 |
| mass | 2617.13 +/- 490.32 | 2933.61 +/- 571.91 | 2193.47 +/- 1008.07 | 2941.68 +/- 849.96 | 2799.09 +/- 477.34 | 2605.49 +/- 456.81 | 2875.53 +/- 498.62 | 2093.85 +/- 750.15 | 2559.32 +/- 874.41 | 2619.37 +/- 628.14 | 2873.49 +/- 489.71 |
| obs_noise | 346.10 +/- 34.00 | 336.83 +/- 44.48 | 327.69 +/- 45.72 | 375.55 +/- 63.05 | 352.36 +/- 40.07 | 330.46 +/- 31.56 | 371.85 +/- 56.92 | 327.73 +/- 50.04 | 390.37 +/- 58.84 | 347.04 +/- 29.89 | 363.69 +/- 34.96 |
| reward_noise | 2764.91 +/- 743.29 | 2714.25 +/- 545.55 | 2029.49 +/- 920.73 | 2919.77 +/- 903.06 | 2596.87 +/- 241.51 | 2540.46 +/- 633.31 | 3060.47 +/- 431.24 | 2075.00 +/- 856.84 | 2535.63 +/- 754.60 | 2664.76 +/- 595.44 | 2944.00 +/- 556.68 |

## Axis overview

| Axis | Model | Nominal return | Mean gain over perturbed scenarios | Positive gain scenarios |
| --- | --- | --- | --- | --- |
| actuator_bias | No-op (1e9) | 2599.83 +/- 452.74 | +94.83 | 6/10 |
| actuator_bias | TV cap=2.85 | 2175.15 +/- 995.98 | -178.19 | 1/10 |
| actuator_bias | TV cap=2.95 | 2993.21 +/- 918.91 | -170.33 | 1/10 |
| actuator_bias | TV cap=3.00 | 2788.00 +/- 429.27 | -206.59 | 1/10 |
| actuator_bias | TV cap=3.05 | 2688.57 +/- 634.97 | -35.32 | 4/10 |
| actuator_bias | TV cap=3.10 | 3036.10 +/- 317.02 | -61.45 | 3/10 |
| actuator_bias | TV cap=3.20 | 2253.92 +/- 837.34 | -139.71 | 1/10 |
| actuator_bias | TV cap=3.50 | 2507.62 +/- 847.31 | -125.29 | 2/10 |
| actuator_bias | TV cap=3.70 | 2523.79 +/- 488.52 | -20.78 | 4/10 |
| actuator_bias | TV cap=4.00 | 2919.62 +/- 546.39 | -64.49 | 6/10 |
| actuator_gain | No-op (1e9) | 2657.55 +/- 532.70 | -13.27 | 4/10 |
| actuator_gain | TV cap=2.85 | 2275.51 +/- 1104.57 | -358.41 | 0/10 |
| actuator_gain | TV cap=2.95 | 3028.33 +/- 799.70 | -337.57 | 0/10 |
| actuator_gain | TV cap=3.00 | 2718.59 +/- 402.25 | -299.71 | 0/10 |
| actuator_gain | TV cap=3.05 | 2681.98 +/- 564.01 | -222.03 | 1/10 |
| actuator_gain | TV cap=3.10 | 3034.00 +/- 494.97 | -250.45 | 1/10 |
| actuator_gain | TV cap=3.20 | 2183.75 +/- 760.57 | -196.75 | 2/10 |
| actuator_gain | TV cap=3.50 | 2555.88 +/- 882.15 | -344.90 | 0/10 |
| actuator_gain | TV cap=3.70 | 2654.50 +/- 708.47 | -342.23 | 0/10 |
| actuator_gain | TV cap=4.00 | 2970.52 +/- 452.23 | -277.81 | 0/10 |
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
| mass | No-op (1e9) | 2933.61 +/- 571.91 | -266.70 | 0/10 |
| mass | TV cap=2.85 | 2193.47 +/- 1008.07 | -298.12 | 0/10 |
| mass | TV cap=2.95 | 2941.68 +/- 849.96 | -321.33 | 0/10 |
| mass | TV cap=3.00 | 2799.09 +/- 477.34 | -364.17 | 0/10 |
| mass | TV cap=3.05 | 2605.49 +/- 456.81 | -149.17 | 1/10 |
| mass | TV cap=3.10 | 2875.53 +/- 498.62 | -114.58 | 4/10 |
| mass | TV cap=3.20 | 2093.85 +/- 750.15 | -168.63 | 1/10 |
| mass | TV cap=3.50 | 2559.32 +/- 874.41 | -365.84 | 0/10 |
| mass | TV cap=3.70 | 2619.37 +/- 628.14 | -331.09 | 0/10 |
| mass | TV cap=4.00 | 2873.49 +/- 489.71 | -240.43 | 0/10 |
| obs_noise | No-op (1e9) | 336.83 +/- 44.48 | +18.19 | 10/10 |
| obs_noise | TV cap=2.85 | 327.69 +/- 45.72 | +10.32 | 6/10 |
| obs_noise | TV cap=2.95 | 375.55 +/- 63.05 | -11.36 | 2/10 |
| obs_noise | TV cap=3.00 | 352.36 +/- 40.07 | +27.11 | 7/10 |
| obs_noise | TV cap=3.05 | 330.46 +/- 31.56 | +37.00 | 9/10 |
| obs_noise | TV cap=3.10 | 371.85 +/- 56.92 | +14.09 | 5/10 |
| obs_noise | TV cap=3.20 | 327.73 +/- 50.04 | +10.72 | 7/10 |
| obs_noise | TV cap=3.50 | 390.37 +/- 58.84 | +12.82 | 7/10 |
| obs_noise | TV cap=3.70 | 347.04 +/- 29.89 | +44.28 | 9/10 |
| obs_noise | TV cap=4.00 | 363.69 +/- 34.96 | +18.00 | 5/10 |
| reward_noise | No-op (1e9) | 2714.25 +/- 545.55 | +23.63 | 5/10 |
| reward_noise | TV cap=2.85 | 2029.49 +/- 920.73 | +15.81 | 3/10 |
| reward_noise | TV cap=2.95 | 2919.77 +/- 903.06 | -114.60 | 2/10 |
| reward_noise | TV cap=3.00 | 2596.87 +/- 241.51 | -26.69 | 4/10 |
| reward_noise | TV cap=3.05 | 2540.46 +/- 633.31 | +75.21 | 7/10 |
| reward_noise | TV cap=3.10 | 3060.47 +/- 431.24 | -85.42 | 3/10 |
| reward_noise | TV cap=3.20 | 2075.00 +/- 856.84 | +61.79 | 6/10 |
| reward_noise | TV cap=3.50 | 2535.63 +/- 754.60 | -116.90 | 2/10 |
| reward_noise | TV cap=3.70 | 2664.76 +/- 595.44 | -147.61 | 2/10 |
| reward_noise | TV cap=4.00 | 2944.00 +/- 556.68 | -79.83 | 3/10 |

## Plot files

- `plots/return_curves_panel.png`
- `plots/gain_curves_panel.png`
- `plots/actuator_bias_return_curve.png`
- `plots/actuator_bias_gain_curve.png`
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
