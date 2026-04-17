# PPO_Adversarial_walker_targeted_xml_20260413

Date: 2026-04-15

This folder packages the final pinned robustness evaluation for normalized PPO.

## Contents

- Raw metrics: `/home/users/am1015/cleanrl/sweeps/results/PPO_Adversarial_walker_targeted_xml_20260413/raw_metrics`
- Aggregated outputs: `/home/users/am1015/cleanrl/sweeps/results/PPO_Adversarial_walker_targeted_xml_20260413/outputs`
- Plots: `/home/users/am1015/cleanrl/sweeps/results/PPO_Adversarial_walker_targeted_xml_20260413/plots`

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
| foot_left_actuator_gain | 2684.99 +/- 430.16 | 2731.68 +/- 664.10 | 2272.55 +/- 1027.47 | 2955.11 +/- 936.84 | 2479.47 +/- 321.22 | 2665.43 +/- 545.09 | 3035.02 +/- 470.51 | 2123.89 +/- 681.49 | 2525.30 +/- 1047.81 | 2713.29 +/- 674.49 | 2851.16 +/- 519.92 |
| foot_left_damping | 2914.44 +/- 557.10 | 2857.88 +/- 505.41 | 2239.74 +/- 1062.70 | 2787.62 +/- 833.76 | 2690.05 +/- 467.75 | 2786.95 +/- 655.70 | 3164.38 +/- 435.67 | 2204.26 +/- 873.99 | 2507.25 +/- 822.08 | 2514.14 +/- 621.62 | 3018.54 +/- 310.48 |
| foot_left_friction | 2875.47 +/- 604.14 | 2658.56 +/- 528.24 | 2008.25 +/- 888.13 | 2903.73 +/- 765.05 | 2870.56 +/- 430.27 | 2785.75 +/- 666.44 | 3007.97 +/- 493.68 | 2230.36 +/- 811.36 | 2523.48 +/- 836.76 | 2577.03 +/- 628.61 | 3000.57 +/- 520.00 |
| foot_left_mass | 3008.09 +/- 662.36 | 3100.70 +/- 629.41 | 2060.53 +/- 894.95 | 2792.93 +/- 819.11 | 2722.31 +/- 438.76 | 2486.28 +/- 483.97 | 3142.12 +/- 263.58 | 2033.04 +/- 858.11 | 2553.59 +/- 921.56 | 2505.87 +/- 572.32 | 2832.60 +/- 406.47 |
| leg_left_actuator_gain | 2965.73 +/- 501.87 | 3027.88 +/- 370.11 | 2086.62 +/- 903.64 | 3160.85 +/- 814.50 | 2498.52 +/- 316.94 | 2690.34 +/- 486.75 | 2865.95 +/- 457.71 | 2048.35 +/- 782.73 | 2490.15 +/- 774.60 | 2521.90 +/- 668.92 | 2923.57 +/- 545.01 |
| leg_left_damping | 2822.13 +/- 750.60 | 2905.41 +/- 588.21 | 2157.29 +/- 993.09 | 2725.88 +/- 671.17 | 2539.64 +/- 304.01 | 2759.95 +/- 623.81 | 3201.85 +/- 342.61 | 2173.15 +/- 896.09 | 2439.36 +/- 791.17 | 2645.39 +/- 631.96 | 3062.46 +/- 433.96 |
| leg_left_mass | 2954.64 +/- 342.68 | 2942.27 +/- 473.87 | 2116.50 +/- 923.33 | 2856.88 +/- 905.52 | 2561.93 +/- 278.60 | 2821.13 +/- 784.88 | 3187.87 +/- 442.19 | 2192.46 +/- 813.79 | 2496.39 +/- 957.88 | 2552.07 +/- 688.45 | 2977.37 +/- 406.41 |
| thigh_left_actuator_gain | 2978.76 +/- 615.86 | 2862.61 +/- 602.45 | 2179.72 +/- 1075.76 | 2740.21 +/- 733.42 | 2556.39 +/- 372.61 | 2436.93 +/- 507.77 | 3140.75 +/- 291.06 | 2123.50 +/- 841.16 | 2316.54 +/- 896.71 | 2453.26 +/- 645.13 | 2974.53 +/- 721.26 |
| thigh_left_damping | 2904.40 +/- 540.61 | 2826.23 +/- 631.09 | 2050.68 +/- 886.70 | 2901.18 +/- 895.88 | 2748.70 +/- 312.59 | 2800.04 +/- 612.98 | 2876.19 +/- 300.43 | 2265.75 +/- 897.44 | 2454.62 +/- 922.21 | 2548.61 +/- 621.53 | 2820.66 +/- 556.41 |
| thigh_left_mass | 2854.72 +/- 722.70 | 2992.29 +/- 547.30 | 2133.42 +/- 1038.04 | 2796.36 +/- 897.71 | 2829.02 +/- 460.96 | 2720.75 +/- 560.91 | 3152.97 +/- 429.81 | 2387.13 +/- 877.43 | 2292.23 +/- 666.23 | 2610.44 +/- 551.31 | 2789.17 +/- 570.17 |

## Axis overview

| Axis | Model | Nominal return | Mean gain over perturbed scenarios | Positive gain scenarios |
| --- | --- | --- | --- | --- |
| foot_left_actuator_gain | No-op (1e9) | 2731.68 +/- 664.10 | -38.43 | 2/10 |
| foot_left_actuator_gain | TV cap=2.85 | 2272.55 +/- 1027.47 | +280.19 | 6/10 |
| foot_left_actuator_gain | TV cap=2.95 | 2955.11 +/- 936.84 | -353.74 | 3/10 |
| foot_left_actuator_gain | TV cap=3.00 | 2479.47 +/- 321.22 | +402.55 | 5/10 |
| foot_left_actuator_gain | TV cap=3.05 | 2665.43 +/- 545.09 | +29.47 | 7/10 |
| foot_left_actuator_gain | TV cap=3.10 | 3035.02 +/- 470.51 | -42.02 | 5/10 |
| foot_left_actuator_gain | TV cap=3.20 | 2123.89 +/- 681.49 | +542.88 | 8/10 |
| foot_left_actuator_gain | TV cap=3.50 | 2525.30 +/- 1047.81 | +58.10 | 5/10 |
| foot_left_actuator_gain | TV cap=3.70 | 2713.29 +/- 674.49 | -143.92 | 5/10 |
| foot_left_actuator_gain | TV cap=4.00 | 2851.16 +/- 519.92 | +89.14 | 7/10 |
| foot_left_damping | No-op (1e9) | 2857.88 +/- 505.41 | +108.52 | 6/10 |
| foot_left_damping | TV cap=2.85 | 2239.74 +/- 1062.70 | -36.68 | 3/10 |
| foot_left_damping | TV cap=2.95 | 2787.62 +/- 833.76 | +164.03 | 10/10 |
| foot_left_damping | TV cap=3.00 | 2690.05 +/- 467.75 | +18.79 | 5/10 |
| foot_left_damping | TV cap=3.05 | 2786.95 +/- 655.70 | -28.20 | 4/10 |
| foot_left_damping | TV cap=3.10 | 3164.38 +/- 435.67 | -64.70 | 3/10 |
| foot_left_damping | TV cap=3.20 | 2204.26 +/- 873.99 | +103.15 | 6/10 |
| foot_left_damping | TV cap=3.50 | 2507.25 +/- 822.08 | +57.85 | 6/10 |
| foot_left_damping | TV cap=3.70 | 2514.14 +/- 621.62 | +183.75 | 9/10 |
| foot_left_damping | TV cap=4.00 | 3018.54 +/- 310.48 | -84.66 | 2/10 |
| foot_left_friction | No-op (1e9) | 2658.56 +/- 528.24 | +216.73 | 10/10 |
| foot_left_friction | TV cap=2.85 | 2008.25 +/- 888.13 | +507.40 | 7/10 |
| foot_left_friction | TV cap=2.95 | 2903.73 +/- 765.05 | +497.69 | 10/10 |
| foot_left_friction | TV cap=3.00 | 2870.56 +/- 430.27 | +219.50 | 5/10 |
| foot_left_friction | TV cap=3.05 | 2785.75 +/- 666.44 | +178.12 | 9/10 |
| foot_left_friction | TV cap=3.10 | 3007.97 +/- 493.68 | +413.75 | 8/10 |
| foot_left_friction | TV cap=3.20 | 2230.36 +/- 811.36 | +505.25 | 10/10 |
| foot_left_friction | TV cap=3.50 | 2523.48 +/- 836.76 | +226.72 | 7/10 |
| foot_left_friction | TV cap=3.70 | 2577.03 +/- 628.61 | +185.21 | 8/10 |
| foot_left_friction | TV cap=4.00 | 3000.57 +/- 520.00 | +79.71 | 4/10 |
| foot_left_mass | No-op (1e9) | 3100.70 +/- 629.41 | -59.29 | 4/10 |
| foot_left_mass | TV cap=2.85 | 2060.53 +/- 894.95 | +650.44 | 10/10 |
| foot_left_mass | TV cap=2.95 | 2792.93 +/- 819.11 | +464.87 | 9/10 |
| foot_left_mass | TV cap=3.00 | 2722.31 +/- 438.76 | +437.45 | 10/10 |
| foot_left_mass | TV cap=3.05 | 2486.28 +/- 483.97 | +693.32 | 10/10 |
| foot_left_mass | TV cap=3.10 | 3142.12 +/- 263.58 | +285.59 | 6/10 |
| foot_left_mass | TV cap=3.20 | 2033.04 +/- 858.11 | +671.02 | 10/10 |
| foot_left_mass | TV cap=3.50 | 2553.59 +/- 921.56 | +275.66 | 9/10 |
| foot_left_mass | TV cap=3.70 | 2505.87 +/- 572.32 | +481.29 | 10/10 |
| foot_left_mass | TV cap=4.00 | 2832.60 +/- 406.47 | +423.23 | 10/10 |
| leg_left_actuator_gain | No-op (1e9) | 3027.88 +/- 370.11 | -46.96 | 4/10 |
| leg_left_actuator_gain | TV cap=2.85 | 2086.62 +/- 903.64 | +354.36 | 9/10 |
| leg_left_actuator_gain | TV cap=2.95 | 3160.85 +/- 814.50 | -129.65 | 2/10 |
| leg_left_actuator_gain | TV cap=3.00 | 2498.52 +/- 316.94 | +436.18 | 10/10 |
| leg_left_actuator_gain | TV cap=3.05 | 2690.34 +/- 486.75 | +133.56 | 8/10 |
| leg_left_actuator_gain | TV cap=3.10 | 2865.95 +/- 457.71 | +377.64 | 10/10 |
| leg_left_actuator_gain | TV cap=3.20 | 2048.35 +/- 782.73 | +383.99 | 9/10 |
| leg_left_actuator_gain | TV cap=3.50 | 2490.15 +/- 774.60 | +254.97 | 9/10 |
| leg_left_actuator_gain | TV cap=3.70 | 2521.90 +/- 668.92 | +379.32 | 10/10 |
| leg_left_actuator_gain | TV cap=4.00 | 2923.57 +/- 545.01 | +306.55 | 10/10 |
| leg_left_damping | No-op (1e9) | 2905.41 +/- 588.21 | -99.56 | 1/10 |
| leg_left_damping | TV cap=2.85 | 2157.29 +/- 993.09 | -66.60 | 4/10 |
| leg_left_damping | TV cap=2.95 | 2725.88 +/- 671.17 | +212.57 | 9/10 |
| leg_left_damping | TV cap=3.00 | 2539.64 +/- 304.01 | +116.20 | 8/10 |
| leg_left_damping | TV cap=3.05 | 2759.95 +/- 623.81 | -100.52 | 2/10 |
| leg_left_damping | TV cap=3.10 | 3201.85 +/- 342.61 | -147.99 | 1/10 |
| leg_left_damping | TV cap=3.20 | 2173.15 +/- 896.09 | +66.57 | 5/10 |
| leg_left_damping | TV cap=3.50 | 2439.36 +/- 791.17 | +43.53 | 5/10 |
| leg_left_damping | TV cap=3.70 | 2645.39 +/- 631.96 | -44.22 | 4/10 |
| leg_left_damping | TV cap=4.00 | 3062.46 +/- 433.96 | -122.71 | 2/10 |
| leg_left_mass | No-op (1e9) | 2942.27 +/- 473.87 | +15.20 | 6/10 |
| leg_left_mass | TV cap=2.85 | 2116.50 +/- 923.33 | +384.21 | 10/10 |
| leg_left_mass | TV cap=2.95 | 2856.88 +/- 905.52 | +237.90 | 9/10 |
| leg_left_mass | TV cap=3.00 | 2561.93 +/- 278.60 | +404.61 | 9/10 |
| leg_left_mass | TV cap=3.05 | 2821.13 +/- 784.88 | +92.15 | 8/10 |
| leg_left_mass | TV cap=3.10 | 3187.87 +/- 442.19 | +79.61 | 7/10 |
| leg_left_mass | TV cap=3.20 | 2192.46 +/- 813.79 | +360.69 | 9/10 |
| leg_left_mass | TV cap=3.50 | 2496.39 +/- 957.88 | +312.31 | 9/10 |
| leg_left_mass | TV cap=3.70 | 2552.07 +/- 688.45 | +314.73 | 8/10 |
| leg_left_mass | TV cap=4.00 | 2977.37 +/- 406.41 | +220.20 | 10/10 |
| thigh_left_actuator_gain | No-op (1e9) | 2862.61 +/- 602.45 | +125.87 | 8/10 |
| thigh_left_actuator_gain | TV cap=2.85 | 2179.72 +/- 1075.76 | +11.74 | 6/10 |
| thigh_left_actuator_gain | TV cap=2.95 | 2740.21 +/- 733.42 | -198.69 | 4/10 |
| thigh_left_actuator_gain | TV cap=3.00 | 2556.39 +/- 372.61 | +255.87 | 10/10 |
| thigh_left_actuator_gain | TV cap=3.05 | 2436.93 +/- 507.77 | +337.32 | 10/10 |
| thigh_left_actuator_gain | TV cap=3.10 | 3140.75 +/- 291.06 | -99.54 | 4/10 |
| thigh_left_actuator_gain | TV cap=3.20 | 2123.50 +/- 841.16 | +317.25 | 10/10 |
| thigh_left_actuator_gain | TV cap=3.50 | 2316.54 +/- 896.71 | +354.07 | 10/10 |
| thigh_left_actuator_gain | TV cap=3.70 | 2453.26 +/- 645.13 | +125.98 | 9/10 |
| thigh_left_actuator_gain | TV cap=4.00 | 2974.53 +/- 721.26 | +76.20 | 6/10 |
| thigh_left_damping | No-op (1e9) | 2826.23 +/- 631.09 | +97.14 | 6/10 |
| thigh_left_damping | TV cap=2.85 | 2050.68 +/- 886.70 | +72.49 | 7/10 |
| thigh_left_damping | TV cap=2.95 | 2901.18 +/- 895.88 | +48.78 | 6/10 |
| thigh_left_damping | TV cap=3.00 | 2748.70 +/- 312.59 | -99.72 | 2/10 |
| thigh_left_damping | TV cap=3.05 | 2800.04 +/- 612.98 | -40.41 | 2/10 |
| thigh_left_damping | TV cap=3.10 | 2876.19 +/- 300.43 | +136.86 | 9/10 |
| thigh_left_damping | TV cap=3.20 | 2265.75 +/- 897.44 | -57.11 | 3/10 |
| thigh_left_damping | TV cap=3.50 | 2454.62 +/- 922.21 | +14.35 | 5/10 |
| thigh_left_damping | TV cap=3.70 | 2548.61 +/- 621.53 | +23.19 | 5/10 |
| thigh_left_damping | TV cap=4.00 | 2820.66 +/- 556.41 | +98.34 | 6/10 |
| thigh_left_mass | No-op (1e9) | 2992.29 +/- 547.30 | -180.09 | 1/10 |
| thigh_left_mass | TV cap=2.85 | 2133.42 +/- 1038.04 | +248.17 | 8/10 |
| thigh_left_mass | TV cap=2.95 | 2796.36 +/- 897.71 | +39.23 | 5/10 |
| thigh_left_mass | TV cap=3.00 | 2829.02 +/- 460.96 | +56.64 | 5/10 |
| thigh_left_mass | TV cap=3.05 | 2720.75 +/- 560.91 | +47.69 | 5/10 |
| thigh_left_mass | TV cap=3.10 | 3152.97 +/- 429.81 | -89.98 | 3/10 |
| thigh_left_mass | TV cap=3.20 | 2387.13 +/- 877.43 | -5.56 | 4/10 |
| thigh_left_mass | TV cap=3.50 | 2292.23 +/- 666.23 | +461.21 | 10/10 |
| thigh_left_mass | TV cap=3.70 | 2610.44 +/- 551.31 | +148.60 | 6/10 |
| thigh_left_mass | TV cap=4.00 | 2789.17 +/- 570.17 | +220.70 | 9/10 |

## Plot files

- `plots/return_curves_panel.png`
- `plots/gain_curves_panel.png`
- `plots/foot_left_actuator_gain_return_curve.png`
- `plots/foot_left_actuator_gain_gain_curve.png`
- `plots/foot_left_damping_return_curve.png`
- `plots/foot_left_damping_gain_curve.png`
- `plots/foot_left_friction_return_curve.png`
- `plots/foot_left_friction_gain_curve.png`
- `plots/foot_left_mass_return_curve.png`
- `plots/foot_left_mass_gain_curve.png`
- `plots/leg_left_actuator_gain_return_curve.png`
- `plots/leg_left_actuator_gain_gain_curve.png`
- `plots/leg_left_damping_return_curve.png`
- `plots/leg_left_damping_gain_curve.png`
- `plots/leg_left_mass_return_curve.png`
- `plots/leg_left_mass_gain_curve.png`
- `plots/thigh_left_actuator_gain_return_curve.png`
- `plots/thigh_left_actuator_gain_gain_curve.png`
- `plots/thigh_left_damping_return_curve.png`
- `plots/thigh_left_damping_gain_curve.png`
- `plots/thigh_left_mass_return_curve.png`
- `plots/thigh_left_mass_gain_curve.png`

## Output CSV files

- `outputs/eval_metrics_final.csv`: latest merged per-seed eval rows.
- `outputs/summary_by_scenario.csv`: per-model, per-axis, per-factor aggregate return table.
- `outputs/drop_summary.csv`: nominal-minus-perturbed drop table.
- `outputs/gain_summary.csv`: paired robust-gain table against vanilla.
- `outputs/axis_overview.csv`: compact axis-level overview.
- `outputs/curve_points.csv`: same data used for return plots.
- `outputs/gain_curve_points.csv`: same data used for gain plots.
