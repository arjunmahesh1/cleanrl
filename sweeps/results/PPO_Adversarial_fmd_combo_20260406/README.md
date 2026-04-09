# PPO_Adversarial_fmd_combo_20260406

Date: 2026-04-08

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
| friction_damping | 2808.40 +/- 695.57 | 2767.06 +/- 430.61 | 2149.05 +/- 1044.81 | 2912.85 +/- 828.27 | 2778.04 +/- 370.68 | 2760.32 +/- 750.42 | 2799.06 +/- 245.44 | 2200.36 +/- 866.77 | 2435.86 +/- 983.55 | 2458.55 +/- 677.15 | 2803.85 +/- 513.49 |
| friction_mass | 2915.79 +/- 530.83 | 2856.61 +/- 707.20 | 2212.58 +/- 1023.54 | 2798.67 +/- 788.89 | 2642.19 +/- 580.81 | 2567.57 +/- 467.58 | 3008.27 +/- 456.76 | 2344.97 +/- 976.71 | 2508.36 +/- 855.16 | 2394.36 +/- 693.47 | 3007.57 +/- 458.35 |
| friction_mass_damping | 2941.68 +/- 572.05 | 2805.42 +/- 468.11 | 2167.30 +/- 976.27 | 2820.39 +/- 771.43 | 2705.09 +/- 474.75 | 2756.59 +/- 594.96 | 3065.35 +/- 451.45 | 2231.94 +/- 774.80 | 2433.29 +/- 744.43 | 2619.13 +/- 742.56 | 2736.38 +/- 642.47 |
| mass_damping | 3005.52 +/- 589.80 | 2995.40 +/- 602.88 | 2197.35 +/- 1141.46 | 3084.08 +/- 1022.67 | 2568.15 +/- 313.24 | 2943.40 +/- 643.95 | 3001.72 +/- 372.52 | 2189.67 +/- 864.47 | 2438.91 +/- 898.10 | 2545.83 +/- 452.40 | 2886.78 +/- 573.33 |

## Axis overview

| Axis | Model | Nominal return | Mean gain over perturbed scenarios | Positive gain scenarios |
| --- | --- | --- | --- | --- |
| friction_damping | No-op (1e9) | 2767.06 +/- 430.61 | +34.63 | 5/6 |
| friction_damping | TV cap=2.85 | 2149.05 +/- 1044.81 | +130.74 | 3/6 |
| friction_damping | TV cap=2.95 | 2912.85 +/- 828.27 | +361.44 | 5/6 |
| friction_damping | TV cap=3.00 | 2778.04 +/- 370.68 | -26.69 | 3/6 |
| friction_damping | TV cap=3.05 | 2760.32 +/- 750.42 | -30.65 | 3/6 |
| friction_damping | TV cap=3.10 | 2799.06 +/- 245.44 | +466.45 | 5/6 |
| friction_damping | TV cap=3.20 | 2200.36 +/- 866.77 | +164.06 | 5/6 |
| friction_damping | TV cap=3.50 | 2435.86 +/- 983.55 | +56.93 | 3/6 |
| friction_damping | TV cap=3.70 | 2458.55 +/- 677.15 | +202.06 | 5/6 |
| friction_damping | TV cap=4.00 | 2803.85 +/- 513.49 | +212.27 | 4/6 |
| friction_mass | No-op (1e9) | 2856.61 +/- 707.20 | +15.95 | 4/6 |
| friction_mass | TV cap=2.85 | 2212.58 +/- 1023.54 | +191.77 | 3/6 |
| friction_mass | TV cap=2.95 | 2798.67 +/- 788.89 | +533.92 | 5/6 |
| friction_mass | TV cap=3.00 | 2642.19 +/- 580.81 | +258.92 | 6/6 |
| friction_mass | TV cap=3.05 | 2567.57 +/- 467.58 | +319.34 | 6/6 |
| friction_mass | TV cap=3.10 | 3008.27 +/- 456.76 | +349.79 | 5/6 |
| friction_mass | TV cap=3.20 | 2344.97 +/- 976.71 | +166.28 | 5/6 |
| friction_mass | TV cap=3.50 | 2508.36 +/- 855.16 | +80.37 | 3/6 |
| friction_mass | TV cap=3.70 | 2394.36 +/- 693.47 | +347.06 | 6/6 |
| friction_mass | TV cap=4.00 | 3007.57 +/- 458.35 | +56.70 | 3/6 |
| friction_mass_damping | No-op (1e9) | 2805.42 +/- 468.11 | +177.89 | 6/6 |
| friction_mass_damping | TV cap=2.85 | 2167.30 +/- 976.27 | +158.78 | 4/6 |
| friction_mass_damping | TV cap=2.95 | 2820.39 +/- 771.43 | +388.60 | 6/6 |
| friction_mass_damping | TV cap=3.00 | 2705.09 +/- 474.75 | +55.89 | 3/6 |
| friction_mass_damping | TV cap=3.05 | 2756.59 +/- 594.96 | +81.69 | 4/6 |
| friction_mass_damping | TV cap=3.10 | 3065.35 +/- 451.45 | +261.51 | 5/6 |
| friction_mass_damping | TV cap=3.20 | 2231.94 +/- 774.80 | +238.44 | 5/6 |
| friction_mass_damping | TV cap=3.50 | 2433.29 +/- 744.43 | +111.63 | 4/6 |
| friction_mass_damping | TV cap=3.70 | 2619.13 +/- 742.56 | +111.33 | 5/6 |
| friction_mass_damping | TV cap=4.00 | 2736.38 +/- 642.47 | +273.98 | 5/6 |
| mass_damping | No-op (1e9) | 2995.40 +/- 602.88 | +43.43 | 4/6 |
| mass_damping | TV cap=2.85 | 2197.35 +/- 1141.46 | +167.81 | 5/6 |
| mass_damping | TV cap=2.95 | 3084.08 +/- 1022.67 | -23.91 | 3/6 |
| mass_damping | TV cap=3.00 | 2568.15 +/- 313.24 | +286.07 | 6/6 |
| mass_damping | TV cap=3.05 | 2943.40 +/- 643.95 | -23.06 | 4/6 |
| mass_damping | TV cap=3.10 | 3001.72 +/- 372.52 | +321.34 | 6/6 |
| mass_damping | TV cap=3.20 | 2189.67 +/- 864.47 | +194.92 | 6/6 |
| mass_damping | TV cap=3.50 | 2438.91 +/- 898.10 | +226.70 | 6/6 |
| mass_damping | TV cap=3.70 | 2545.83 +/- 452.40 | +385.78 | 6/6 |
| mass_damping | TV cap=4.00 | 2886.78 +/- 573.33 | +204.55 | 6/6 |

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
