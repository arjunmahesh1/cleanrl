# PPO_Walker_fmd_mass_0p1_2p0_20260430

Date: 2026-05-15

This folder packages the final pinned robustness evaluation for normalized PPO.

## Contents

- Raw metrics: `/Users/arjunmahesh/Library/CloudStorage/OneDrive-Personal/Duke/Research/RL/cleanrl/sweeps/results/PPO_Walker_final_presentable_20260430/raw_metrics/combo_mass`
- Aggregated outputs: `/Users/arjunmahesh/Library/CloudStorage/OneDrive-Personal/Duke/Research/RL/cleanrl/sweeps/results/PPO_Walker_fmd_mass_0p1_2p0_20260430/outputs`
- Plots: `/Users/arjunmahesh/Library/CloudStorage/OneDrive-Personal/Duke/Research/RL/cleanrl/sweeps/results/PPO_Walker_fmd_mass_0p1_2p0_20260430/plots`

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
| friction_mass | 2749.92 +/- 528.52 | 2138.25 +/- 1043.43 | 3032.42 +/- 822.07 | 2816.91 +/- 480.31 | 2718.19 +/- 470.37 | 2958.27 +/- 353.65 | 2244.00 +/- 834.25 | 2648.53 +/- 833.58 | 2678.89 +/- 579.57 | 3020.29 +/- 459.82 |
| friction_mass_damping | 2814.79 +/- 447.43 | 2146.28 +/- 940.78 | 2838.92 +/- 921.62 | 2519.23 +/- 489.43 | 2636.99 +/- 537.95 | 2970.37 +/- 431.54 | 2057.68 +/- 745.92 | 2518.58 +/- 883.75 | 2701.10 +/- 654.53 | 2946.18 +/- 575.87 |
| mass_damping | 2936.62 +/- 582.83 | 2197.82 +/- 1075.76 | 2820.84 +/- 819.72 | 2597.21 +/- 306.39 | 2714.55 +/- 553.71 | 3057.84 +/- 271.37 | 2300.54 +/- 856.24 | 2582.20 +/- 894.65 | 2475.00 +/- 635.27 | 2784.01 +/- 388.84 |

## Axis overview

| Axis | Model | Nominal return | Mean gain over perturbed scenarios | Positive gain scenarios |
| --- | --- | --- | --- | --- |
| friction_mass | TV cap=2.85 | 2138.25 +/- 1043.43 | +476.67 | 14/14 |
| friction_mass | TV cap=2.95 | 3032.42 +/- 822.07 | -239.57 | 2/14 |
| friction_mass | TV cap=3.00 | 2816.91 +/- 480.31 | -52.74 | 3/14 |
| friction_mass | TV cap=3.05 | 2718.19 +/- 470.37 | +97.25 | 11/14 |
| friction_mass | TV cap=3.10 | 2958.27 +/- 353.65 | -200.84 | 1/14 |
| friction_mass | TV cap=3.20 | 2244.00 +/- 834.25 | +306.34 | 13/14 |
| friction_mass | TV cap=3.50 | 2648.53 +/- 833.58 | -80.71 | 7/14 |
| friction_mass | TV cap=3.70 | 2678.89 +/- 579.57 | +105.29 | 10/14 |
| friction_mass | TV cap=4.00 | 3020.29 +/- 459.82 | -173.74 | 2/14 |
| friction_mass_damping | TV cap=2.85 | 2146.28 +/- 940.78 | +513.57 | 14/14 |
| friction_mass_damping | TV cap=2.95 | 2838.92 +/- 921.62 | -11.54 | 5/14 |
| friction_mass_damping | TV cap=3.00 | 2519.23 +/- 489.43 | +279.97 | 12/14 |
| friction_mass_damping | TV cap=3.05 | 2636.99 +/- 537.95 | +236.29 | 14/14 |
| friction_mass_damping | TV cap=3.10 | 2970.37 +/- 431.54 | -116.43 | 2/14 |
| friction_mass_damping | TV cap=3.20 | 2057.68 +/- 745.92 | +572.55 | 14/14 |
| friction_mass_damping | TV cap=3.50 | 2518.58 +/- 883.75 | +171.98 | 11/14 |
| friction_mass_damping | TV cap=3.70 | 2701.10 +/- 654.53 | +147.66 | 13/14 |
| friction_mass_damping | TV cap=4.00 | 2946.18 +/- 575.87 | +18.87 | 4/14 |
| mass_damping | TV cap=2.85 | 2197.82 +/- 1075.76 | +712.72 | 14/14 |
| mass_damping | TV cap=2.95 | 2820.84 +/- 819.72 | +300.16 | 14/14 |
| mass_damping | TV cap=3.00 | 2597.21 +/- 306.39 | +371.98 | 13/14 |
| mass_damping | TV cap=3.05 | 2714.55 +/- 553.71 | +429.99 | 14/14 |
| mass_damping | TV cap=3.10 | 3057.84 +/- 271.37 | +1.68 | 5/14 |
| mass_damping | TV cap=3.20 | 2300.54 +/- 856.24 | +551.62 | 14/14 |
| mass_damping | TV cap=3.50 | 2582.20 +/- 894.65 | +172.66 | 11/14 |
| mass_damping | TV cap=3.70 | 2475.00 +/- 635.27 | +495.90 | 14/14 |
| mass_damping | TV cap=4.00 | 2784.01 +/- 388.84 | +321.85 | 14/14 |

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
- `plots/with_variance/friction_mass_return_curve.png`
- `plots/with_variance/friction_mass_return_curve.pdf`
- `plots/with_variance/friction_mass_gain_curve.png`
- `plots/with_variance/friction_mass_gain_curve.pdf`
- `plots/without_variance/friction_mass_return_curve.png`
- `plots/without_variance/friction_mass_return_curve.pdf`
- `plots/without_variance/friction_mass_gain_curve.png`
- `plots/without_variance/friction_mass_gain_curve.pdf`
- `plots/with_variance/friction_mass_damping_return_curve.png`
- `plots/with_variance/friction_mass_damping_return_curve.pdf`
- `plots/with_variance/friction_mass_damping_gain_curve.png`
- `plots/with_variance/friction_mass_damping_gain_curve.pdf`
- `plots/without_variance/friction_mass_damping_return_curve.png`
- `plots/without_variance/friction_mass_damping_return_curve.pdf`
- `plots/without_variance/friction_mass_damping_gain_curve.png`
- `plots/without_variance/friction_mass_damping_gain_curve.pdf`
- `plots/with_variance/mass_damping_return_curve.png`
- `plots/with_variance/mass_damping_return_curve.pdf`
- `plots/with_variance/mass_damping_gain_curve.png`
- `plots/with_variance/mass_damping_gain_curve.pdf`
- `plots/without_variance/mass_damping_return_curve.png`
- `plots/without_variance/mass_damping_return_curve.pdf`
- `plots/without_variance/mass_damping_gain_curve.png`
- `plots/without_variance/mass_damping_gain_curve.pdf`

## Output CSV files

- `outputs/eval_metrics_final.csv`: latest merged per-seed eval rows.
- `outputs/summary_by_scenario.csv`: per-model, per-axis, per-factor aggregate return table.
- `outputs/drop_summary.csv`: nominal-minus-perturbed drop table.
- `outputs/gain_summary.csv`: paired robust-gain table against vanilla.
- `outputs/axis_overview.csv`: compact axis-level overview.
- `outputs/curve_points.csv`: same data used for return plots.
- `outputs/gain_curve_points.csv`: same data used for gain plots.
