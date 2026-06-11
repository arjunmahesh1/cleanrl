# HalfCheetah PPO Full 30-Seed No-Noop: combos

Date: 2026-06-10

This folder packages the final pinned robustness evaluation for normalized PPO.

## Contents

- Raw metrics: `/home/users/am1015/cleanrl/sweeps/results/PPO_full30_allcaps_no_noop_20260524/HalfCheetah/combos/raw_metrics`
- Aggregated outputs: `/home/users/am1015/cleanrl/sweeps/results/PPO_full30_allcaps_no_noop_20260524/HalfCheetah/combos/outputs`
- Plots: `/home/users/am1015/cleanrl/sweeps/results/PPO_full30_allcaps_no_noop_20260524/HalfCheetah/combos/plots`

## Evaluation protocol

- Models are compared on the same perturbation grid for the configured axes.
- Nominal reference within each axis is the `factor=1.0` point.
- Curves show mean return across seeds with `95% CI` shading.
- Robust gain is defined as `vanilla_drop - model_drop`; positive is better.
- Plot files are exported in both `PNG` and vector `PDF` format.

## Model labels

- `vanilla` -> Vanilla
- `a2p20` -> a2p20
- `a2p40` -> a2p40
- `a2p55` -> a2p55
- `a2p65` -> a2p65
- `a2p70` -> a2p70
- `a2p75` -> a2p75
- `a2p80` -> a2p80
- `a3p00` -> a3p00
- `a3p05` -> TV cap=3.05
- `a3p10` -> a3p10
- `a3p20` -> a3p20
- `a3p40` -> a3p40
- `a3p70` -> a3p70

## Nominal returns by axis

| Axis | Vanilla | a2p20 | a2p40 | a2p55 | a2p65 | a2p70 | a2p75 | a2p80 | a3p00 | TV cap=3.05 | a3p10 | a3p20 | a3p40 | a3p70 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| friction_damping | 2644.08 +/- 481.35 | 2055.37 +/- 345.30 | 2221.58 +/- 387.95 | 2795.67 +/- 524.88 | 2079.96 +/- 339.43 | 2416.59 +/- 436.17 | 2550.82 +/- 499.29 | 2977.88 +/- 525.23 | 2494.97 +/- 457.26 | 2562.31 +/- 487.98 | 2832.98 +/- 543.70 | 2367.29 +/- 466.77 | 2770.06 +/- 529.61 | 2561.30 +/- 482.02 |
| friction_mass | 2644.08 +/- 481.35 | 2055.37 +/- 345.30 | 2221.58 +/- 387.95 | 2795.67 +/- 524.88 | 2079.96 +/- 339.43 | 2416.59 +/- 436.17 | 2550.82 +/- 499.29 | 2977.88 +/- 525.23 | 2494.97 +/- 457.26 | 2562.31 +/- 487.98 | 2832.98 +/- 543.70 | 2367.29 +/- 466.77 | 2765.16 +/- 525.04 | 2561.30 +/- 482.02 |
| friction_mass_damping | 2649.90 +/- 477.88 | 2061.01 +/- 346.00 | 2221.54 +/- 396.19 | 2793.03 +/- 525.29 | 2079.96 +/- 339.43 | 2416.59 +/- 436.17 | 2517.41 +/- 476.87 | 2998.65 +/- 524.52 | 2494.97 +/- 457.26 | 2562.31 +/- 487.98 | 2832.98 +/- 543.70 | 2363.25 +/- 467.49 | 2765.16 +/- 525.04 | 2561.30 +/- 482.02 |
| mass_damping | 2644.08 +/- 481.35 | 2055.37 +/- 345.30 | 2221.58 +/- 387.95 | 2795.67 +/- 524.88 | 2079.96 +/- 339.43 | 2416.59 +/- 436.17 | 2550.82 +/- 499.29 | 2977.88 +/- 525.23 | 2494.97 +/- 457.26 | 2562.31 +/- 487.98 | 2837.85 +/- 548.01 | 2354.02 +/- 456.27 | 2765.16 +/- 525.04 | 2544.29 +/- 478.37 |

## Axis overview

| Axis | Model | Nominal return | Mean gain over perturbed scenarios | Positive gain scenarios |
| --- | --- | --- | --- | --- |
| friction_damping | Vanilla | 2644.08 +/- 481.35 | +0.00 | 0/0 |
| friction_damping | Vanilla | 2644.08 +/- 481.35 | +0.00 | 0/15 |
| friction_damping | a2p20 | 2055.37 +/- 345.30 | +915.07 | 15/15 |
| friction_damping | a2p40 | 2221.58 +/- 387.95 | +612.29 | 15/15 |
| friction_damping | a2p55 | 2795.67 +/- 524.88 | +246.32 | 8/15 |
| friction_damping | a2p65 | 2079.96 +/- 339.43 | +753.86 | 15/15 |
| friction_damping | a2p70 | 2416.59 +/- 436.17 | +429.46 | 14/15 |
| friction_damping | a2p75 | 2550.82 +/- 499.29 | +551.09 | 15/15 |
| friction_damping | a2p80 | 2977.88 +/- 525.23 | +188.96 | 8/15 |
| friction_damping | a3p00 | 2494.97 +/- 457.26 | +446.72 | 15/15 |
| friction_damping | TV cap=3.05 | 2562.31 +/- 487.98 | +468.08 | 14/15 |
| friction_damping | a3p10 | 2832.98 +/- 543.70 | +69.72 | 6/15 |
| friction_damping | a3p20 | 2367.29 +/- 466.77 | +823.74 | 13/15 |
| friction_damping | a3p40 | 2770.06 +/- 529.61 | +254.05 | 10/15 |
| friction_damping | a3p70 | 2561.30 +/- 482.02 | +419.69 | 12/15 |
| friction_mass | Vanilla | 2644.08 +/- 481.35 | +0.00 | 0/0 |
| friction_mass | Vanilla | 2644.08 +/- 481.35 | +0.00 | 0/14 |
| friction_mass | a2p20 | 2055.37 +/- 345.30 | +398.28 | 14/14 |
| friction_mass | a2p40 | 2221.58 +/- 387.95 | +251.25 | 14/14 |
| friction_mass | a2p55 | 2795.67 +/- 524.88 | -92.01 | 5/14 |
| friction_mass | a2p65 | 2079.96 +/- 339.43 | +430.67 | 14/14 |
| friction_mass | a2p70 | 2416.59 +/- 436.17 | +167.83 | 12/14 |
| friction_mass | a2p75 | 2550.82 +/- 499.29 | +34.52 | 10/14 |
| friction_mass | a2p80 | 2977.88 +/- 525.23 | -267.81 | 2/14 |
| friction_mass | a3p00 | 2494.97 +/- 457.26 | +133.89 | 13/14 |
| friction_mass | TV cap=3.05 | 2562.31 +/- 487.98 | +62.48 | 10/14 |
| friction_mass | a3p10 | 2832.98 +/- 543.70 | -91.40 | 4/14 |
| friction_mass | a3p20 | 2367.29 +/- 466.77 | +159.94 | 10/14 |
| friction_mass | a3p40 | 2765.16 +/- 525.04 | -55.84 | 6/14 |
| friction_mass | a3p70 | 2561.30 +/- 482.02 | +38.57 | 11/14 |
| friction_mass_damping | Vanilla | 2649.90 +/- 477.88 | +0.00 | 0/0 |
| friction_mass_damping | Vanilla | 2649.90 +/- 477.88 | +0.00 | 0/14 |
| friction_mass_damping | a2p20 | 2061.01 +/- 346.00 | +541.88 | 14/14 |
| friction_mass_damping | a2p40 | 2221.54 +/- 396.19 | +362.63 | 13/14 |
| friction_mass_damping | a2p55 | 2793.03 +/- 525.29 | -140.60 | 0/14 |
| friction_mass_damping | a2p65 | 2079.96 +/- 339.43 | +663.11 | 14/14 |
| friction_mass_damping | a2p70 | 2416.59 +/- 436.17 | +180.48 | 13/14 |
| friction_mass_damping | a2p75 | 2517.41 +/- 476.87 | +133.90 | 13/14 |
| friction_mass_damping | a2p80 | 2998.65 +/- 524.52 | -356.60 | 1/14 |
| friction_mass_damping | a3p00 | 2494.97 +/- 457.26 | +203.27 | 14/14 |
| friction_mass_damping | TV cap=3.05 | 2562.31 +/- 487.98 | +56.39 | 12/14 |
| friction_mass_damping | a3p10 | 2832.98 +/- 543.70 | -134.79 | 1/14 |
| friction_mass_damping | a3p20 | 2363.25 +/- 467.49 | +272.50 | 13/14 |
| friction_mass_damping | a3p40 | 2765.16 +/- 525.04 | -14.42 | 8/14 |
| friction_mass_damping | a3p70 | 2561.30 +/- 482.02 | +228.95 | 12/14 |
| mass_damping | Vanilla | 2644.08 +/- 481.35 | +0.00 | 0/0 |
| mass_damping | Vanilla | 2644.08 +/- 481.35 | +0.00 | 0/14 |
| mass_damping | a2p20 | 2055.37 +/- 345.30 | +640.00 | 14/14 |
| mass_damping | a2p40 | 2221.58 +/- 387.95 | +395.13 | 14/14 |
| mass_damping | a2p55 | 2795.67 +/- 524.88 | -161.80 | 0/14 |
| mass_damping | a2p65 | 2079.96 +/- 339.43 | +599.91 | 14/14 |
| mass_damping | a2p70 | 2416.59 +/- 436.17 | +211.10 | 14/14 |
| mass_damping | a2p75 | 2550.82 +/- 499.29 | +191.38 | 13/14 |
| mass_damping | a2p80 | 2977.88 +/- 525.23 | -322.92 | 1/14 |
| mass_damping | a3p00 | 2494.97 +/- 457.26 | +187.40 | 13/14 |
| mass_damping | TV cap=3.05 | 2562.31 +/- 487.98 | +84.82 | 12/14 |
| mass_damping | a3p10 | 2837.85 +/- 548.01 | -86.56 | 1/14 |
| mass_damping | a3p20 | 2354.02 +/- 456.27 | +279.30 | 14/14 |
| mass_damping | a3p40 | 2765.16 +/- 525.04 | -8.38 | 6/14 |
| mass_damping | a3p70 | 2544.29 +/- 478.37 | +179.33 | 12/14 |

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
- `plots/with_variance/friction_damping_return_curve.png`
- `plots/with_variance/friction_damping_return_curve.pdf`
- `plots/with_variance/friction_damping_gain_curve.png`
- `plots/with_variance/friction_damping_gain_curve.pdf`
- `plots/without_variance/friction_damping_return_curve.png`
- `plots/without_variance/friction_damping_return_curve.pdf`
- `plots/without_variance/friction_damping_gain_curve.png`
- `plots/without_variance/friction_damping_gain_curve.pdf`
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
