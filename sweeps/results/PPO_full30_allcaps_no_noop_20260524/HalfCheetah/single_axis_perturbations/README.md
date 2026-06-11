# HalfCheetah PPO Full 30-Seed No-Noop: single_axis_perturbations

Date: 2026-06-10

This folder packages the final pinned robustness evaluation for normalized PPO.

## Contents

- Raw metrics: `/home/users/am1015/cleanrl/sweeps/results/PPO_full30_allcaps_no_noop_20260524/HalfCheetah/single_axis_perturbations/raw_metrics`
- Aggregated outputs: `/home/users/am1015/cleanrl/sweeps/results/PPO_full30_allcaps_no_noop_20260524/HalfCheetah/single_axis_perturbations/outputs`
- Plots: `/home/users/am1015/cleanrl/sweeps/results/PPO_full30_allcaps_no_noop_20260524/HalfCheetah/single_axis_perturbations/plots`

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
| damping | 2644.08 +/- 481.35 | 2055.37 +/- 345.30 | 2221.58 +/- 387.95 | 2795.67 +/- 524.88 | 2079.96 +/- 339.43 | 2416.59 +/- 436.17 | 2550.82 +/- 499.29 | 2977.88 +/- 525.23 | 2494.97 +/- 457.26 | 2562.31 +/- 487.98 | 2832.98 +/- 543.70 | 2367.29 +/- 466.77 | 2765.16 +/- 525.04 | 2561.30 +/- 482.02 |
| friction | 2644.08 +/- 481.35 | 2055.37 +/- 345.30 | 2221.58 +/- 387.95 | 2795.67 +/- 524.88 | 2089.20 +/- 351.65 | 2401.64 +/- 424.12 | 2550.82 +/- 499.29 | 2977.88 +/- 525.23 | 2494.97 +/- 457.26 | 2562.31 +/- 487.98 | 2832.98 +/- 543.70 | 2367.29 +/- 466.77 | 2765.16 +/- 525.04 | 2561.30 +/- 482.02 |
| gear | 2644.08 +/- 481.35 | 2055.37 +/- 345.30 | 2221.58 +/- 387.95 | 2795.67 +/- 524.88 | 2079.96 +/- 339.43 | 2416.59 +/- 436.17 | 2550.82 +/- 499.29 | 2977.88 +/- 525.23 | 2494.97 +/- 457.26 | 2562.31 +/- 487.98 | 2832.98 +/- 543.70 | 2367.29 +/- 466.77 | 2770.06 +/- 529.61 | 2561.30 +/- 482.02 |
| gravity | 2058.62 +/- 369.40 | 1302.25 +/- 265.64 | 1613.61 +/- 349.58 | 2067.04 +/- 438.82 | 1577.71 +/- 263.05 | 1819.21 +/- 327.09 | 1979.84 +/- 448.18 | 2084.02 +/- 421.84 | 1910.12 +/- 395.06 | 2017.68 +/- 456.25 | 2078.76 +/- 463.06 | 1831.17 +/- 335.67 | 2157.39 +/- 448.95 | 1915.41 +/- 344.42 |
| mass | 2644.08 +/- 481.35 | 2059.20 +/- 339.38 | 2221.58 +/- 387.95 | 2795.67 +/- 524.88 | 2079.96 +/- 339.43 | 2416.59 +/- 436.17 | 2550.82 +/- 499.29 | 3000.79 +/- 523.58 | 2494.97 +/- 457.26 | 2562.31 +/- 487.98 | 2832.98 +/- 543.70 | 2367.29 +/- 466.77 | 2765.16 +/- 525.04 | 2561.30 +/- 482.02 |

## Axis overview

| Axis | Model | Nominal return | Mean gain over perturbed scenarios | Positive gain scenarios |
| --- | --- | --- | --- | --- |
| damping | Vanilla | 2644.08 +/- 481.35 | +0.00 | 0/0 |
| damping | Vanilla | 2644.08 +/- 481.35 | +0.00 | 0/15 |
| damping | a2p20 | 2055.37 +/- 345.30 | +510.59 | 15/15 |
| damping | a2p40 | 2221.58 +/- 387.95 | +343.11 | 15/15 |
| damping | a2p55 | 2795.67 +/- 524.88 | -122.50 | 2/15 |
| damping | a2p65 | 2079.96 +/- 339.43 | +516.00 | 15/15 |
| damping | a2p70 | 2416.59 +/- 436.17 | +218.31 | 15/15 |
| damping | a2p75 | 2550.82 +/- 499.29 | +158.29 | 15/15 |
| damping | a2p80 | 2977.88 +/- 525.23 | -222.63 | 3/15 |
| damping | a3p00 | 2494.97 +/- 457.26 | +161.95 | 15/15 |
| damping | TV cap=3.05 | 2562.31 +/- 487.98 | +175.32 | 14/15 |
| damping | a3p10 | 2832.98 +/- 543.70 | -147.20 | 1/15 |
| damping | a3p20 | 2367.29 +/- 466.77 | +277.16 | 13/15 |
| damping | a3p40 | 2765.16 +/- 525.04 | -7.41 | 9/15 |
| damping | a3p70 | 2561.30 +/- 482.02 | +109.98 | 12/15 |
| friction | Vanilla | 2644.08 +/- 481.35 | +0.00 | 0/0 |
| friction | Vanilla | 2644.08 +/- 481.35 | +0.00 | 0/15 |
| friction | a2p20 | 2055.37 +/- 345.30 | +255.20 | 11/15 |
| friction | a2p40 | 2221.58 +/- 387.95 | +64.11 | 9/15 |
| friction | a2p55 | 2795.67 +/- 524.88 | -44.43 | 7/15 |
| friction | a2p65 | 2089.20 +/- 351.65 | +85.25 | 9/15 |
| friction | a2p70 | 2401.64 +/- 424.12 | +87.67 | 8/15 |
| friction | a2p75 | 2550.82 +/- 499.29 | +75.36 | 10/15 |
| friction | a2p80 | 2977.88 +/- 525.23 | -0.49 | 6/15 |
| friction | a3p00 | 2494.97 +/- 457.26 | +196.84 | 10/15 |
| friction | TV cap=3.05 | 2562.31 +/- 487.98 | -35.98 | 6/15 |
| friction | a3p10 | 2832.98 +/- 543.70 | -32.52 | 7/15 |
| friction | a3p20 | 2367.29 +/- 466.77 | +309.39 | 8/15 |
| friction | a3p40 | 2765.16 +/- 525.04 | -26.29 | 6/15 |
| friction | a3p70 | 2561.30 +/- 482.02 | +114.51 | 6/15 |
| gear | Vanilla | 2644.08 +/- 481.35 | +0.00 | 0/0 |
| gear | Vanilla | 2644.08 +/- 481.35 | +0.00 | 0/15 |
| gear | a2p20 | 2055.37 +/- 345.30 | +500.22 | 15/15 |
| gear | a2p40 | 2221.58 +/- 387.95 | +360.13 | 15/15 |
| gear | a2p55 | 2795.67 +/- 524.88 | -167.82 | 2/15 |
| gear | a2p65 | 2079.96 +/- 339.43 | +477.66 | 15/15 |
| gear | a2p70 | 2416.59 +/- 436.17 | +172.34 | 15/15 |
| gear | a2p75 | 2550.82 +/- 499.29 | +108.68 | 14/15 |
| gear | a2p80 | 2977.88 +/- 525.23 | -291.55 | 1/15 |
| gear | a3p00 | 2494.97 +/- 457.26 | +143.98 | 15/15 |
| gear | TV cap=3.05 | 2562.31 +/- 487.98 | +84.22 | 13/15 |
| gear | a3p10 | 2832.98 +/- 543.70 | -172.60 | 0/15 |
| gear | a3p20 | 2367.29 +/- 466.77 | +268.85 | 15/15 |
| gear | a3p40 | 2770.06 +/- 529.61 | -59.28 | 3/15 |
| gear | a3p70 | 2561.30 +/- 482.02 | +79.38 | 13/15 |
| gravity | Vanilla | 2058.62 +/- 369.40 | +0.00 | 0/0 |
| gravity | Vanilla | 2058.62 +/- 369.40 | +0.00 | 0/15 |
| gravity | a2p20 | 1302.25 +/- 265.64 | +26.47 | 10/15 |
| gravity | a2p40 | 1613.61 +/- 349.58 | -14.39 | 8/15 |
| gravity | a2p55 | 2067.04 +/- 438.82 | -27.46 | 9/15 |
| gravity | a2p65 | 1577.71 +/- 263.05 | -54.27 | 1/15 |
| gravity | a2p70 | 1819.21 +/- 327.09 | -17.87 | 7/15 |
| gravity | a2p75 | 1979.84 +/- 448.18 | +7.68 | 10/15 |
| gravity | a2p80 | 2084.02 +/- 421.84 | +65.88 | 10/15 |
| gravity | a3p00 | 1910.12 +/- 395.06 | +22.47 | 11/15 |
| gravity | TV cap=3.05 | 2017.68 +/- 456.25 | -26.76 | 4/15 |
| gravity | a3p10 | 2078.76 +/- 463.06 | +45.80 | 11/15 |
| gravity | a3p20 | 1831.17 +/- 335.67 | -13.72 | 6/15 |
| gravity | a3p40 | 2157.39 +/- 448.95 | -36.98 | 7/15 |
| gravity | a3p70 | 1915.41 +/- 344.42 | +11.09 | 9/15 |
| mass | Vanilla | 2644.08 +/- 481.35 | +0.00 | 0/0 |
| mass | Vanilla | 2644.08 +/- 481.35 | +0.00 | 0/14 |
| mass | a2p20 | 2059.20 +/- 339.38 | +378.73 | 14/14 |
| mass | a2p40 | 2221.58 +/- 387.95 | +279.64 | 14/14 |
| mass | a2p55 | 2795.67 +/- 524.88 | -108.26 | 5/14 |
| mass | a2p65 | 2079.96 +/- 339.43 | +433.12 | 14/14 |
| mass | a2p70 | 2416.59 +/- 436.17 | +152.18 | 13/14 |
| mass | a2p75 | 2550.82 +/- 499.29 | +64.65 | 9/14 |
| mass | a2p80 | 3000.79 +/- 523.58 | -310.97 | 2/14 |
| mass | a3p00 | 2494.97 +/- 457.26 | +131.13 | 12/14 |
| mass | TV cap=3.05 | 2562.31 +/- 487.98 | +38.91 | 10/14 |
| mass | a3p10 | 2832.98 +/- 543.70 | -92.85 | 3/14 |
| mass | a3p20 | 2367.29 +/- 466.77 | +152.75 | 13/14 |
| mass | a3p40 | 2765.16 +/- 525.04 | -78.03 | 5/14 |
| mass | a3p70 | 2561.30 +/- 482.02 | +30.91 | 10/14 |

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
- `plots/with_variance/gear_return_curve.png`
- `plots/with_variance/gear_return_curve.pdf`
- `plots/with_variance/gear_gain_curve.png`
- `plots/with_variance/gear_gain_curve.pdf`
- `plots/without_variance/gear_return_curve.png`
- `plots/without_variance/gear_return_curve.pdf`
- `plots/without_variance/gear_gain_curve.png`
- `plots/without_variance/gear_gain_curve.pdf`
- `plots/with_variance/gravity_return_curve.png`
- `plots/with_variance/gravity_return_curve.pdf`
- `plots/with_variance/gravity_gain_curve.png`
- `plots/with_variance/gravity_gain_curve.pdf`
- `plots/without_variance/gravity_return_curve.png`
- `plots/without_variance/gravity_return_curve.pdf`
- `plots/without_variance/gravity_gain_curve.png`
- `plots/without_variance/gravity_gain_curve.pdf`
- `plots/with_variance/mass_return_curve.png`
- `plots/with_variance/mass_return_curve.pdf`
- `plots/with_variance/mass_gain_curve.png`
- `plots/with_variance/mass_gain_curve.pdf`
- `plots/without_variance/mass_return_curve.png`
- `plots/without_variance/mass_return_curve.pdf`
- `plots/without_variance/mass_gain_curve.png`
- `plots/without_variance/mass_gain_curve.pdf`

## Output CSV files

- `outputs/eval_metrics_final.csv`: latest merged per-seed eval rows.
- `outputs/summary_by_scenario.csv`: per-model, per-axis, per-factor aggregate return table.
- `outputs/drop_summary.csv`: nominal-minus-perturbed drop table.
- `outputs/gain_summary.csv`: paired robust-gain table against vanilla.
- `outputs/axis_overview.csv`: compact axis-level overview.
- `outputs/curve_points.csv`: same data used for return plots.
- `outputs/gain_curve_points.csv`: same data used for gain plots.
