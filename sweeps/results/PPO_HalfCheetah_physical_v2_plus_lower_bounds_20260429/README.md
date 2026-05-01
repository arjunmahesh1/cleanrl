# PPO_HalfCheetah_physical_v2_plus_lower_bounds_20260429

Date: 2026-04-29

This folder packages the final pinned robustness evaluation for normalized PPO.

## Contents

- Raw metrics: `/Users/arjunmahesh/Library/CloudStorage/OneDrive-Personal/Duke/Research/RL/cleanrl/sweeps/results/PPO_HalfCheetah_physical_v2_plus_lower_bounds_20260429/raw_metrics`
- Aggregated outputs: `/Users/arjunmahesh/Library/CloudStorage/OneDrive-Personal/Duke/Research/RL/cleanrl/sweeps/results/PPO_HalfCheetah_physical_v2_plus_lower_bounds_20260429/outputs`
- Plots: `/Users/arjunmahesh/Library/CloudStorage/OneDrive-Personal/Duke/Research/RL/cleanrl/sweeps/results/PPO_HalfCheetah_physical_v2_plus_lower_bounds_20260429/plots`

## Evaluation protocol

- Models are compared on the same perturbation grid for the configured axes.
- Nominal reference within each axis is the `factor=1.0` point.
- Curves show mean return across seeds with `95% CI` shading.
- Robust gain is defined as `vanilla_drop - model_drop`; positive is better.

## Model labels

- `vanilla` -> Vanilla
- `a1e9` -> No-op (1e9)
- `a2p20` -> TV cap=2.20
- `a2p40` -> TV cap=2.40
- `a2p55` -> TV cap=2.55
- `a2p65` -> TV cap=2.65
- `a2p70` -> TV cap=2.70
- `a2p75` -> TV cap=2.75
- `a2p80` -> TV cap=2.80
- `a3p00` -> TV cap=3.00
- `a3p05` -> TV cap=3.05
- `a3p10` -> TV cap=3.10
- `a3p20` -> TV cap=3.20
- `a3p40` -> TV cap=3.40
- `a3p70` -> TV cap=3.70

## Nominal returns by axis

| Axis | Vanilla | No-op (1e9) | TV cap=2.20 | TV cap=2.40 | TV cap=2.55 | TV cap=2.65 | TV cap=2.70 | TV cap=2.75 | TV cap=2.80 | TV cap=3.00 | TV cap=3.05 | TV cap=3.10 | TV cap=3.20 | TV cap=3.40 | TV cap=3.70 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| friction | 1802.86 +/- 729.50 | 2314.40 +/- 754.82 | 1781.59 +/- 738.08 | 1879.61 +/- 723.56 | 2228.89 +/- 1000.26 | 1818.06 +/- 842.78 | 1457.48 +/- 86.50 | 1406.77 +/- 47.04 | 1757.96 +/- 753.45 | 1421.97 +/- 117.26 | 1397.43 +/- 51.14 | 1365.97 +/- 71.63 | 1434.56 +/- 83.42 | 1391.46 +/- 95.40 | 1416.26 +/- 85.59 |
| gear | 1833.57 +/- 736.79 | 2240.41 +/- 773.73 | 1798.20 +/- 769.26 | 1876.02 +/- 772.52 | 2203.69 +/- 944.47 | 1749.08 +/- 705.28 | 1458.47 +/- 78.01 | 1407.06 +/- 46.08 | 1820.04 +/- 855.97 | 1411.33 +/- 108.64 | 1423.82 +/- 63.77 | 1393.70 +/- 51.92 | 1441.78 +/- 79.22 | 1402.34 +/- 78.63 | 1407.45 +/- 96.59 |
| gravity | 1343.82 +/- 418.75 | 1405.80 +/- 452.92 | 1293.43 +/- 431.34 | 1296.19 +/- 419.29 | 1623.34 +/- 675.66 | 1259.77 +/- 503.24 | 1115.21 +/- 89.20 | 1082.15 +/- 48.94 | 1352.21 +/- 716.72 | 1066.97 +/- 132.93 | 952.79 +/- 175.29 | 797.18 +/- 104.37 | 1072.41 +/- 141.34 | 1020.39 +/- 146.58 | 997.97 +/- 147.66 |
| mass | 1827.74 +/- 725.21 | 2304.17 +/- 772.93 | 1777.85 +/- 780.83 | 1870.65 +/- 698.51 | 2184.89 +/- 977.27 | 1760.54 +/- 735.20 | 1446.58 +/- 89.74 | 1407.62 +/- 43.67 | 1829.12 +/- 858.70 | 1429.26 +/- 105.76 | 1425.78 +/- 52.67 | 1397.38 +/- 57.68 | 1438.96 +/- 86.48 | 1399.10 +/- 90.53 | 1384.21 +/- 104.98 |

## Axis overview

| Axis | Model | Nominal return | Mean gain over perturbed scenarios | Positive gain scenarios |
| --- | --- | --- | --- | --- |
| friction | No-op (1e9) | 2314.40 +/- 754.82 | -148.08 | 2/10 |
| friction | TV cap=2.20 | 1781.59 +/- 738.08 | -43.38 | 3/10 |
| friction | TV cap=2.40 | 1879.61 +/- 723.56 | -37.11 | 1/10 |
| friction | TV cap=2.55 | 2228.89 +/- 1000.26 | -101.17 | 2/10 |
| friction | TV cap=2.65 | 1818.06 +/- 842.78 | -76.95 | 1/10 |
| friction | TV cap=2.70 | 1457.48 +/- 86.50 | -0.80 | 6/10 |
| friction | TV cap=2.75 | 1406.77 +/- 47.04 | -17.16 | 3/10 |
| friction | TV cap=2.80 | 1757.96 +/- 753.45 | +45.90 | 7/10 |
| friction | TV cap=3.00 | 1421.97 +/- 117.26 | -17.73 | 3/10 |
| friction | TV cap=3.05 | 1397.43 +/- 51.14 | +14.87 | 4/10 |
| friction | TV cap=3.10 | 1365.97 +/- 71.63 | +3.33 | 3/10 |
| friction | TV cap=3.20 | 1434.56 +/- 83.42 | -5.41 | 3/10 |
| friction | TV cap=3.40 | 1391.46 +/- 95.40 | -7.95 | 3/10 |
| friction | TV cap=3.70 | 1416.26 +/- 85.59 | -3.83 | 3/10 |
| gear | No-op (1e9) | 2240.41 +/- 773.73 | -203.79 | 1/10 |
| gear | TV cap=2.20 | 1798.20 +/- 769.26 | -47.72 | 4/10 |
| gear | TV cap=2.40 | 1876.02 +/- 772.52 | -68.42 | 2/10 |
| gear | TV cap=2.55 | 2203.69 +/- 944.47 | -263.39 | 0/10 |
| gear | TV cap=2.65 | 1749.08 +/- 705.28 | +8.73 | 3/10 |
| gear | TV cap=2.70 | 1458.47 +/- 78.01 | +290.17 | 9/10 |
| gear | TV cap=2.75 | 1407.06 +/- 46.08 | +283.39 | 9/10 |
| gear | TV cap=2.80 | 1820.04 +/- 855.97 | -85.01 | 4/10 |
| gear | TV cap=3.00 | 1411.33 +/- 108.64 | +347.48 | 9/10 |
| gear | TV cap=3.05 | 1423.82 +/- 63.77 | +351.29 | 8/10 |
| gear | TV cap=3.10 | 1393.70 +/- 51.92 | +274.32 | 8/10 |
| gear | TV cap=3.20 | 1441.78 +/- 79.22 | +258.84 | 9/10 |
| gear | TV cap=3.40 | 1402.34 +/- 78.63 | +270.18 | 9/10 |
| gear | TV cap=3.70 | 1407.45 +/- 96.59 | +288.20 | 9/10 |
| gravity | No-op (1e9) | 1405.80 +/- 452.92 | +7.82 | 5/11 |
| gravity | TV cap=2.20 | 1293.43 +/- 431.34 | -1.81 | 7/10 |
| gravity | TV cap=2.40 | 1296.19 +/- 419.29 | -91.92 | 3/10 |
| gravity | TV cap=2.55 | 1623.34 +/- 675.66 | -48.38 | 5/10 |
| gravity | TV cap=2.65 | 1259.77 +/- 503.24 | +83.49 | 10/10 |
| gravity | TV cap=2.70 | 1115.21 +/- 89.20 | +27.89 | 8/11 |
| gravity | TV cap=2.75 | 1082.15 +/- 48.94 | +32.41 | 5/10 |
| gravity | TV cap=2.80 | 1352.21 +/- 716.72 | +13.52 | 6/10 |
| gravity | TV cap=3.00 | 1066.97 +/- 132.93 | -18.15 | 5/11 |
| gravity | TV cap=3.05 | 952.79 +/- 175.29 | -11.41 | 4/11 |
| gravity | TV cap=3.10 | 797.18 +/- 104.37 | +77.95 | 10/11 |
| gravity | TV cap=3.20 | 1072.41 +/- 141.34 | -0.94 | 6/11 |
| gravity | TV cap=3.40 | 1020.39 +/- 146.58 | +34.66 | 7/11 |
| gravity | TV cap=3.70 | 997.97 +/- 147.66 | -10.35 | 5/11 |
| mass | No-op (1e9) | 2304.17 +/- 772.93 | -191.38 | 0/10 |
| mass | TV cap=2.20 | 1777.85 +/- 780.83 | +92.84 | 7/10 |
| mass | TV cap=2.40 | 1870.65 +/- 698.51 | +5.97 | 5/10 |
| mass | TV cap=2.55 | 2184.89 +/- 977.27 | -89.98 | 5/10 |
| mass | TV cap=2.65 | 1760.54 +/- 735.20 | +52.30 | 6/10 |
| mass | TV cap=2.70 | 1446.58 +/- 89.74 | +189.73 | 10/10 |
| mass | TV cap=2.75 | 1407.62 +/- 43.67 | +136.57 | 10/10 |
| mass | TV cap=2.80 | 1829.12 +/- 858.70 | -106.28 | 1/10 |
| mass | TV cap=3.00 | 1429.26 +/- 105.76 | +196.98 | 10/10 |
| mass | TV cap=3.05 | 1425.78 +/- 52.67 | +193.21 | 9/10 |
| mass | TV cap=3.10 | 1397.38 +/- 57.68 | +188.30 | 9/10 |
| mass | TV cap=3.20 | 1438.96 +/- 86.48 | +146.70 | 10/10 |
| mass | TV cap=3.40 | 1399.10 +/- 90.53 | +174.62 | 10/10 |
| mass | TV cap=3.70 | 1384.21 +/- 104.98 | +161.14 | 10/10 |

## Plot files

- `plots/with_variance/`: full plot set with variance whiskers.
- `plots/without_variance/`: matching plot set without variance whiskers.
- `plots/with_variance/return_curves_panel.png`
- `plots/with_variance/gain_curves_panel.png`
- `plots/without_variance/return_curves_panel.png`
- `plots/without_variance/gain_curves_panel.png`
- `plots/with_variance/friction_return_curve.png`
- `plots/with_variance/friction_gain_curve.png`
- `plots/without_variance/friction_return_curve.png`
- `plots/without_variance/friction_gain_curve.png`
- `plots/with_variance/gear_return_curve.png`
- `plots/with_variance/gear_gain_curve.png`
- `plots/without_variance/gear_return_curve.png`
- `plots/without_variance/gear_gain_curve.png`
- `plots/with_variance/gravity_return_curve.png`
- `plots/with_variance/gravity_gain_curve.png`
- `plots/without_variance/gravity_return_curve.png`
- `plots/without_variance/gravity_gain_curve.png`
- `plots/with_variance/mass_return_curve.png`
- `plots/with_variance/mass_gain_curve.png`
- `plots/without_variance/mass_return_curve.png`
- `plots/without_variance/mass_gain_curve.png`

## Output CSV files

- `outputs/eval_metrics_final.csv`: latest merged per-seed eval rows.
- `outputs/summary_by_scenario.csv`: per-model, per-axis, per-factor aggregate return table.
- `outputs/drop_summary.csv`: nominal-minus-perturbed drop table.
- `outputs/gain_summary.csv`: paired robust-gain table against vanilla.
- `outputs/axis_overview.csv`: compact axis-level overview.
- `outputs/curve_points.csv`: same data used for return plots.
- `outputs/gain_curve_points.csv`: same data used for gain plots.
