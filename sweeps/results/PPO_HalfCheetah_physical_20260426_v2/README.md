# PPO_HalfCheetah_physical_20260426_v2

Date: 2026-04-27

This folder packages the final pinned robustness evaluation for normalized PPO.

## Contents

- Raw metrics: `/home/users/am1015/cleanrl/sweeps/results/PPO_HalfCheetah_physical_20260426_v2/raw_metrics`
- Aggregated outputs: `/home/users/am1015/cleanrl/sweeps/results/PPO_HalfCheetah_physical_20260426_v2/outputs`
- Plots: `/home/users/am1015/cleanrl/sweeps/results/PPO_HalfCheetah_physical_20260426_v2/plots`

## Evaluation protocol

- Models are compared on the same perturbation grid for the configured axes.
- Nominal reference within each axis is the `factor=1.0` point.
- Curves show mean return across seeds with `95% CI` shading.
- Robust gain is defined as `vanilla_drop - model_drop`; positive is better.

## Model labels

- `vanilla` -> Vanilla
- `a1e9` -> No-op (1e9)
- `a2p70` -> TV cap=2.70
- `a3p00` -> TV cap=3.00
- `a3p05` -> TV cap=3.05
- `a3p10` -> TV cap=3.10
- `a3p20` -> TV cap=3.20
- `a3p40` -> TV cap=3.40
- `a3p70` -> TV cap=3.70

## Nominal returns by axis

| Axis | Vanilla | No-op (1e9) | TV cap=2.70 | TV cap=3.00 | TV cap=3.05 | TV cap=3.10 | TV cap=3.20 | TV cap=3.40 | TV cap=3.70 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| friction | 1738.54 +/- 657.46 | 1735.15 +/- 398.98 | 1327.52 +/- 0.00 | 1421.97 +/- 117.26 | 1397.43 +/- 51.14 | 1365.97 +/- 71.63 | 1434.56 +/- 83.42 | 1391.46 +/- 95.40 | 1416.26 +/- 85.59 |
| gear | 1698.70 +/- 584.20 | 1845.29 +/- 512.07 | 1338.17 +/- 0.00 | 1411.33 +/- 108.64 | 1423.82 +/- 63.77 | 1393.70 +/- 51.92 | 1441.78 +/- 79.22 | 1402.34 +/- 78.63 | 1407.45 +/- 96.59 |
| gravity | 1054.94 +/- 85.07 | 1177.31 +/- 127.62 | 979.51 +/- 0.00 | 1066.97 +/- 132.93 | 952.79 +/- 175.29 | 797.18 +/- 104.37 | 1072.41 +/- 141.34 | 1020.39 +/- 146.58 | 997.97 +/- 147.66 |
| mass | 1626.72 +/- 461.70 | 1795.33 +/- 467.68 | 1333.02 +/- 0.00 | 1429.26 +/- 105.76 | 1425.78 +/- 52.67 | 1397.38 +/- 57.68 | 1438.96 +/- 86.48 | 1399.10 +/- 90.53 | 1384.21 +/- 104.98 |

## Axis overview

| Axis | Model | Nominal return | Mean gain over perturbed scenarios | Positive gain scenarios |
| --- | --- | --- | --- | --- |
| friction | No-op (1e9) | 1735.15 +/- 398.98 | +116.24 | 10/10 |
| friction | TV cap=2.70 | 1327.52 +/- 0.00 | +18.11 | 9/10 |
| friction | TV cap=3.00 | 1421.97 +/- 117.26 | +112.31 | 10/10 |
| friction | TV cap=3.05 | 1397.43 +/- 51.14 | +15.94 | 6/10 |
| friction | TV cap=3.10 | 1365.97 +/- 71.63 | +109.22 | 10/10 |
| friction | TV cap=3.20 | 1434.56 +/- 83.42 | +103.32 | 10/10 |
| friction | TV cap=3.40 | 1391.46 +/- 95.40 | +93.98 | 10/10 |
| friction | TV cap=3.70 | 1416.26 +/- 85.59 | +108.96 | 10/10 |
| gear | No-op (1e9) | 1845.29 +/- 512.07 | +62.81 | 8/10 |
| gear | TV cap=2.70 | 1338.17 +/- 0.00 | +280.02 | 10/10 |
| gear | TV cap=3.00 | 1411.33 +/- 108.64 | +254.64 | 10/10 |
| gear | TV cap=3.05 | 1423.82 +/- 63.77 | -35.49 | 4/10 |
| gear | TV cap=3.10 | 1393.70 +/- 51.92 | +265.14 | 10/10 |
| gear | TV cap=3.20 | 1441.78 +/- 79.22 | +250.74 | 10/10 |
| gear | TV cap=3.40 | 1402.34 +/- 78.63 | +256.01 | 10/10 |
| gear | TV cap=3.70 | 1407.45 +/- 96.59 | +271.33 | 10/10 |
| gravity | No-op (1e9) | 1177.31 +/- 127.62 | -69.05 | 1/11 |
| gravity | TV cap=2.70 | 979.51 +/- 0.00 | -54.27 | 2/11 |
| gravity | TV cap=3.00 | 1066.97 +/- 132.93 | -63.56 | 5/11 |
| gravity | TV cap=3.05 | 952.79 +/- 175.29 | -109.15 | 3/11 |
| gravity | TV cap=3.10 | 797.18 +/- 104.37 | +16.19 | 7/11 |
| gravity | TV cap=3.20 | 1072.41 +/- 141.34 | -60.34 | 5/11 |
| gravity | TV cap=3.40 | 1020.39 +/- 146.58 | -26.07 | 5/11 |
| gravity | TV cap=3.70 | 997.97 +/- 147.66 | -59.81 | 4/11 |
| mass | No-op (1e9) | 1795.33 +/- 467.68 | +12.90 | 3/10 |
| mass | TV cap=2.70 | 1333.02 +/- 0.00 | +135.58 | 8/10 |
| mass | TV cap=3.00 | 1429.26 +/- 105.76 | +155.04 | 7/10 |
| mass | TV cap=3.05 | 1425.78 +/- 52.67 | -8.04 | 3/10 |
| mass | TV cap=3.10 | 1397.38 +/- 57.68 | +154.87 | 7/10 |
| mass | TV cap=3.20 | 1438.96 +/- 86.48 | +101.17 | 7/10 |
| mass | TV cap=3.40 | 1399.10 +/- 90.53 | +143.10 | 7/10 |
| mass | TV cap=3.70 | 1384.21 +/- 104.98 | +118.14 | 6/10 |

## Plot files

- `plots/return_curves_panel.png`
- `plots/gain_curves_panel.png`
- `plots/friction_return_curve.png`
- `plots/friction_gain_curve.png`
- `plots/gear_return_curve.png`
- `plots/gear_gain_curve.png`
- `plots/gravity_return_curve.png`
- `plots/gravity_gain_curve.png`
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
