# PPO_HalfCheetah_physical_20260423

Date: 2026-04-26

This folder packages the final pinned robustness evaluation for normalized PPO.

## Contents

- Raw metrics: `/home/users/am1015/cleanrl/sweeps/results/PPO_HalfCheetah_physical_20260423/raw_metrics`
- Aggregated outputs: `/home/users/am1015/cleanrl/sweeps/results/PPO_HalfCheetah_physical_20260423/outputs`
- Plots: `/home/users/am1015/cleanrl/sweeps/results/PPO_HalfCheetah_physical_20260423/plots`

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
| friction | 2042.37 +/- 781.90 | 1414.17 +/- 75.62 | 1815.95 +/- 856.77 | 2095.87 +/- 800.14 | 1418.56 +/- 48.14 | 1387.27 +/- 49.10 | 1842.95 +/- 732.39 | 1732.44 +/- 627.80 | 1359.19 +/- 26.60 | 1447.29 +/- 76.48 | 2383.99 +/- 1124.07 |
| gear | 2059.63 +/- 771.25 | 1491.59 +/- 216.30 | 1818.88 +/- 860.94 | 2035.27 +/- 722.97 | 1448.89 +/- 94.42 | 1394.24 +/- 66.80 | 1869.22 +/- 761.78 | 1720.92 +/- 624.69 | 1374.31 +/- 31.56 | 1436.08 +/- 79.58 | 2372.25 +/- 1066.58 |
| gravity | 1383.43 +/- 418.46 | 931.16 +/- 122.99 | 1233.29 +/- 555.29 | 1611.66 +/- 646.33 | 1030.32 +/- 126.75 | 956.93 +/- 129.93 | 1313.72 +/- 447.53 | 1204.92 +/- 381.74 | 956.24 +/- 80.51 | 933.97 +/- 231.08 | 1529.26 +/- 727.49 |
| mass | 2090.15 +/- 812.08 | 1445.26 +/- 129.34 | 1831.39 +/- 882.13 | 2105.36 +/- 825.36 | 1428.30 +/- 127.07 | 1433.95 +/- 44.55 | 1871.36 +/- 764.86 | 1740.84 +/- 615.23 | 1370.90 +/- 28.24 | 1396.92 +/- 90.70 | 2351.82 +/- 1068.23 |

## Axis overview

| Axis | Model | Nominal return | Mean gain over perturbed scenarios | Positive gain scenarios |
| --- | --- | --- | --- | --- |
| friction | No-op (1e9) | 1414.17 +/- 75.62 | +26.70 | 7/10 |
| friction | TV cap=2.85 | 1815.95 +/- 856.77 | -12.48 | 5/10 |
| friction | TV cap=2.95 | 2095.87 +/- 800.14 | -26.69 | 5/10 |
| friction | TV cap=3.00 | 1418.56 +/- 48.14 | +38.71 | 7/10 |
| friction | TV cap=3.05 | 1387.27 +/- 49.10 | +32.39 | 7/10 |
| friction | TV cap=3.10 | 1842.95 +/- 732.39 | +13.73 | 4/10 |
| friction | TV cap=3.20 | 1732.44 +/- 627.80 | -45.53 | 4/10 |
| friction | TV cap=3.50 | 1359.19 +/- 26.60 | +15.32 | 5/10 |
| friction | TV cap=3.70 | 1447.29 +/- 76.48 | -20.11 | 4/10 |
| friction | TV cap=4.00 | 2383.99 +/- 1124.07 | -54.14 | 3/10 |
| gear | No-op (1e9) | 1491.59 +/- 216.30 | +411.73 | 9/10 |
| gear | TV cap=2.85 | 1818.88 +/- 860.94 | +277.31 | 10/10 |
| gear | TV cap=2.95 | 2035.27 +/- 722.97 | +78.20 | 7/10 |
| gear | TV cap=3.00 | 1448.89 +/- 94.42 | +490.19 | 10/10 |
| gear | TV cap=3.05 | 1394.24 +/- 66.80 | +473.50 | 9/10 |
| gear | TV cap=3.10 | 1869.22 +/- 761.78 | +194.42 | 10/10 |
| gear | TV cap=3.20 | 1720.92 +/- 624.69 | +266.64 | 9/10 |
| gear | TV cap=3.50 | 1374.31 +/- 31.56 | +513.13 | 10/10 |
| gear | TV cap=3.70 | 1436.08 +/- 79.58 | +486.45 | 10/10 |
| gear | TV cap=4.00 | 2372.25 +/- 1066.58 | -152.10 | 1/10 |
| gravity | No-op (1e9) | 931.16 +/- 122.99 | +103.28 | 7/10 |
| gravity | TV cap=2.85 | 1233.29 +/- 555.29 | +45.02 | 7/10 |
| gravity | TV cap=2.95 | 1611.66 +/- 646.33 | +46.84 | 6/10 |
| gravity | TV cap=3.00 | 1030.32 +/- 126.75 | +21.16 | 5/10 |
| gravity | TV cap=3.05 | 956.93 +/- 129.93 | +28.30 | 6/10 |
| gravity | TV cap=3.10 | 1313.72 +/- 447.53 | +71.45 | 8/10 |
| gravity | TV cap=3.20 | 1204.92 +/- 381.74 | +61.62 | 8/10 |
| gravity | TV cap=3.50 | 956.24 +/- 80.51 | +81.17 | 6/10 |
| gravity | TV cap=3.70 | 933.97 +/- 231.08 | +90.25 | 7/10 |
| gravity | TV cap=4.00 | 1529.26 +/- 727.49 | +65.09 | 10/10 |
| mass | No-op (1e9) | 1445.26 +/- 129.34 | +324.98 | 10/10 |
| mass | TV cap=2.85 | 1831.39 +/- 882.13 | +116.82 | 7/10 |
| mass | TV cap=2.95 | 2105.36 +/- 825.36 | +60.59 | 6/10 |
| mass | TV cap=3.00 | 1428.30 +/- 127.07 | +351.42 | 10/10 |
| mass | TV cap=3.05 | 1433.95 +/- 44.55 | +260.77 | 9/10 |
| mass | TV cap=3.10 | 1871.36 +/- 764.86 | +122.06 | 9/10 |
| mass | TV cap=3.20 | 1740.84 +/- 615.23 | +127.48 | 9/10 |
| mass | TV cap=3.50 | 1370.90 +/- 28.24 | +362.54 | 10/10 |
| mass | TV cap=3.70 | 1396.92 +/- 90.70 | +326.54 | 10/10 |
| mass | TV cap=4.00 | 2351.82 +/- 1068.23 | -41.81 | 5/10 |

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
