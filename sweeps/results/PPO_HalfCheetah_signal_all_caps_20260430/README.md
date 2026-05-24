# PPO_HalfCheetah_signal_all_caps_20260430

Date: 2026-05-15

This folder packages the final pinned robustness evaluation for normalized PPO.

## Contents

- Raw metrics: `/Users/arjunmahesh/Library/CloudStorage/OneDrive-Personal/Duke/Research/RL/cleanrl/sweeps/results/PPO_HalfCheetah_final_presentable_20260430/raw_metrics/signal`
- Aggregated outputs: `/Users/arjunmahesh/Library/CloudStorage/OneDrive-Personal/Duke/Research/RL/cleanrl/sweeps/results/PPO_HalfCheetah_signal_all_caps_20260430/outputs`
- Plots: `/Users/arjunmahesh/Library/CloudStorage/OneDrive-Personal/Duke/Research/RL/cleanrl/sweeps/results/PPO_HalfCheetah_signal_all_caps_20260430/plots`

## Evaluation protocol

- Models are compared on the same perturbation grid for the configured axes.
- Nominal reference within each axis is the `factor=0.0` point.
- Curves show mean return across seeds with `95% CI` shading.
- Robust gain is defined as `vanilla_drop - model_drop`; positive is better.
- Plot files are exported in both `PNG` and vector `PDF` format.

## Model labels

- `vanilla` -> Vanilla
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

| Axis | Vanilla | TV cap=2.20 | TV cap=2.40 | TV cap=2.55 | TV cap=2.65 | TV cap=2.70 | TV cap=2.75 | TV cap=2.80 | TV cap=3.00 | TV cap=3.05 | TV cap=3.10 | TV cap=3.20 | TV cap=3.40 | TV cap=3.70 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| action_noise | 2003.65 +/- 778.78 | 1795.91 +/- 771.89 | 1878.03 +/- 737.82 | 2298.69 +/- 1071.72 | 1707.82 +/- 625.35 | 1418.17 +/- 82.52 | 1385.49 +/- 31.82 | 1839.57 +/- 933.03 | 1400.48 +/- 69.91 | 1424.31 +/- 71.38 | 1396.33 +/- 54.60 | 1424.91 +/- 94.06 | 1379.03 +/- 89.14 | 1429.81 +/- 87.50 |
| action_noise_bernoulli | 2057.55 +/- 838.97 | 1776.83 +/- 724.71 | 1856.02 +/- 717.63 | 2159.36 +/- 910.10 | 1756.51 +/- 736.13 | 1415.04 +/- 78.84 | 1413.88 +/- 44.55 | 1861.42 +/- 912.85 | 1435.34 +/- 89.89 | 1406.33 +/- 61.84 | 1378.12 +/- 74.24 | 1427.21 +/- 82.82 | 1392.79 +/- 73.99 | 1420.03 +/- 89.24 |
| state_noise | 2041.98 +/- 822.86 | 1734.80 +/- 648.27 | 1852.35 +/- 689.41 | 2179.48 +/- 946.01 | 1789.61 +/- 794.76 | 1418.27 +/- 78.42 | 1408.70 +/- 50.38 | 1735.36 +/- 679.08 | 1411.57 +/- 58.27 | 1408.07 +/- 45.51 | 1378.31 +/- 79.87 | 1419.31 +/- 90.37 | 1394.57 +/- 90.31 | 1428.61 +/- 82.71 |

## Axis overview

| Axis | Model | Nominal return | Mean gain over perturbed scenarios | Positive gain scenarios |
| --- | --- | --- | --- | --- |

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
- `plots/with_variance/action_noise_return_curve.png`
- `plots/with_variance/action_noise_return_curve.pdf`
- `plots/with_variance/action_noise_gain_curve.png`
- `plots/with_variance/action_noise_gain_curve.pdf`
- `plots/without_variance/action_noise_return_curve.png`
- `plots/without_variance/action_noise_return_curve.pdf`
- `plots/without_variance/action_noise_gain_curve.png`
- `plots/without_variance/action_noise_gain_curve.pdf`
- `plots/with_variance/action_noise_bernoulli_return_curve.png`
- `plots/with_variance/action_noise_bernoulli_return_curve.pdf`
- `plots/with_variance/action_noise_bernoulli_gain_curve.png`
- `plots/with_variance/action_noise_bernoulli_gain_curve.pdf`
- `plots/without_variance/action_noise_bernoulli_return_curve.png`
- `plots/without_variance/action_noise_bernoulli_return_curve.pdf`
- `plots/without_variance/action_noise_bernoulli_gain_curve.png`
- `plots/without_variance/action_noise_bernoulli_gain_curve.pdf`
- `plots/with_variance/state_noise_return_curve.png`
- `plots/with_variance/state_noise_return_curve.pdf`
- `plots/with_variance/state_noise_gain_curve.png`
- `plots/with_variance/state_noise_gain_curve.pdf`
- `plots/without_variance/state_noise_return_curve.png`
- `plots/without_variance/state_noise_return_curve.pdf`
- `plots/without_variance/state_noise_gain_curve.png`
- `plots/without_variance/state_noise_gain_curve.pdf`

## Output CSV files

- `outputs/eval_metrics_final.csv`: latest merged per-seed eval rows.
- `outputs/summary_by_scenario.csv`: per-model, per-axis, per-factor aggregate return table.
- `outputs/drop_summary.csv`: nominal-minus-perturbed drop table.
- `outputs/gain_summary.csv`: paired robust-gain table against vanilla.
- `outputs/axis_overview.csv`: compact axis-level overview.
- `outputs/curve_points.csv`: same data used for return plots.
- `outputs/gain_curve_points.csv`: same data used for gain plots.
