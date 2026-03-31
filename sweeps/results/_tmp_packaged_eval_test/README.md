# Tmp Robust Eval Test

Date: 2026-03-30

This folder packages the final pinned robustness evaluation for normalized PPO.

## Contents

- Raw metrics: `C:/Users/Arjun/OneDrive/Duke/Research/RL/cleanrl/sweeps/results/PPO_FixedAlpha_vs_P95_friction_20260306/outputs/raw_metrics`
- Aggregated outputs: `C:/Users/Arjun/OneDrive/Duke/Research/RL/cleanrl/sweeps/results/_tmp_packaged_eval_test/outputs`
- Plots: `C:/Users/Arjun/OneDrive/Duke/Research/RL/cleanrl/sweeps/results/_tmp_packaged_eval_test/plots`

## Evaluation protocol

- Models are compared on the same perturbation grid for `friction`, `mass`, and `damping`.
- Nominal reference within each axis is the `factor=1.0` point.
- Curves show mean return across seeds with `95% CI` shading.
- Robust gain is defined as `vanilla_drop - model_drop`; positive is better.

## Model labels

- `vanilla` -> Vanilla
- `tv_a3200` -> tv_a3200
- `tv_a3600` -> tv_a3600
- `tv_a4000` -> tv_a4000
- `tv_a4400` -> tv_a4400
- `tv_a4800` -> tv_a4800
- `tv_p95` -> tv_p95

## Nominal returns by axis

| Axis | Vanilla | tv_a3200 | tv_a3600 | tv_a4000 | tv_a4400 | tv_a4800 | tv_p95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| friction | 3162.23 +/- 638.55 | 2582.66 +/- 635.59 | 2946.12 +/- 899.81 | 3200.57 +/- 630.07 | 2754.81 +/- 456.17 | 2599.45 +/- 291.95 | 2745.95 +/- 440.69 |
| nominal | 3182.98 +/- 650.92 | 2518.83 +/- 519.50 | 2944.23 +/- 874.19 | 3374.31 +/- 604.50 | 2746.16 +/- 574.29 | 2631.97 +/- 360.69 | 2769.36 +/- 485.39 |

## Axis overview

| Axis | Model | Nominal return | Mean gain over perturbed scenarios | Positive gain scenarios |
| --- | --- | --- | --- | --- |

## Plot files

- `plots/return_curves_panel.png`
- `plots/gain_curves_panel.png`
- `plots/friction_return_curve.png`
- `plots/friction_gain_curve.png`
- `plots/nominal_return_curve.png`
- `plots/nominal_gain_curve.png`

## Output CSV files

- `outputs/eval_metrics_final.csv`: latest merged per-seed eval rows.
- `outputs/summary_by_scenario.csv`: per-model, per-axis, per-factor aggregate return table.
- `outputs/drop_summary.csv`: nominal-minus-perturbed drop table.
- `outputs/gain_summary.csv`: paired robust-gain table against vanilla.
- `outputs/axis_overview.csv`: compact axis-level overview.
- `outputs/curve_points.csv`: same data used for return plots.
- `outputs/gain_curve_points.csv`: same data used for gain plots.
