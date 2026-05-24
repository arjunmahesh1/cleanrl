# PPO_Walker_fmd_nonmass_0p0_2p0_20260430

Date: 2026-05-15

This folder packages the final pinned robustness evaluation for normalized PPO.

## Contents

- Raw metrics: `/Users/arjunmahesh/Library/CloudStorage/OneDrive-Personal/Duke/Research/RL/cleanrl/sweeps/results/PPO_Walker_final_presentable_20260430/raw_metrics/combo_nonmass`
- Aggregated outputs: `/Users/arjunmahesh/Library/CloudStorage/OneDrive-Personal/Duke/Research/RL/cleanrl/sweeps/results/PPO_Walker_fmd_nonmass_0p0_2p0_20260430/outputs`
- Plots: `/Users/arjunmahesh/Library/CloudStorage/OneDrive-Personal/Duke/Research/RL/cleanrl/sweeps/results/PPO_Walker_fmd_nonmass_0p0_2p0_20260430/plots`

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
| friction_damping | 2864.34 +/- 613.86 | 2180.05 +/- 976.16 | 2979.39 +/- 823.14 | 2562.64 +/- 457.18 | 2662.97 +/- 496.32 | 2871.02 +/- 522.58 | 2117.96 +/- 815.18 | 2365.75 +/- 944.25 | 2662.95 +/- 468.62 | 2964.35 +/- 580.49 |

## Axis overview

| Axis | Model | Nominal return | Mean gain over perturbed scenarios | Positive gain scenarios |
| --- | --- | --- | --- | --- |
| friction_damping | TV cap=2.85 | 2180.05 +/- 976.16 | +442.88 | 12/15 |
| friction_damping | TV cap=2.95 | 2979.39 +/- 823.14 | +179.17 | 9/15 |
| friction_damping | TV cap=3.00 | 2562.64 +/- 457.18 | +335.63 | 13/15 |
| friction_damping | TV cap=3.05 | 2662.97 +/- 496.32 | +245.38 | 14/15 |
| friction_damping | TV cap=3.10 | 2871.02 +/- 522.58 | +278.68 | 14/15 |
| friction_damping | TV cap=3.20 | 2117.96 +/- 815.18 | +564.62 | 14/15 |
| friction_damping | TV cap=3.50 | 2365.75 +/- 944.25 | +334.16 | 13/15 |
| friction_damping | TV cap=3.70 | 2662.95 +/- 468.62 | +177.84 | 14/15 |
| friction_damping | TV cap=4.00 | 2964.35 +/- 580.49 | +59.41 | 7/15 |

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

## Output CSV files

- `outputs/eval_metrics_final.csv`: latest merged per-seed eval rows.
- `outputs/summary_by_scenario.csv`: per-model, per-axis, per-factor aggregate return table.
- `outputs/drop_summary.csv`: nominal-minus-perturbed drop table.
- `outputs/gain_summary.csv`: paired robust-gain table against vanilla.
- `outputs/axis_overview.csv`: compact axis-level overview.
- `outputs/curve_points.csv`: same data used for return plots.
- `outputs/gain_curve_points.csv`: same data used for gain plots.
