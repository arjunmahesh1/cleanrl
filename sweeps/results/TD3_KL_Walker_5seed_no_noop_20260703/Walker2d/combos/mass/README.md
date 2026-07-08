# Walker2d TD3-KL 5-Seed: Mass Combos

Date: 2026-07-08

This folder packages the final pinned robustness evaluation for normalized PPO.

## Contents

- Raw metrics: `/home/users/am1015/cleanrl/sweeps/results/TD3_KL_Walker_5seed_no_noop_20260703/Walker2d/combos/mass/raw_metrics`
- Aggregated outputs: `/home/users/am1015/cleanrl/sweeps/results/TD3_KL_Walker_5seed_no_noop_20260703/Walker2d/combos/mass/outputs`
- Plots: `/home/users/am1015/cleanrl/sweeps/results/TD3_KL_Walker_5seed_no_noop_20260703/Walker2d/combos/mass/plots`

## Evaluation protocol

- Models are compared on the same perturbation grid for the configured axes.
- Nominal reference within each axis is the `factor=1.0` point.
- Curves show mean return across seeds with `95% CI` shading.
- Robust gain is defined as `vanilla_drop - model_drop`; positive is better.
- Plot files are exported in both `PNG` and vector `PDF` format.

## Model labels

- `vanilla` -> Vanilla
- `klb0p5` -> klb0p5
- `klb1` -> klb1
- `klb2` -> klb2
- `klb5` -> klb5
- `klb10` -> klb10
- `klb20` -> klb20
- `klb50` -> klb50
- `klb100` -> klb100

## Nominal returns by axis

| Axis | Vanilla | klb0p5 | klb1 | klb2 | klb5 | klb10 | klb20 | klb50 | klb100 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| friction_mass | 2099.77 +/- 1067.34 | 370.45 +/- 58.73 | 444.66 +/- 154.30 | 821.98 +/- 156.40 | 386.18 +/- 222.62 | 474.71 +/- 93.98 | 669.28 +/- 286.53 | 584.61 +/- 113.03 | 603.98 +/- 124.09 |
| friction_mass_damping | 2099.77 +/- 1067.34 | 368.94 +/- 53.52 | 444.66 +/- 154.30 | 821.98 +/- 156.40 | 386.18 +/- 222.62 | 474.71 +/- 93.98 | 669.28 +/- 286.53 | 584.61 +/- 113.03 | 603.98 +/- 124.09 |
| mass_damping | 2151.99 +/- 1174.63 | 368.94 +/- 53.52 | 444.66 +/- 154.30 | 821.98 +/- 156.40 | 392.05 +/- 225.65 | 486.69 +/- 80.90 | 669.28 +/- 286.53 | 584.61 +/- 113.03 | 596.38 +/- 114.66 |

## Axis overview

| Axis | Model | Nominal return | Mean gain over perturbed scenarios | Positive gain scenarios |
| --- | --- | --- | --- | --- |
| friction_mass | klb0p5 | 370.45 +/- 58.73 | +908.29 | 11/14 |
| friction_mass | klb1 | 444.66 +/- 154.30 | +973.25 | 12/14 |
| friction_mass | klb2 | 821.98 +/- 156.40 | +617.79 | 11/14 |
| friction_mass | klb5 | 386.18 +/- 222.62 | +975.79 | 12/14 |
| friction_mass | klb10 | 474.71 +/- 93.98 | +912.83 | 12/14 |
| friction_mass | klb20 | 669.28 +/- 286.53 | +882.86 | 13/14 |
| friction_mass | klb50 | 584.61 +/- 113.03 | +902.92 | 12/14 |
| friction_mass | klb100 | 603.98 +/- 124.09 | +895.46 | 13/14 |
| friction_mass_damping | klb0p5 | 368.94 +/- 53.52 | +951.14 | 12/14 |
| friction_mass_damping | klb1 | 444.66 +/- 154.30 | +1017.37 | 13/14 |
| friction_mass_damping | klb2 | 821.98 +/- 156.40 | +655.48 | 11/14 |
| friction_mass_damping | klb5 | 386.18 +/- 222.62 | +1006.63 | 13/14 |
| friction_mass_damping | klb10 | 474.71 +/- 93.98 | +947.15 | 12/14 |
| friction_mass_damping | klb20 | 669.28 +/- 286.53 | +922.47 | 13/14 |
| friction_mass_damping | klb50 | 584.61 +/- 113.03 | +942.13 | 13/14 |
| friction_mass_damping | klb100 | 603.98 +/- 124.09 | +929.90 | 13/14 |
| mass_damping | klb0p5 | 368.94 +/- 53.52 | +1084.04 | 14/14 |
| mass_damping | klb1 | 444.66 +/- 154.30 | +1124.30 | 14/14 |
| mass_damping | klb2 | 821.98 +/- 156.40 | +831.17 | 12/14 |
| mass_damping | klb5 | 392.05 +/- 225.65 | +1139.56 | 14/14 |
| mass_damping | klb10 | 486.69 +/- 80.90 | +1082.51 | 14/14 |
| mass_damping | klb20 | 669.28 +/- 286.53 | +1086.02 | 14/14 |
| mass_damping | klb50 | 584.61 +/- 113.03 | +1092.70 | 14/14 |
| mass_damping | klb100 | 596.38 +/- 114.66 | +1045.36 | 14/14 |

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
