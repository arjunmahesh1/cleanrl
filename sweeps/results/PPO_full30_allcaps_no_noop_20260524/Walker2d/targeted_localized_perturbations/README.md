# Walker2d PPO Full 30-Seed No-Noop: targeted_localized_perturbations

Date: 2026-06-10

This folder packages the final pinned robustness evaluation for normalized PPO.

## Contents

- Raw metrics: `/home/users/am1015/cleanrl/sweeps/results/PPO_full30_allcaps_no_noop_20260524/Walker2d/targeted_localized_perturbations/raw_metrics`
- Aggregated outputs: `/home/users/am1015/cleanrl/sweeps/results/PPO_full30_allcaps_no_noop_20260524/Walker2d/targeted_localized_perturbations/outputs`
- Plots: `/home/users/am1015/cleanrl/sweeps/results/PPO_full30_allcaps_no_noop_20260524/Walker2d/targeted_localized_perturbations/plots`

## Evaluation protocol

- Models are compared on the same perturbation grid for the configured axes.
- Nominal reference within each axis is the `factor=1.0` point.
- Curves show mean return across seeds with `95% CI` shading.
- Robust gain is defined as `vanilla_drop - model_drop`; positive is better.
- Plot files are exported in both `PNG` and vector `PDF` format.

## Model labels

- `vanilla` -> Vanilla
- `a2p85` -> a2p85
- `a2p95` -> a2p95
- `a3p00` -> a3p00
- `a3p05` -> TV cap=3.05
- `a3p10` -> a3p10
- `a3p20` -> a3p20
- `a3p50` -> a3p50
- `a3p70` -> a3p70
- `a4p00` -> a4p00

## Nominal returns by axis

| Axis | Vanilla | a2p85 | a2p95 | a3p00 | TV cap=3.05 | a3p10 | a3p20 | a3p50 | a3p70 | a4p00 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| foot_left_actuator_gain | 3513.60 +/- 343.01 | 3258.71 +/- 378.06 | 3208.90 +/- 404.12 | 3508.89 +/- 293.95 | 3523.74 +/- 243.10 | 3753.62 +/- 408.31 | 3466.75 +/- 319.14 | 3780.53 +/- 343.67 | 3557.00 +/- 332.48 | 3850.41 +/- 330.96 |
| foot_left_damping | 3448.05 +/- 342.32 | 3258.71 +/- 378.06 | 3208.90 +/- 404.12 | 3508.89 +/- 293.95 | 3492.91 +/- 245.96 | 3753.62 +/- 408.31 | 3466.75 +/- 319.14 | 3780.53 +/- 343.67 | 3557.00 +/- 332.48 | 3850.41 +/- 330.96 |
| foot_left_friction | 3513.60 +/- 343.01 | 3258.71 +/- 378.06 | 3208.90 +/- 404.12 | 3508.89 +/- 293.95 | 3492.91 +/- 245.96 | 3753.62 +/- 408.31 | 3373.08 +/- 342.70 | 3780.53 +/- 343.67 | 3557.00 +/- 332.48 | 3850.41 +/- 330.96 |
| foot_left_mass | 3513.60 +/- 343.01 | 3155.56 +/- 368.31 | 3208.90 +/- 404.12 | 3508.89 +/- 293.95 | 3492.91 +/- 245.96 | 3753.62 +/- 408.31 | 3466.75 +/- 319.14 | 3780.53 +/- 343.67 | 3557.00 +/- 332.48 | 3850.41 +/- 330.96 |
| leg_left_actuator_gain | 3513.60 +/- 343.01 | 3258.71 +/- 378.06 | 3208.90 +/- 404.12 | 3508.89 +/- 293.95 | 3523.74 +/- 243.10 | 3742.31 +/- 412.06 | 3466.75 +/- 319.14 | 3780.53 +/- 343.67 | 3557.00 +/- 332.48 | 3850.41 +/- 330.96 |
| leg_left_damping | 3513.60 +/- 343.01 | 3258.71 +/- 378.06 | 3208.90 +/- 404.12 | 3500.66 +/- 288.98 | 3492.91 +/- 245.96 | 3753.62 +/- 408.31 | 3422.29 +/- 325.97 | 3780.53 +/- 343.67 | 3560.97 +/- 337.02 | 3850.41 +/- 330.96 |
| leg_left_mass | 3513.60 +/- 343.01 | 3258.71 +/- 378.06 | 3208.90 +/- 404.12 | 3508.89 +/- 293.95 | 3492.91 +/- 245.96 | 3753.62 +/- 408.31 | 3466.75 +/- 319.14 | 3780.53 +/- 343.67 | 3557.00 +/- 332.48 | 3850.41 +/- 330.96 |
| thigh_left_actuator_gain | 3513.60 +/- 343.01 | 3258.71 +/- 378.06 | 3208.90 +/- 404.12 | 3508.89 +/- 293.95 | 3492.91 +/- 245.96 | 3753.62 +/- 408.31 | 3466.75 +/- 319.14 | 3780.53 +/- 343.67 | 3557.00 +/- 332.48 | 3850.41 +/- 330.96 |
| thigh_left_damping | 3477.64 +/- 349.74 | 3258.71 +/- 378.06 | 3208.90 +/- 404.12 | 3500.66 +/- 288.98 | 3492.91 +/- 245.96 | 3741.59 +/- 412.33 | 3466.75 +/- 319.14 | 3780.53 +/- 343.67 | 3557.00 +/- 332.48 | 3850.41 +/- 330.96 |
| thigh_left_mass | 3412.09 +/- 348.17 | 3258.71 +/- 378.06 | 3208.90 +/- 404.12 | 3508.89 +/- 293.95 | 3492.91 +/- 245.96 | 3753.62 +/- 408.31 | 3466.75 +/- 319.14 | 3780.53 +/- 343.67 | 3557.00 +/- 332.48 | 3796.28 +/- 326.47 |

## Axis overview

| Axis | Model | Nominal return | Mean gain over perturbed scenarios | Positive gain scenarios |
| --- | --- | --- | --- | --- |
| foot_left_actuator_gain | Vanilla | 3513.60 +/- 343.01 | +0.00 | 0/0 |
| foot_left_actuator_gain | Vanilla | 3513.60 +/- 343.01 | +0.00 | 0/15 |
| foot_left_actuator_gain | a2p85 | 3258.71 +/- 378.06 | +102.48 | 10/15 |
| foot_left_actuator_gain | a2p95 | 3208.90 +/- 404.12 | +369.62 | 14/15 |
| foot_left_actuator_gain | a3p00 | 3508.89 +/- 293.95 | -20.67 | 5/15 |
| foot_left_actuator_gain | TV cap=3.05 | 3523.74 +/- 243.10 | -125.13 | 2/15 |
| foot_left_actuator_gain | a3p10 | 3753.62 +/- 408.31 | -216.05 | 0/15 |
| foot_left_actuator_gain | a3p20 | 3466.75 +/- 319.14 | +69.42 | 13/15 |
| foot_left_actuator_gain | a3p50 | 3780.53 +/- 343.67 | -319.84 | 2/15 |
| foot_left_actuator_gain | a3p70 | 3557.00 +/- 332.48 | +104.69 | 11/15 |
| foot_left_actuator_gain | a4p00 | 3850.41 +/- 330.96 | -303.91 | 0/15 |
| foot_left_damping | Vanilla | 3448.05 +/- 342.32 | +0.00 | 0/0 |
| foot_left_damping | Vanilla | 3448.05 +/- 342.32 | +0.00 | 0/15 |
| foot_left_damping | a2p85 | 3258.71 +/- 378.06 | -80.70 | 0/15 |
| foot_left_damping | a2p95 | 3208.90 +/- 404.12 | -18.53 | 5/15 |
| foot_left_damping | a3p00 | 3508.89 +/- 293.95 | -40.03 | 5/15 |
| foot_left_damping | TV cap=3.05 | 3492.91 +/- 245.96 | -8.50 | 6/15 |
| foot_left_damping | a3p10 | 3753.62 +/- 408.31 | -78.82 | 0/15 |
| foot_left_damping | a3p20 | 3466.75 +/- 319.14 | -50.38 | 3/15 |
| foot_left_damping | a3p50 | 3780.53 +/- 343.67 | -2.02 | 7/15 |
| foot_left_damping | a3p70 | 3557.00 +/- 332.48 | +21.66 | 11/15 |
| foot_left_damping | a4p00 | 3850.41 +/- 330.96 | -42.99 | 3/15 |
| foot_left_friction | Vanilla | 3513.60 +/- 343.01 | +0.00 | 0/0 |
| foot_left_friction | Vanilla | 3513.60 +/- 343.01 | +0.00 | 0/15 |
| foot_left_friction | a2p85 | 3258.71 +/- 378.06 | +15.69 | 8/15 |
| foot_left_friction | a2p95 | 3208.90 +/- 404.12 | +279.02 | 15/15 |
| foot_left_friction | a3p00 | 3508.89 +/- 293.95 | +102.62 | 9/15 |
| foot_left_friction | TV cap=3.05 | 3492.91 +/- 245.96 | -256.74 | 1/15 |
| foot_left_friction | a3p10 | 3753.62 +/- 408.31 | +0.14 | 8/15 |
| foot_left_friction | a3p20 | 3373.08 +/- 342.70 | +54.14 | 10/15 |
| foot_left_friction | a3p50 | 3780.53 +/- 343.67 | -56.18 | 7/15 |
| foot_left_friction | a3p70 | 3557.00 +/- 332.48 | +91.93 | 13/15 |
| foot_left_friction | a4p00 | 3850.41 +/- 330.96 | -85.79 | 6/15 |
| foot_left_mass | Vanilla | 3513.60 +/- 343.01 | +0.00 | 0/0 |
| foot_left_mass | Vanilla | 3513.60 +/- 343.01 | +0.00 | 0/14 |
| foot_left_mass | a2p85 | 3155.56 +/- 368.31 | +101.03 | 11/14 |
| foot_left_mass | a2p95 | 3208.90 +/- 404.12 | +240.98 | 14/14 |
| foot_left_mass | a3p00 | 3508.89 +/- 293.95 | +216.35 | 13/14 |
| foot_left_mass | TV cap=3.05 | 3492.91 +/- 245.96 | -72.56 | 8/14 |
| foot_left_mass | a3p10 | 3753.62 +/- 408.31 | -76.47 | 2/14 |
| foot_left_mass | a3p20 | 3466.75 +/- 319.14 | -78.70 | 4/14 |
| foot_left_mass | a3p50 | 3780.53 +/- 343.67 | -14.27 | 8/14 |
| foot_left_mass | a3p70 | 3557.00 +/- 332.48 | +84.82 | 9/14 |
| foot_left_mass | a4p00 | 3850.41 +/- 330.96 | -118.98 | 4/14 |
| leg_left_actuator_gain | Vanilla | 3513.60 +/- 343.01 | +0.00 | 0/0 |
| leg_left_actuator_gain | Vanilla | 3513.60 +/- 343.01 | +0.00 | 0/15 |
| leg_left_actuator_gain | a2p85 | 3258.71 +/- 378.06 | +23.95 | 7/15 |
| leg_left_actuator_gain | a2p95 | 3208.90 +/- 404.12 | +125.08 | 9/15 |
| leg_left_actuator_gain | a3p00 | 3508.89 +/- 293.95 | +42.09 | 8/15 |
| leg_left_actuator_gain | TV cap=3.05 | 3523.74 +/- 243.10 | -81.38 | 5/15 |
| leg_left_actuator_gain | a3p10 | 3742.31 +/- 412.06 | -191.57 | 3/15 |
| leg_left_actuator_gain | a3p20 | 3466.75 +/- 319.14 | +73.65 | 10/15 |
| leg_left_actuator_gain | a3p50 | 3780.53 +/- 343.67 | -183.45 | 1/15 |
| leg_left_actuator_gain | a3p70 | 3557.00 +/- 332.48 | +28.17 | 11/15 |
| leg_left_actuator_gain | a4p00 | 3850.41 +/- 330.96 | -165.65 | 4/15 |
| leg_left_damping | Vanilla | 3513.60 +/- 343.01 | +0.00 | 0/0 |
| leg_left_damping | Vanilla | 3513.60 +/- 343.01 | +0.00 | 0/15 |
| leg_left_damping | a2p85 | 3258.71 +/- 378.06 | -12.95 | 6/15 |
| leg_left_damping | a2p95 | 3208.90 +/- 404.12 | +59.19 | 14/15 |
| leg_left_damping | a3p00 | 3500.66 +/- 288.98 | +43.45 | 12/15 |
| leg_left_damping | TV cap=3.05 | 3492.91 +/- 245.96 | +95.06 | 15/15 |
| leg_left_damping | a3p10 | 3753.62 +/- 408.31 | -0.32 | 8/15 |
| leg_left_damping | a3p20 | 3422.29 +/- 325.97 | +36.12 | 13/15 |
| leg_left_damping | a3p50 | 3780.53 +/- 343.67 | +74.85 | 14/15 |
| leg_left_damping | a3p70 | 3560.97 +/- 337.02 | +61.61 | 13/15 |
| leg_left_damping | a4p00 | 3850.41 +/- 330.96 | +20.16 | 10/15 |
| leg_left_mass | Vanilla | 3513.60 +/- 343.01 | +0.00 | 0/0 |
| leg_left_mass | Vanilla | 3513.60 +/- 343.01 | +0.00 | 0/14 |
| leg_left_mass | a2p85 | 3258.71 +/- 378.06 | -19.71 | 4/14 |
| leg_left_mass | a2p95 | 3208.90 +/- 404.12 | +186.46 | 14/14 |
| leg_left_mass | a3p00 | 3508.89 +/- 293.95 | +199.02 | 14/14 |
| leg_left_mass | TV cap=3.05 | 3492.91 +/- 245.96 | -77.86 | 3/14 |
| leg_left_mass | a3p10 | 3753.62 +/- 408.31 | +15.79 | 9/14 |
| leg_left_mass | a3p20 | 3466.75 +/- 319.14 | +50.81 | 11/14 |
| leg_left_mass | a3p50 | 3780.53 +/- 343.67 | -167.15 | 4/14 |
| leg_left_mass | a3p70 | 3557.00 +/- 332.48 | +84.30 | 10/14 |
| leg_left_mass | a4p00 | 3850.41 +/- 330.96 | -89.40 | 4/14 |
| thigh_left_actuator_gain | Vanilla | 3513.60 +/- 343.01 | +0.00 | 0/0 |
| thigh_left_actuator_gain | Vanilla | 3513.60 +/- 343.01 | +0.00 | 0/15 |
| thigh_left_actuator_gain | a2p85 | 3258.71 +/- 378.06 | -241.61 | 2/15 |
| thigh_left_actuator_gain | a2p95 | 3208.90 +/- 404.12 | -154.78 | 6/15 |
| thigh_left_actuator_gain | a3p00 | 3508.89 +/- 293.95 | -79.44 | 7/15 |
| thigh_left_actuator_gain | TV cap=3.05 | 3492.91 +/- 245.96 | -298.59 | 0/15 |
| thigh_left_actuator_gain | a3p10 | 3753.62 +/- 408.31 | -488.58 | 2/15 |
| thigh_left_actuator_gain | a3p20 | 3466.75 +/- 319.14 | -151.18 | 6/15 |
| thigh_left_actuator_gain | a3p50 | 3780.53 +/- 343.67 | -308.16 | 3/15 |
| thigh_left_actuator_gain | a3p70 | 3557.00 +/- 332.48 | -134.51 | 6/15 |
| thigh_left_actuator_gain | a4p00 | 3850.41 +/- 330.96 | -504.63 | 1/15 |
| thigh_left_damping | Vanilla | 3477.64 +/- 349.74 | +0.00 | 0/0 |
| thigh_left_damping | Vanilla | 3477.64 +/- 349.74 | +0.00 | 0/15 |
| thigh_left_damping | a2p85 | 3258.71 +/- 378.06 | -52.12 | 0/15 |
| thigh_left_damping | a2p95 | 3208.90 +/- 404.12 | +31.31 | 12/15 |
| thigh_left_damping | a3p00 | 3500.66 +/- 288.98 | +20.44 | 10/15 |
| thigh_left_damping | TV cap=3.05 | 3492.91 +/- 245.96 | +58.78 | 15/15 |
| thigh_left_damping | a3p10 | 3741.59 +/- 412.33 | -28.57 | 4/15 |
| thigh_left_damping | a3p20 | 3466.75 +/- 319.14 | -4.81 | 7/15 |
| thigh_left_damping | a3p50 | 3780.53 +/- 343.67 | +35.72 | 12/15 |
| thigh_left_damping | a3p70 | 3557.00 +/- 332.48 | +60.14 | 14/15 |
| thigh_left_damping | a4p00 | 3850.41 +/- 330.96 | +9.85 | 8/15 |
| thigh_left_mass | Vanilla | 3412.09 +/- 348.17 | +0.00 | 0/0 |
| thigh_left_mass | Vanilla | 3412.09 +/- 348.17 | +0.00 | 0/14 |
| thigh_left_mass | a2p85 | 3258.71 +/- 378.06 | -336.24 | 4/14 |
| thigh_left_mass | a2p95 | 3208.90 +/- 404.12 | -16.62 | 5/14 |
| thigh_left_mass | a3p00 | 3508.89 +/- 293.95 | -87.90 | 3/14 |
| thigh_left_mass | TV cap=3.05 | 3492.91 +/- 245.96 | -242.64 | 1/14 |
| thigh_left_mass | a3p10 | 3753.62 +/- 408.31 | -291.54 | 1/14 |
| thigh_left_mass | a3p20 | 3466.75 +/- 319.14 | -110.69 | 2/14 |
| thigh_left_mass | a3p50 | 3780.53 +/- 343.67 | -232.93 | 3/14 |
| thigh_left_mass | a3p70 | 3557.00 +/- 332.48 | -17.16 | 5/14 |
| thigh_left_mass | a4p00 | 3796.28 +/- 326.47 | -274.39 | 2/14 |

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
- `plots/with_variance/foot_left_actuator_gain_return_curve.png`
- `plots/with_variance/foot_left_actuator_gain_return_curve.pdf`
- `plots/with_variance/foot_left_actuator_gain_gain_curve.png`
- `plots/with_variance/foot_left_actuator_gain_gain_curve.pdf`
- `plots/without_variance/foot_left_actuator_gain_return_curve.png`
- `plots/without_variance/foot_left_actuator_gain_return_curve.pdf`
- `plots/without_variance/foot_left_actuator_gain_gain_curve.png`
- `plots/without_variance/foot_left_actuator_gain_gain_curve.pdf`
- `plots/with_variance/foot_left_damping_return_curve.png`
- `plots/with_variance/foot_left_damping_return_curve.pdf`
- `plots/with_variance/foot_left_damping_gain_curve.png`
- `plots/with_variance/foot_left_damping_gain_curve.pdf`
- `plots/without_variance/foot_left_damping_return_curve.png`
- `plots/without_variance/foot_left_damping_return_curve.pdf`
- `plots/without_variance/foot_left_damping_gain_curve.png`
- `plots/without_variance/foot_left_damping_gain_curve.pdf`
- `plots/with_variance/foot_left_friction_return_curve.png`
- `plots/with_variance/foot_left_friction_return_curve.pdf`
- `plots/with_variance/foot_left_friction_gain_curve.png`
- `plots/with_variance/foot_left_friction_gain_curve.pdf`
- `plots/without_variance/foot_left_friction_return_curve.png`
- `plots/without_variance/foot_left_friction_return_curve.pdf`
- `plots/without_variance/foot_left_friction_gain_curve.png`
- `plots/without_variance/foot_left_friction_gain_curve.pdf`
- `plots/with_variance/foot_left_mass_return_curve.png`
- `plots/with_variance/foot_left_mass_return_curve.pdf`
- `plots/with_variance/foot_left_mass_gain_curve.png`
- `plots/with_variance/foot_left_mass_gain_curve.pdf`
- `plots/without_variance/foot_left_mass_return_curve.png`
- `plots/without_variance/foot_left_mass_return_curve.pdf`
- `plots/without_variance/foot_left_mass_gain_curve.png`
- `plots/without_variance/foot_left_mass_gain_curve.pdf`
- `plots/with_variance/leg_left_actuator_gain_return_curve.png`
- `plots/with_variance/leg_left_actuator_gain_return_curve.pdf`
- `plots/with_variance/leg_left_actuator_gain_gain_curve.png`
- `plots/with_variance/leg_left_actuator_gain_gain_curve.pdf`
- `plots/without_variance/leg_left_actuator_gain_return_curve.png`
- `plots/without_variance/leg_left_actuator_gain_return_curve.pdf`
- `plots/without_variance/leg_left_actuator_gain_gain_curve.png`
- `plots/without_variance/leg_left_actuator_gain_gain_curve.pdf`
- `plots/with_variance/leg_left_damping_return_curve.png`
- `plots/with_variance/leg_left_damping_return_curve.pdf`
- `plots/with_variance/leg_left_damping_gain_curve.png`
- `plots/with_variance/leg_left_damping_gain_curve.pdf`
- `plots/without_variance/leg_left_damping_return_curve.png`
- `plots/without_variance/leg_left_damping_return_curve.pdf`
- `plots/without_variance/leg_left_damping_gain_curve.png`
- `plots/without_variance/leg_left_damping_gain_curve.pdf`
- `plots/with_variance/leg_left_mass_return_curve.png`
- `plots/with_variance/leg_left_mass_return_curve.pdf`
- `plots/with_variance/leg_left_mass_gain_curve.png`
- `plots/with_variance/leg_left_mass_gain_curve.pdf`
- `plots/without_variance/leg_left_mass_return_curve.png`
- `plots/without_variance/leg_left_mass_return_curve.pdf`
- `plots/without_variance/leg_left_mass_gain_curve.png`
- `plots/without_variance/leg_left_mass_gain_curve.pdf`
- `plots/with_variance/thigh_left_actuator_gain_return_curve.png`
- `plots/with_variance/thigh_left_actuator_gain_return_curve.pdf`
- `plots/with_variance/thigh_left_actuator_gain_gain_curve.png`
- `plots/with_variance/thigh_left_actuator_gain_gain_curve.pdf`
- `plots/without_variance/thigh_left_actuator_gain_return_curve.png`
- `plots/without_variance/thigh_left_actuator_gain_return_curve.pdf`
- `plots/without_variance/thigh_left_actuator_gain_gain_curve.png`
- `plots/without_variance/thigh_left_actuator_gain_gain_curve.pdf`
- `plots/with_variance/thigh_left_damping_return_curve.png`
- `plots/with_variance/thigh_left_damping_return_curve.pdf`
- `plots/with_variance/thigh_left_damping_gain_curve.png`
- `plots/with_variance/thigh_left_damping_gain_curve.pdf`
- `plots/without_variance/thigh_left_damping_return_curve.png`
- `plots/without_variance/thigh_left_damping_return_curve.pdf`
- `plots/without_variance/thigh_left_damping_gain_curve.png`
- `plots/without_variance/thigh_left_damping_gain_curve.pdf`
- `plots/with_variance/thigh_left_mass_return_curve.png`
- `plots/with_variance/thigh_left_mass_return_curve.pdf`
- `plots/with_variance/thigh_left_mass_gain_curve.png`
- `plots/with_variance/thigh_left_mass_gain_curve.pdf`
- `plots/without_variance/thigh_left_mass_return_curve.png`
- `plots/without_variance/thigh_left_mass_return_curve.pdf`
- `plots/without_variance/thigh_left_mass_gain_curve.png`
- `plots/without_variance/thigh_left_mass_gain_curve.pdf`

## Output CSV files

- `outputs/eval_metrics_final.csv`: latest merged per-seed eval rows.
- `outputs/summary_by_scenario.csv`: per-model, per-axis, per-factor aggregate return table.
- `outputs/drop_summary.csv`: nominal-minus-perturbed drop table.
- `outputs/gain_summary.csv`: paired robust-gain table against vanilla.
- `outputs/axis_overview.csv`: compact axis-level overview.
- `outputs/curve_points.csv`: same data used for return plots.
- `outputs/gain_curve_points.csv`: same data used for gain plots.
