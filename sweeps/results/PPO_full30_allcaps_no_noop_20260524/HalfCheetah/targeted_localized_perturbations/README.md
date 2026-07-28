# HalfCheetah PPO Full 30-Seed No-Noop: targeted_localized_perturbations

Date: 2026-06-10

This folder packages the final pinned robustness evaluation for normalized PPO.

## Contents

- Raw metrics: `/home/users/am1015/cleanrl/sweeps/results/PPO_full30_allcaps_no_noop_20260524/HalfCheetah/targeted_localized_perturbations/raw_metrics`
- Aggregated outputs: `/home/users/am1015/cleanrl/sweeps/results/PPO_full30_allcaps_no_noop_20260524/HalfCheetah/targeted_localized_perturbations/outputs`
- Plots: `/home/users/am1015/cleanrl/sweeps/results/PPO_full30_allcaps_no_noop_20260524/HalfCheetah/targeted_localized_perturbations/plots`

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
| bfoot_actuator_gain | 2654.39 +/- 476.99 | 2055.37 +/- 345.30 | 2219.98 +/- 388.24 | 2775.56 +/- 521.72 | 2079.96 +/- 339.43 | 2416.59 +/- 436.17 | 2550.82 +/- 499.29 | 2998.65 +/- 524.52 | 2494.97 +/- 457.26 | 2562.31 +/- 487.98 | 2832.98 +/- 543.70 | 2367.29 +/- 466.77 | 2765.16 +/- 525.04 | 2561.30 +/- 482.02 |
| bfoot_damping | 2644.08 +/- 481.35 | 2055.37 +/- 345.30 | 2221.58 +/- 387.95 | 2795.67 +/- 524.88 | 2079.96 +/- 339.43 | 2408.03 +/- 432.08 | 2550.82 +/- 499.29 | 2977.88 +/- 525.23 | 2494.97 +/- 457.26 | 2562.31 +/- 487.98 | 2832.98 +/- 543.70 | 2367.29 +/- 466.77 | 2765.16 +/- 525.04 | 2561.30 +/- 482.02 |
| bfoot_friction | 2644.08 +/- 481.35 | 2055.37 +/- 345.30 | 2221.58 +/- 387.95 | 2795.67 +/- 524.88 | 2079.96 +/- 339.43 | 2416.59 +/- 436.17 | 2550.82 +/- 499.29 | 3000.79 +/- 523.58 | 2494.97 +/- 457.26 | 2572.60 +/- 498.71 | 2832.98 +/- 543.70 | 2367.29 +/- 466.77 | 2765.16 +/- 525.04 | 2561.30 +/- 482.02 |
| bfoot_mass | 2644.08 +/- 481.35 | 2055.37 +/- 345.30 | 2221.58 +/- 387.95 | 2795.67 +/- 524.88 | 2079.96 +/- 339.43 | 2416.59 +/- 436.17 | 2550.82 +/- 499.29 | 2977.88 +/- 525.23 | 2494.97 +/- 457.26 | 2562.31 +/- 487.98 | 2843.63 +/- 549.52 | 2367.29 +/- 466.77 | 2765.16 +/- 525.04 | 2561.30 +/- 482.02 |
| bshin_actuator_gain | 2644.08 +/- 481.35 | 2053.56 +/- 338.67 | 2221.58 +/- 387.95 | 2795.67 +/- 524.88 | 2079.96 +/- 339.43 | 2416.59 +/- 436.17 | 2550.82 +/- 499.29 | 2977.88 +/- 525.23 | 2494.97 +/- 457.26 | 2562.31 +/- 487.98 | 2832.98 +/- 543.70 | 2354.02 +/- 456.27 | 2765.16 +/- 525.04 | 2534.56 +/- 474.91 |
| bshin_damping | 2654.39 +/- 476.99 | 2055.37 +/- 345.30 | 2221.58 +/- 387.95 | 2795.67 +/- 524.88 | 2079.96 +/- 339.43 | 2416.59 +/- 436.17 | 2550.82 +/- 499.29 | 2977.88 +/- 525.23 | 2494.97 +/- 457.26 | 2562.31 +/- 487.98 | 2832.98 +/- 543.70 | 2367.29 +/- 466.77 | 2765.16 +/- 525.04 | 2561.30 +/- 482.02 |
| bshin_friction | 2644.08 +/- 481.35 | 2055.37 +/- 345.30 | 2219.98 +/- 388.24 | 2795.67 +/- 524.88 | 2089.20 +/- 351.65 | 2416.59 +/- 436.17 | 2550.82 +/- 499.29 | 2977.88 +/- 525.23 | 2494.97 +/- 457.26 | 2562.31 +/- 487.98 | 2843.63 +/- 549.52 | 2367.29 +/- 466.77 | 2765.16 +/- 525.04 | 2561.30 +/- 482.02 |
| bshin_mass | 2644.08 +/- 481.35 | 2055.37 +/- 345.30 | 2221.58 +/- 387.95 | 2795.67 +/- 524.88 | 2079.96 +/- 339.43 | 2416.59 +/- 436.17 | 2550.82 +/- 499.29 | 2977.88 +/- 525.23 | 2494.97 +/- 457.26 | 2562.31 +/- 487.98 | 2832.98 +/- 543.70 | 2367.29 +/- 466.77 | 2765.16 +/- 525.04 | 2551.58 +/- 478.63 |
| bthigh_actuator_gain | 2644.08 +/- 481.35 | 2055.37 +/- 345.30 | 2221.58 +/- 387.95 | 2795.67 +/- 524.88 | 2079.96 +/- 339.43 | 2410.19 +/- 428.31 | 2550.82 +/- 499.29 | 2977.88 +/- 525.23 | 2494.97 +/- 457.26 | 2562.31 +/- 487.98 | 2832.98 +/- 543.70 | 2367.29 +/- 466.77 | 2765.16 +/- 525.04 | 2561.30 +/- 482.02 |
| bthigh_damping | 2644.08 +/- 481.35 | 2055.37 +/- 345.30 | 2221.58 +/- 387.95 | 2795.67 +/- 524.88 | 2079.96 +/- 339.43 | 2416.59 +/- 436.17 | 2550.82 +/- 499.29 | 2977.88 +/- 525.23 | 2499.53 +/- 460.40 | 2562.31 +/- 487.98 | 2832.98 +/- 543.70 | 2367.29 +/- 466.77 | 2765.16 +/- 525.04 | 2561.30 +/- 482.02 |
| bthigh_friction | 2644.08 +/- 481.35 | 2055.37 +/- 345.30 | 2221.58 +/- 387.95 | 2795.67 +/- 524.88 | 2079.96 +/- 339.43 | 2416.59 +/- 436.17 | 2550.82 +/- 499.29 | 2977.88 +/- 525.23 | 2494.97 +/- 457.26 | 2562.31 +/- 487.98 | 2832.98 +/- 543.70 | 2354.02 +/- 456.27 | 2765.16 +/- 525.04 | 2561.30 +/- 482.02 |
| bthigh_mass | 2654.39 +/- 476.99 | 2055.37 +/- 345.30 | 2219.94 +/- 396.47 | 2795.67 +/- 524.88 | 2079.96 +/- 339.43 | 2416.59 +/- 436.17 | 2550.82 +/- 499.29 | 2977.88 +/- 525.23 | 2494.97 +/- 457.26 | 2562.31 +/- 487.98 | 2832.98 +/- 543.70 | 2367.29 +/- 466.77 | 2765.16 +/- 525.04 | 2561.30 +/- 482.02 |
| ffoot_actuator_gain | 2644.08 +/- 481.35 | 2055.37 +/- 345.30 | 2219.98 +/- 388.24 | 2795.67 +/- 524.88 | 2079.96 +/- 339.43 | 2416.59 +/- 436.17 | 2550.82 +/- 499.29 | 2977.88 +/- 525.23 | 2494.97 +/- 457.26 | 2562.31 +/- 487.98 | 2832.98 +/- 543.70 | 2367.29 +/- 466.77 | 2765.16 +/- 525.04 | 2561.30 +/- 482.02 |
| ffoot_damping | 2644.08 +/- 481.35 | 2055.37 +/- 345.30 | 2219.94 +/- 396.47 | 2795.67 +/- 524.88 | 2079.96 +/- 339.43 | 2416.59 +/- 436.17 | 2550.82 +/- 499.29 | 2977.88 +/- 525.23 | 2494.97 +/- 457.26 | 2562.31 +/- 487.98 | 2837.85 +/- 548.01 | 2367.29 +/- 466.77 | 2770.06 +/- 529.61 | 2561.30 +/- 482.02 |
| ffoot_friction | 2644.08 +/- 481.35 | 2055.37 +/- 345.30 | 2221.58 +/- 387.95 | 2793.03 +/- 525.29 | 2079.96 +/- 339.43 | 2416.59 +/- 436.17 | 2550.82 +/- 499.29 | 2977.88 +/- 525.23 | 2494.97 +/- 457.26 | 2562.31 +/- 487.98 | 2843.63 +/- 549.52 | 2349.98 +/- 456.99 | 2765.16 +/- 525.04 | 2551.58 +/- 478.63 |
| ffoot_mass | 2644.08 +/- 481.35 | 2055.37 +/- 345.30 | 2221.58 +/- 387.95 | 2795.67 +/- 524.88 | 2079.96 +/- 339.43 | 2416.59 +/- 436.17 | 2550.82 +/- 499.29 | 2977.88 +/- 525.23 | 2494.97 +/- 457.26 | 2554.09 +/- 481.89 | 2832.98 +/- 543.70 | 2367.29 +/- 466.77 | 2765.16 +/- 525.04 | 2561.30 +/- 482.02 |
| fshin_actuator_gain | 2644.08 +/- 481.35 | 2055.37 +/- 345.30 | 2221.58 +/- 387.95 | 2795.67 +/- 524.88 | 2079.96 +/- 339.43 | 2416.59 +/- 436.17 | 2550.82 +/- 499.29 | 2977.88 +/- 525.23 | 2494.97 +/- 457.26 | 2562.31 +/- 487.98 | 2832.98 +/- 543.70 | 2367.29 +/- 466.77 | 2765.16 +/- 525.04 | 2561.30 +/- 482.02 |
| fshin_damping | 2644.08 +/- 481.35 | 2055.37 +/- 345.30 | 2221.58 +/- 387.95 | 2775.56 +/- 521.72 | 2079.96 +/- 339.43 | 2416.59 +/- 436.17 | 2550.82 +/- 499.29 | 2977.88 +/- 525.23 | 2494.97 +/- 457.26 | 2562.31 +/- 487.98 | 2832.98 +/- 543.70 | 2367.29 +/- 466.77 | 2765.16 +/- 525.04 | 2561.30 +/- 482.02 |
| fshin_friction | 2644.08 +/- 481.35 | 2055.37 +/- 345.30 | 2219.94 +/- 396.47 | 2795.67 +/- 524.88 | 2079.96 +/- 339.43 | 2416.59 +/- 436.17 | 2550.82 +/- 499.29 | 2977.88 +/- 525.23 | 2494.97 +/- 457.26 | 2562.31 +/- 487.98 | 2832.98 +/- 543.70 | 2367.29 +/- 466.77 | 2765.16 +/- 525.04 | 2561.30 +/- 482.02 |
| fshin_mass | 2644.08 +/- 481.35 | 2055.37 +/- 345.30 | 2221.58 +/- 387.95 | 2795.67 +/- 524.88 | 2079.96 +/- 339.43 | 2416.59 +/- 436.17 | 2550.82 +/- 499.29 | 3000.79 +/- 523.58 | 2491.22 +/- 457.96 | 2572.60 +/- 498.71 | 2843.63 +/- 549.52 | 2367.29 +/- 466.77 | 2765.16 +/- 525.04 | 2561.30 +/- 482.02 |
| fthigh_actuator_gain | 2654.39 +/- 476.99 | 2055.37 +/- 345.30 | 2219.94 +/- 396.47 | 2775.56 +/- 521.72 | 2079.96 +/- 339.43 | 2416.59 +/- 436.17 | 2550.82 +/- 499.29 | 3000.79 +/- 523.58 | 2494.97 +/- 457.26 | 2562.31 +/- 487.98 | 2832.98 +/- 543.70 | 2367.29 +/- 466.77 | 2765.16 +/- 525.04 | 2561.30 +/- 482.02 |
| fthigh_damping | 2644.08 +/- 481.35 | 2055.37 +/- 345.30 | 2221.58 +/- 387.95 | 2795.67 +/- 524.88 | 2079.96 +/- 339.43 | 2416.59 +/- 436.17 | 2550.82 +/- 499.29 | 2977.88 +/- 525.23 | 2494.97 +/- 457.26 | 2562.31 +/- 487.98 | 2832.98 +/- 543.70 | 2367.29 +/- 466.77 | 2765.16 +/- 525.04 | 2561.30 +/- 482.02 |
| fthigh_friction | 2644.08 +/- 481.35 | 2055.37 +/- 345.30 | 2221.54 +/- 396.19 | 2795.67 +/- 524.88 | 2079.96 +/- 339.43 | 2416.59 +/- 436.17 | 2550.82 +/- 499.29 | 2977.88 +/- 525.23 | 2494.97 +/- 457.26 | 2562.31 +/- 487.98 | 2832.98 +/- 543.70 | 2367.29 +/- 466.77 | 2765.16 +/- 525.04 | 2561.30 +/- 482.02 |
| fthigh_mass | 2644.08 +/- 481.35 | 2055.37 +/- 345.30 | 2221.58 +/- 387.95 | 2793.03 +/- 525.29 | 2076.53 +/- 339.62 | 2416.59 +/- 436.17 | 2550.82 +/- 499.29 | 2977.88 +/- 525.23 | 2494.97 +/- 457.26 | 2562.31 +/- 487.98 | 2832.98 +/- 543.70 | 2367.29 +/- 466.77 | 2778.38 +/- 527.31 | 2561.30 +/- 482.02 |

## Axis overview

| Axis | Model | Nominal return | Mean gain over perturbed scenarios | Positive gain scenarios |
| --- | --- | --- | --- | --- |
| bfoot_actuator_gain | Vanilla | 2654.39 +/- 476.99 | +0.00 | 0/0 |
| bfoot_actuator_gain | Vanilla | 2654.39 +/- 476.99 | +0.00 | 0/15 |
| bfoot_actuator_gain | a2p20 | 2055.37 +/- 345.30 | +138.61 | 14/15 |
| bfoot_actuator_gain | a2p40 | 2219.98 +/- 388.24 | +136.81 | 14/15 |
| bfoot_actuator_gain | a2p55 | 2775.56 +/- 521.72 | +7.78 | 8/15 |
| bfoot_actuator_gain | a2p65 | 2079.96 +/- 339.43 | +144.98 | 13/15 |
| bfoot_actuator_gain | a2p70 | 2416.59 +/- 436.17 | +98.75 | 14/15 |
| bfoot_actuator_gain | a2p75 | 2550.82 +/- 499.29 | +95.30 | 13/15 |
| bfoot_actuator_gain | a2p80 | 2998.65 +/- 524.52 | -74.57 | 3/15 |
| bfoot_actuator_gain | a3p00 | 2494.97 +/- 457.26 | +90.37 | 11/15 |
| bfoot_actuator_gain | TV cap=3.05 | 2562.31 +/- 487.98 | +82.44 | 12/15 |
| bfoot_actuator_gain | a3p10 | 2832.98 +/- 543.70 | -47.94 | 5/15 |
| bfoot_actuator_gain | a3p20 | 2367.29 +/- 466.77 | +152.84 | 14/15 |
| bfoot_actuator_gain | a3p40 | 2765.16 +/- 525.04 | +86.28 | 12/15 |
| bfoot_actuator_gain | a3p70 | 2561.30 +/- 482.02 | -8.45 | 6/15 |
| bfoot_damping | Vanilla | 2644.08 +/- 481.35 | +0.00 | 0/0 |
| bfoot_damping | Vanilla | 2644.08 +/- 481.35 | +0.00 | 0/15 |
| bfoot_damping | a2p20 | 2055.37 +/- 345.30 | +74.86 | 14/15 |
| bfoot_damping | a2p40 | 2221.58 +/- 387.95 | +75.52 | 13/15 |
| bfoot_damping | a2p55 | 2795.67 +/- 524.88 | -79.96 | 0/15 |
| bfoot_damping | a2p65 | 2079.96 +/- 339.43 | +84.51 | 14/15 |
| bfoot_damping | a2p70 | 2408.03 +/- 432.08 | +38.66 | 12/15 |
| bfoot_damping | a2p75 | 2550.82 +/- 499.29 | +24.57 | 11/15 |
| bfoot_damping | a2p80 | 2977.88 +/- 525.23 | -58.71 | 2/15 |
| bfoot_damping | a3p00 | 2494.97 +/- 457.26 | -10.34 | 5/15 |
| bfoot_damping | TV cap=3.05 | 2562.31 +/- 487.98 | +27.90 | 11/15 |
| bfoot_damping | a3p10 | 2832.98 +/- 543.70 | -111.80 | 2/15 |
| bfoot_damping | a3p20 | 2367.29 +/- 466.77 | +40.98 | 12/15 |
| bfoot_damping | a3p40 | 2765.16 +/- 525.04 | -18.66 | 4/15 |
| bfoot_damping | a3p70 | 2561.30 +/- 482.02 | -6.83 | 7/15 |
| bfoot_friction | Vanilla | 2644.08 +/- 481.35 | +0.00 | 0/0 |
| bfoot_friction | Vanilla | 2644.08 +/- 481.35 | +0.00 | 0/15 |
| bfoot_friction | a2p20 | 2055.37 +/- 345.30 | +0.00 | 0/15 |
| bfoot_friction | a2p40 | 2221.58 +/- 387.95 | -0.39 | 0/15 |
| bfoot_friction | a2p55 | 2795.67 +/- 524.88 | +0.00 | 0/15 |
| bfoot_friction | a2p65 | 2079.96 +/- 339.43 | +3.82 | 6/15 |
| bfoot_friction | a2p70 | 2416.59 +/- 436.17 | +0.00 | 0/15 |
| bfoot_friction | a2p75 | 2550.82 +/- 499.29 | -7.86 | 0/15 |
| bfoot_friction | a2p80 | 3000.79 +/- 523.58 | -19.72 | 0/15 |
| bfoot_friction | a3p00 | 2494.97 +/- 457.26 | -0.45 | 1/15 |
| bfoot_friction | TV cap=3.05 | 2572.60 +/- 498.71 | -8.88 | 1/15 |
| bfoot_friction | a3p10 | 2832.98 +/- 543.70 | +0.00 | 0/15 |
| bfoot_friction | a3p20 | 2367.29 +/- 466.77 | +0.00 | 0/15 |
| bfoot_friction | a3p40 | 2765.16 +/- 525.04 | +1.09 | 6/15 |
| bfoot_friction | a3p70 | 2561.30 +/- 482.02 | -5.70 | 0/15 |
| bfoot_mass | Vanilla | 2644.08 +/- 481.35 | +0.00 | 0/0 |
| bfoot_mass | Vanilla | 2644.08 +/- 481.35 | +0.00 | 0/14 |
| bfoot_mass | a2p20 | 2055.37 +/- 345.30 | +0.00 | 0/14 |
| bfoot_mass | a2p40 | 2221.58 +/- 387.95 | +0.00 | 0/14 |
| bfoot_mass | a2p55 | 2795.67 +/- 524.88 | +0.00 | 0/14 |
| bfoot_mass | a2p65 | 2079.96 +/- 339.43 | +0.00 | 0/14 |
| bfoot_mass | a2p70 | 2416.59 +/- 436.17 | +0.00 | 0/14 |
| bfoot_mass | a2p75 | 2550.82 +/- 499.29 | +0.00 | 0/14 |
| bfoot_mass | a2p80 | 2977.88 +/- 525.23 | +0.00 | 0/14 |
| bfoot_mass | a3p00 | 2494.97 +/- 457.26 | -0.58 | 0/14 |
| bfoot_mass | TV cap=3.05 | 2562.31 +/- 487.98 | +0.00 | 0/14 |
| bfoot_mass | a3p10 | 2843.63 +/- 549.52 | -7.29 | 0/14 |
| bfoot_mass | a3p20 | 2367.29 +/- 466.77 | +0.00 | 0/14 |
| bfoot_mass | a3p40 | 2765.16 +/- 525.04 | +0.00 | 0/14 |
| bfoot_mass | a3p70 | 2561.30 +/- 482.02 | +0.00 | 0/14 |
| bshin_actuator_gain | Vanilla | 2644.08 +/- 481.35 | +0.00 | 0/0 |
| bshin_actuator_gain | Vanilla | 2644.08 +/- 481.35 | +0.00 | 0/15 |
| bshin_actuator_gain | a2p20 | 2053.56 +/- 338.67 | +288.30 | 15/15 |
| bshin_actuator_gain | a2p40 | 2221.58 +/- 387.95 | +223.64 | 15/15 |
| bshin_actuator_gain | a2p55 | 2795.67 +/- 524.88 | -35.35 | 9/15 |
| bshin_actuator_gain | a2p65 | 2079.96 +/- 339.43 | +223.04 | 15/15 |
| bshin_actuator_gain | a2p70 | 2416.59 +/- 436.17 | +87.25 | 13/15 |
| bshin_actuator_gain | a2p75 | 2550.82 +/- 499.29 | +107.21 | 13/15 |
| bshin_actuator_gain | a2p80 | 2977.88 +/- 525.23 | -65.42 | 5/15 |
| bshin_actuator_gain | a3p00 | 2494.97 +/- 457.26 | +46.17 | 8/15 |
| bshin_actuator_gain | TV cap=3.05 | 2562.31 +/- 487.98 | +172.59 | 15/15 |
| bshin_actuator_gain | a3p10 | 2832.98 +/- 543.70 | -108.97 | 6/15 |
| bshin_actuator_gain | a3p20 | 2354.02 +/- 456.27 | +142.29 | 14/15 |
| bshin_actuator_gain | a3p40 | 2765.16 +/- 525.04 | +121.40 | 11/15 |
| bshin_actuator_gain | a3p70 | 2534.56 +/- 474.91 | +50.76 | 11/15 |
| bshin_damping | Vanilla | 2654.39 +/- 476.99 | +0.00 | 0/0 |
| bshin_damping | Vanilla | 2654.39 +/- 476.99 | +0.00 | 0/15 |
| bshin_damping | a2p20 | 2055.37 +/- 345.30 | +413.14 | 15/15 |
| bshin_damping | a2p40 | 2221.58 +/- 387.95 | +304.17 | 15/15 |
| bshin_damping | a2p55 | 2795.67 +/- 524.88 | +4.12 | 6/15 |
| bshin_damping | a2p65 | 2079.96 +/- 339.43 | +358.52 | 15/15 |
| bshin_damping | a2p70 | 2416.59 +/- 436.17 | +209.78 | 15/15 |
| bshin_damping | a2p75 | 2550.82 +/- 499.29 | +156.41 | 13/15 |
| bshin_damping | a2p80 | 2977.88 +/- 525.23 | +87.59 | 12/15 |
| bshin_damping | a3p00 | 2494.97 +/- 457.26 | +90.58 | 10/15 |
| bshin_damping | TV cap=3.05 | 2562.31 +/- 487.98 | +260.00 | 15/15 |
| bshin_damping | a3p10 | 2832.98 +/- 543.70 | -40.68 | 6/15 |
| bshin_damping | a3p20 | 2367.29 +/- 466.77 | +290.16 | 15/15 |
| bshin_damping | a3p40 | 2765.16 +/- 525.04 | +65.70 | 13/15 |
| bshin_damping | a3p70 | 2561.30 +/- 482.02 | +135.42 | 14/15 |
| bshin_friction | Vanilla | 2644.08 +/- 481.35 | +0.00 | 0/0 |
| bshin_friction | Vanilla | 2644.08 +/- 481.35 | +0.00 | 0/15 |
| bshin_friction | a2p20 | 2055.37 +/- 345.30 | +0.51 | 2/15 |
| bshin_friction | a2p40 | 2219.98 +/- 388.24 | +1.23 | 9/15 |
| bshin_friction | a2p55 | 2795.67 +/- 524.88 | +0.00 | 0/15 |
| bshin_friction | a2p65 | 2089.20 +/- 351.65 | -8.11 | 0/15 |
| bshin_friction | a2p70 | 2416.59 +/- 436.17 | -3.79 | 0/15 |
| bshin_friction | a2p75 | 2550.82 +/- 499.29 | -9.02 | 0/15 |
| bshin_friction | a2p80 | 2977.88 +/- 525.23 | +1.38 | 1/15 |
| bshin_friction | a3p00 | 2494.97 +/- 457.26 | +0.00 | 0/15 |
| bshin_friction | TV cap=3.05 | 2562.31 +/- 487.98 | +2.09 | 3/15 |
| bshin_friction | a3p10 | 2843.63 +/- 549.52 | -8.73 | 0/15 |
| bshin_friction | a3p20 | 2367.29 +/- 466.77 | +0.00 | 0/15 |
| bshin_friction | a3p40 | 2765.16 +/- 525.04 | +0.00 | 0/15 |
| bshin_friction | a3p70 | 2561.30 +/- 482.02 | +0.00 | 0/15 |
| bshin_mass | Vanilla | 2644.08 +/- 481.35 | +0.00 | 0/0 |
| bshin_mass | Vanilla | 2644.08 +/- 481.35 | +0.00 | 0/14 |
| bshin_mass | a2p20 | 2055.37 +/- 345.30 | +0.00 | 0/14 |
| bshin_mass | a2p40 | 2221.58 +/- 387.95 | -0.04 | 1/14 |
| bshin_mass | a2p55 | 2795.67 +/- 524.88 | +0.00 | 0/14 |
| bshin_mass | a2p65 | 2079.96 +/- 339.43 | +0.00 | 0/14 |
| bshin_mass | a2p70 | 2416.59 +/- 436.17 | +0.00 | 0/14 |
| bshin_mass | a2p75 | 2550.82 +/- 499.29 | +0.00 | 0/14 |
| bshin_mass | a2p80 | 2977.88 +/- 525.23 | +0.00 | 0/14 |
| bshin_mass | a3p00 | 2494.97 +/- 457.26 | +0.00 | 0/14 |
| bshin_mass | TV cap=3.05 | 2562.31 +/- 487.98 | +0.00 | 0/14 |
| bshin_mass | a3p10 | 2832.98 +/- 543.70 | +0.00 | 0/14 |
| bshin_mass | a3p20 | 2367.29 +/- 466.77 | +0.00 | 0/14 |
| bshin_mass | a3p40 | 2765.16 +/- 525.04 | +0.00 | 0/14 |
| bshin_mass | a3p70 | 2551.58 +/- 478.63 | +3.99 | 11/14 |
| bthigh_actuator_gain | Vanilla | 2644.08 +/- 481.35 | +0.00 | 0/0 |
| bthigh_actuator_gain | Vanilla | 2644.08 +/- 481.35 | +0.00 | 0/15 |
| bthigh_actuator_gain | a2p20 | 2055.37 +/- 345.30 | +298.36 | 15/15 |
| bthigh_actuator_gain | a2p40 | 2221.58 +/- 387.95 | +129.01 | 14/15 |
| bthigh_actuator_gain | a2p55 | 2795.67 +/- 524.88 | -10.60 | 4/15 |
| bthigh_actuator_gain | a2p65 | 2079.96 +/- 339.43 | +257.66 | 15/15 |
| bthigh_actuator_gain | a2p70 | 2410.19 +/- 428.31 | +13.65 | 7/15 |
| bthigh_actuator_gain | a2p75 | 2550.82 +/- 499.29 | +58.00 | 12/15 |
| bthigh_actuator_gain | a2p80 | 2977.88 +/- 525.23 | -227.01 | 1/15 |
| bthigh_actuator_gain | a3p00 | 2494.97 +/- 457.26 | -29.67 | 6/15 |
| bthigh_actuator_gain | TV cap=3.05 | 2562.31 +/- 487.98 | +25.04 | 11/15 |
| bthigh_actuator_gain | a3p10 | 2832.98 +/- 543.70 | -150.44 | 3/15 |
| bthigh_actuator_gain | a3p20 | 2367.29 +/- 466.77 | +55.17 | 10/15 |
| bthigh_actuator_gain | a3p40 | 2765.16 +/- 525.04 | -122.57 | 4/15 |
| bthigh_actuator_gain | a3p70 | 2561.30 +/- 482.02 | -188.23 | 5/15 |
| bthigh_damping | Vanilla | 2644.08 +/- 481.35 | +0.00 | 0/0 |
| bthigh_damping | Vanilla | 2644.08 +/- 481.35 | +0.00 | 0/15 |
| bthigh_damping | a2p20 | 2055.37 +/- 345.30 | +357.72 | 15/15 |
| bthigh_damping | a2p40 | 2221.58 +/- 387.95 | +232.89 | 15/15 |
| bthigh_damping | a2p55 | 2795.67 +/- 524.88 | +76.39 | 10/15 |
| bthigh_damping | a2p65 | 2079.96 +/- 339.43 | +290.12 | 15/15 |
| bthigh_damping | a2p70 | 2416.59 +/- 436.17 | +141.38 | 14/15 |
| bthigh_damping | a2p75 | 2550.82 +/- 499.29 | +126.61 | 13/15 |
| bthigh_damping | a2p80 | 2977.88 +/- 525.23 | -22.84 | 8/15 |
| bthigh_damping | a3p00 | 2499.53 +/- 460.40 | +69.57 | 10/15 |
| bthigh_damping | TV cap=3.05 | 2562.31 +/- 487.98 | +107.50 | 14/15 |
| bthigh_damping | a3p10 | 2832.98 +/- 543.70 | +27.05 | 9/15 |
| bthigh_damping | a3p20 | 2367.29 +/- 466.77 | +135.09 | 13/15 |
| bthigh_damping | a3p40 | 2765.16 +/- 525.04 | -1.03 | 7/15 |
| bthigh_damping | a3p70 | 2561.30 +/- 482.02 | +42.53 | 11/15 |
| bthigh_friction | Vanilla | 2644.08 +/- 481.35 | +0.00 | 0/0 |
| bthigh_friction | Vanilla | 2644.08 +/- 481.35 | +0.00 | 0/15 |
| bthigh_friction | a2p20 | 2055.37 +/- 345.30 | +0.00 | 0/15 |
| bthigh_friction | a2p40 | 2221.58 +/- 387.95 | -0.22 | 0/15 |
| bthigh_friction | a2p55 | 2795.67 +/- 524.88 | -2.51 | 0/15 |
| bthigh_friction | a2p65 | 2079.96 +/- 339.43 | +0.00 | 0/15 |
| bthigh_friction | a2p70 | 2416.59 +/- 436.17 | -3.79 | 0/15 |
| bthigh_friction | a2p75 | 2550.82 +/- 499.29 | -21.93 | 0/15 |
| bthigh_friction | a2p80 | 2977.88 +/- 525.23 | +4.73 | 4/15 |
| bthigh_friction | a3p00 | 2494.97 +/- 457.26 | -1.02 | 1/15 |
| bthigh_friction | TV cap=3.05 | 2562.31 +/- 487.98 | +5.93 | 7/15 |
| bthigh_friction | a3p10 | 2832.98 +/- 543.70 | +1.95 | 4/15 |
| bthigh_friction | a3p20 | 2354.02 +/- 456.27 | +10.74 | 13/15 |
| bthigh_friction | a3p40 | 2765.16 +/- 525.04 | +0.00 | 0/15 |
| bthigh_friction | a3p70 | 2561.30 +/- 482.02 | +0.00 | 0/15 |
| bthigh_mass | Vanilla | 2654.39 +/- 476.99 | +0.00 | 0/0 |
| bthigh_mass | Vanilla | 2654.39 +/- 476.99 | +0.00 | 0/14 |
| bthigh_mass | a2p20 | 2055.37 +/- 345.30 | +8.16 | 12/14 |
| bthigh_mass | a2p40 | 2219.94 +/- 396.47 | +9.87 | 12/14 |
| bthigh_mass | a2p55 | 2795.67 +/- 524.88 | +8.16 | 12/14 |
| bthigh_mass | a2p65 | 2079.96 +/- 339.43 | +8.16 | 12/14 |
| bthigh_mass | a2p70 | 2416.59 +/- 436.17 | +8.16 | 12/14 |
| bthigh_mass | a2p75 | 2550.82 +/- 499.29 | +8.16 | 12/14 |
| bthigh_mass | a2p80 | 2977.88 +/- 525.23 | +8.16 | 12/14 |
| bthigh_mass | a3p00 | 2494.97 +/- 457.26 | +8.16 | 12/14 |
| bthigh_mass | TV cap=3.05 | 2562.31 +/- 487.98 | +8.16 | 12/14 |
| bthigh_mass | a3p10 | 2832.98 +/- 543.70 | +8.16 | 12/14 |
| bthigh_mass | a3p20 | 2367.29 +/- 466.77 | +8.16 | 12/14 |
| bthigh_mass | a3p40 | 2765.16 +/- 525.04 | +8.16 | 12/14 |
| bthigh_mass | a3p70 | 2561.30 +/- 482.02 | +8.16 | 12/14 |
| ffoot_actuator_gain | Vanilla | 2644.08 +/- 481.35 | +0.00 | 0/0 |
| ffoot_actuator_gain | Vanilla | 2644.08 +/- 481.35 | +0.00 | 0/15 |
| ffoot_actuator_gain | a2p20 | 2055.37 +/- 345.30 | +146.00 | 14/15 |
| ffoot_actuator_gain | a2p40 | 2219.98 +/- 388.24 | +66.02 | 14/15 |
| ffoot_actuator_gain | a2p55 | 2795.67 +/- 524.88 | -237.86 | 1/15 |
| ffoot_actuator_gain | a2p65 | 2079.96 +/- 339.43 | +155.53 | 14/15 |
| ffoot_actuator_gain | a2p70 | 2416.59 +/- 436.17 | +43.01 | 12/15 |
| ffoot_actuator_gain | a2p75 | 2550.82 +/- 499.29 | +50.29 | 12/15 |
| ffoot_actuator_gain | a2p80 | 2977.88 +/- 525.23 | -149.25 | 2/15 |
| ffoot_actuator_gain | a3p00 | 2494.97 +/- 457.26 | +121.73 | 14/15 |
| ffoot_actuator_gain | TV cap=3.05 | 2562.31 +/- 487.98 | +35.61 | 12/15 |
| ffoot_actuator_gain | a3p10 | 2832.98 +/- 543.70 | -104.54 | 4/15 |
| ffoot_actuator_gain | a3p20 | 2367.29 +/- 466.77 | +132.00 | 14/15 |
| ffoot_actuator_gain | a3p40 | 2765.16 +/- 525.04 | +16.98 | 11/15 |
| ffoot_actuator_gain | a3p70 | 2561.30 +/- 482.02 | +43.07 | 10/15 |
| ffoot_damping | Vanilla | 2644.08 +/- 481.35 | +0.00 | 0/0 |
| ffoot_damping | Vanilla | 2644.08 +/- 481.35 | +0.00 | 0/15 |
| ffoot_damping | a2p20 | 2055.37 +/- 345.30 | +151.84 | 14/15 |
| ffoot_damping | a2p40 | 2219.94 +/- 396.47 | +109.88 | 13/15 |
| ffoot_damping | a2p55 | 2795.67 +/- 524.88 | -53.12 | 2/15 |
| ffoot_damping | a2p65 | 2079.96 +/- 339.43 | +149.38 | 14/15 |
| ffoot_damping | a2p70 | 2416.59 +/- 436.17 | +77.56 | 11/15 |
| ffoot_damping | a2p75 | 2550.82 +/- 499.29 | +122.95 | 14/15 |
| ffoot_damping | a2p80 | 2977.88 +/- 525.23 | -92.17 | 3/15 |
| ffoot_damping | a3p00 | 2494.97 +/- 457.26 | +141.72 | 13/15 |
| ffoot_damping | TV cap=3.05 | 2562.31 +/- 487.98 | +126.81 | 14/15 |
| ffoot_damping | a3p10 | 2837.85 +/- 548.01 | -58.06 | 5/15 |
| ffoot_damping | a3p20 | 2367.29 +/- 466.77 | +79.26 | 11/15 |
| ffoot_damping | a3p40 | 2770.06 +/- 529.61 | +33.84 | 12/15 |
| ffoot_damping | a3p70 | 2561.30 +/- 482.02 | +46.38 | 13/15 |
| ffoot_friction | Vanilla | 2644.08 +/- 481.35 | +0.00 | 0/0 |
| ffoot_friction | Vanilla | 2644.08 +/- 481.35 | +0.00 | 0/15 |
| ffoot_friction | a2p20 | 2055.37 +/- 345.30 | +0.00 | 0/15 |
| ffoot_friction | a2p40 | 2221.58 +/- 387.95 | +0.00 | 0/15 |
| ffoot_friction | a2p55 | 2793.03 +/- 525.29 | -3.80 | 10/15 |
| ffoot_friction | a2p65 | 2079.96 +/- 339.43 | +0.00 | 0/15 |
| ffoot_friction | a2p70 | 2416.59 +/- 436.17 | -1.80 | 0/15 |
| ffoot_friction | a2p75 | 2550.82 +/- 499.29 | -7.86 | 0/15 |
| ffoot_friction | a2p80 | 2977.88 +/- 525.23 | +0.42 | 1/15 |
| ffoot_friction | a3p00 | 2494.97 +/- 457.26 | +0.00 | 0/15 |
| ffoot_friction | TV cap=3.05 | 2562.31 +/- 487.98 | +0.00 | 0/15 |
| ffoot_friction | a3p10 | 2843.63 +/- 549.52 | -8.73 | 0/15 |
| ffoot_friction | a3p20 | 2349.98 +/- 456.99 | +11.54 | 10/15 |
| ffoot_friction | a3p40 | 2765.16 +/- 525.04 | +0.00 | 0/15 |
| ffoot_friction | a3p70 | 2551.58 +/- 478.63 | -1.32 | 9/15 |
| ffoot_mass | Vanilla | 2644.08 +/- 481.35 | +0.00 | 0/0 |
| ffoot_mass | Vanilla | 2644.08 +/- 481.35 | +0.00 | 0/14 |
| ffoot_mass | a2p20 | 2055.37 +/- 345.30 | +0.00 | 0/14 |
| ffoot_mass | a2p40 | 2221.58 +/- 387.95 | +0.00 | 0/14 |
| ffoot_mass | a2p55 | 2795.67 +/- 524.88 | +0.00 | 0/14 |
| ffoot_mass | a2p65 | 2079.96 +/- 339.43 | +0.00 | 0/14 |
| ffoot_mass | a2p70 | 2416.59 +/- 436.17 | +0.00 | 0/14 |
| ffoot_mass | a2p75 | 2550.82 +/- 499.29 | +0.00 | 0/14 |
| ffoot_mass | a2p80 | 2977.88 +/- 525.23 | +0.00 | 0/14 |
| ffoot_mass | a3p00 | 2494.97 +/- 457.26 | +0.00 | 0/14 |
| ffoot_mass | TV cap=3.05 | 2554.09 +/- 481.89 | +11.06 | 14/14 |
| ffoot_mass | a3p10 | 2832.98 +/- 543.70 | +0.00 | 0/14 |
| ffoot_mass | a3p20 | 2367.29 +/- 466.77 | +0.00 | 0/14 |
| ffoot_mass | a3p40 | 2765.16 +/- 525.04 | +0.00 | 0/14 |
| ffoot_mass | a3p70 | 2561.30 +/- 482.02 | -6.11 | 0/14 |
| fshin_actuator_gain | Vanilla | 2644.08 +/- 481.35 | +0.00 | 0/0 |
| fshin_actuator_gain | Vanilla | 2644.08 +/- 481.35 | +0.00 | 0/15 |
| fshin_actuator_gain | a2p20 | 2055.37 +/- 345.30 | +452.94 | 15/15 |
| fshin_actuator_gain | a2p40 | 2221.58 +/- 387.95 | +176.09 | 15/15 |
| fshin_actuator_gain | a2p55 | 2795.67 +/- 524.88 | -112.41 | 4/15 |
| fshin_actuator_gain | a2p65 | 2079.96 +/- 339.43 | +367.37 | 15/15 |
| fshin_actuator_gain | a2p70 | 2416.59 +/- 436.17 | +261.00 | 14/15 |
| fshin_actuator_gain | a2p75 | 2550.82 +/- 499.29 | +86.43 | 13/15 |
| fshin_actuator_gain | a2p80 | 2977.88 +/- 525.23 | -42.72 | 8/15 |
| fshin_actuator_gain | a3p00 | 2494.97 +/- 457.26 | +64.97 | 12/15 |
| fshin_actuator_gain | TV cap=3.05 | 2562.31 +/- 487.98 | +26.83 | 9/15 |
| fshin_actuator_gain | a3p10 | 2832.98 +/- 543.70 | +2.25 | 8/15 |
| fshin_actuator_gain | a3p20 | 2367.29 +/- 466.77 | +238.90 | 15/15 |
| fshin_actuator_gain | a3p40 | 2765.16 +/- 525.04 | -85.93 | 2/15 |
| fshin_actuator_gain | a3p70 | 2561.30 +/- 482.02 | +100.83 | 14/15 |
| fshin_damping | Vanilla | 2644.08 +/- 481.35 | +0.00 | 0/0 |
| fshin_damping | Vanilla | 2644.08 +/- 481.35 | +0.00 | 0/15 |
| fshin_damping | a2p20 | 2055.37 +/- 345.30 | +276.41 | 14/15 |
| fshin_damping | a2p40 | 2221.58 +/- 387.95 | +183.19 | 14/15 |
| fshin_damping | a2p55 | 2775.56 +/- 521.72 | -66.07 | 5/15 |
| fshin_damping | a2p65 | 2079.96 +/- 339.43 | +318.16 | 15/15 |
| fshin_damping | a2p70 | 2416.59 +/- 436.17 | +204.20 | 15/15 |
| fshin_damping | a2p75 | 2550.82 +/- 499.29 | +99.70 | 13/15 |
| fshin_damping | a2p80 | 2977.88 +/- 525.23 | +15.15 | 9/15 |
| fshin_damping | a3p00 | 2494.97 +/- 457.26 | +93.80 | 14/15 |
| fshin_damping | TV cap=3.05 | 2562.31 +/- 487.98 | +102.27 | 14/15 |
| fshin_damping | a3p10 | 2832.98 +/- 543.70 | +66.79 | 12/15 |
| fshin_damping | a3p20 | 2367.29 +/- 466.77 | +166.18 | 14/15 |
| fshin_damping | a3p40 | 2765.16 +/- 525.04 | -59.68 | 4/15 |
| fshin_damping | a3p70 | 2561.30 +/- 482.02 | +65.96 | 14/15 |
| fshin_friction | Vanilla | 2644.08 +/- 481.35 | +0.00 | 0/0 |
| fshin_friction | Vanilla | 2644.08 +/- 481.35 | +0.00 | 0/15 |
| fshin_friction | a2p20 | 2055.37 +/- 345.30 | -1.32 | 0/15 |
| fshin_friction | a2p40 | 2219.94 +/- 396.47 | -0.01 | 11/15 |
| fshin_friction | a2p55 | 2795.67 +/- 524.88 | -1.32 | 0/15 |
| fshin_friction | a2p65 | 2079.96 +/- 339.43 | -1.32 | 0/15 |
| fshin_friction | a2p70 | 2416.59 +/- 436.17 | -4.07 | 0/15 |
| fshin_friction | a2p75 | 2550.82 +/- 499.29 | -1.32 | 0/15 |
| fshin_friction | a2p80 | 2977.88 +/- 525.23 | +1.74 | 2/15 |
| fshin_friction | a3p00 | 2494.97 +/- 457.26 | -1.32 | 0/15 |
| fshin_friction | TV cap=3.05 | 2562.31 +/- 487.98 | +1.98 | 4/15 |
| fshin_friction | a3p10 | 2832.98 +/- 543.70 | +1.20 | 4/15 |
| fshin_friction | a3p20 | 2367.29 +/- 466.77 | -1.32 | 0/15 |
| fshin_friction | a3p40 | 2765.16 +/- 525.04 | -1.32 | 0/15 |
| fshin_friction | a3p70 | 2561.30 +/- 482.02 | -8.15 | 0/15 |
| fshin_mass | Vanilla | 2644.08 +/- 481.35 | +0.00 | 0/0 |
| fshin_mass | Vanilla | 2644.08 +/- 481.35 | +0.00 | 0/14 |
| fshin_mass | a2p20 | 2055.37 +/- 345.30 | -1.73 | 0/14 |
| fshin_mass | a2p40 | 2221.58 +/- 387.95 | -1.73 | 0/14 |
| fshin_mass | a2p55 | 2795.67 +/- 524.88 | -1.73 | 0/14 |
| fshin_mass | a2p65 | 2079.96 +/- 339.43 | +1.81 | 4/14 |
| fshin_mass | a2p70 | 2416.59 +/- 436.17 | -6.61 | 0/14 |
| fshin_mass | a2p75 | 2550.82 +/- 499.29 | -2.05 | 0/14 |
| fshin_mass | a2p80 | 3000.79 +/- 523.58 | -20.18 | 0/14 |
| fshin_mass | a3p00 | 2491.22 +/- 457.96 | +0.41 | 6/14 |
| fshin_mass | TV cap=3.05 | 2572.60 +/- 498.71 | -11.17 | 0/14 |
| fshin_mass | a3p10 | 2843.63 +/- 549.52 | -7.82 | 0/14 |
| fshin_mass | a3p20 | 2367.29 +/- 466.77 | -2.02 | 0/14 |
| fshin_mass | a3p40 | 2765.16 +/- 525.04 | -1.73 | 0/14 |
| fshin_mass | a3p70 | 2561.30 +/- 482.02 | -8.30 | 0/14 |
| fthigh_actuator_gain | Vanilla | 2654.39 +/- 476.99 | +0.00 | 0/0 |
| fthigh_actuator_gain | Vanilla | 2654.39 +/- 476.99 | +0.00 | 0/15 |
| fthigh_actuator_gain | a2p20 | 2055.37 +/- 345.30 | +469.35 | 13/15 |
| fthigh_actuator_gain | a2p40 | 2219.94 +/- 396.47 | +290.74 | 15/15 |
| fthigh_actuator_gain | a2p55 | 2775.56 +/- 521.72 | +134.42 | 15/15 |
| fthigh_actuator_gain | a2p65 | 2079.96 +/- 339.43 | +371.27 | 15/15 |
| fthigh_actuator_gain | a2p70 | 2416.59 +/- 436.17 | +250.49 | 15/15 |
| fthigh_actuator_gain | a2p75 | 2550.82 +/- 499.29 | +167.82 | 15/15 |
| fthigh_actuator_gain | a2p80 | 3000.79 +/- 523.58 | +2.78 | 9/15 |
| fthigh_actuator_gain | a3p00 | 2494.97 +/- 457.26 | +182.23 | 15/15 |
| fthigh_actuator_gain | TV cap=3.05 | 2562.31 +/- 487.98 | +83.94 | 12/15 |
| fthigh_actuator_gain | a3p10 | 2832.98 +/- 543.70 | +7.10 | 8/15 |
| fthigh_actuator_gain | a3p20 | 2367.29 +/- 466.77 | +234.69 | 15/15 |
| fthigh_actuator_gain | a3p40 | 2765.16 +/- 525.04 | +35.10 | 12/15 |
| fthigh_actuator_gain | a3p70 | 2561.30 +/- 482.02 | +138.42 | 13/15 |
| fthigh_damping | Vanilla | 2644.08 +/- 481.35 | +0.00 | 0/0 |
| fthigh_damping | Vanilla | 2644.08 +/- 481.35 | +0.00 | 0/15 |
| fthigh_damping | a2p20 | 2055.37 +/- 345.30 | +416.98 | 14/15 |
| fthigh_damping | a2p40 | 2221.58 +/- 387.95 | +279.30 | 14/15 |
| fthigh_damping | a2p55 | 2795.67 +/- 524.88 | +135.09 | 11/15 |
| fthigh_damping | a2p65 | 2079.96 +/- 339.43 | +429.51 | 15/15 |
| fthigh_damping | a2p70 | 2416.59 +/- 436.17 | +201.19 | 14/15 |
| fthigh_damping | a2p75 | 2550.82 +/- 499.29 | +120.93 | 14/15 |
| fthigh_damping | a2p80 | 2977.88 +/- 525.23 | +215.09 | 14/15 |
| fthigh_damping | a3p00 | 2494.97 +/- 457.26 | +187.21 | 15/15 |
| fthigh_damping | TV cap=3.05 | 2562.31 +/- 487.98 | +138.87 | 15/15 |
| fthigh_damping | a3p10 | 2832.98 +/- 543.70 | +6.55 | 10/15 |
| fthigh_damping | a3p20 | 2367.29 +/- 466.77 | +258.59 | 13/15 |
| fthigh_damping | a3p40 | 2765.16 +/- 525.04 | +152.38 | 13/15 |
| fthigh_damping | a3p70 | 2561.30 +/- 482.02 | +284.06 | 14/15 |
| fthigh_friction | Vanilla | 2644.08 +/- 481.35 | +0.00 | 0/0 |
| fthigh_friction | Vanilla | 2644.08 +/- 481.35 | +0.00 | 0/15 |
| fthigh_friction | a2p20 | 2055.37 +/- 345.30 | -0.70 | 2/15 |
| fthigh_friction | a2p40 | 2221.54 +/- 396.19 | -2.01 | 9/15 |
| fthigh_friction | a2p55 | 2795.67 +/- 524.88 | -1.43 | 0/15 |
| fthigh_friction | a2p65 | 2079.96 +/- 339.43 | -1.43 | 0/15 |
| fthigh_friction | a2p70 | 2416.59 +/- 436.17 | -5.23 | 0/15 |
| fthigh_friction | a2p75 | 2550.82 +/- 499.29 | -14.82 | 0/15 |
| fthigh_friction | a2p80 | 2977.88 +/- 525.23 | +4.26 | 4/15 |
| fthigh_friction | a3p00 | 2494.97 +/- 457.26 | -2.49 | 0/15 |
| fthigh_friction | TV cap=3.05 | 2562.31 +/- 487.98 | +0.49 | 1/15 |
| fthigh_friction | a3p10 | 2832.98 +/- 543.70 | -1.43 | 0/15 |
| fthigh_friction | a3p20 | 2367.29 +/- 466.77 | -1.43 | 0/15 |
| fthigh_friction | a3p40 | 2765.16 +/- 525.04 | -1.43 | 0/15 |
| fthigh_friction | a3p70 | 2561.30 +/- 482.02 | -1.43 | 0/15 |
| fthigh_mass | Vanilla | 2644.08 +/- 481.35 | +0.00 | 0/0 |
| fthigh_mass | Vanilla | 2644.08 +/- 481.35 | +0.00 | 0/14 |
| fthigh_mass | a2p20 | 2055.37 +/- 345.30 | -2.21 | 0/14 |
| fthigh_mass | a2p40 | 2221.58 +/- 387.95 | -2.21 | 0/14 |
| fthigh_mass | a2p55 | 2793.03 +/- 525.29 | -3.60 | 8/14 |
| fthigh_mass | a2p65 | 2076.53 +/- 339.62 | +4.52 | 11/14 |
| fthigh_mass | a2p70 | 2416.59 +/- 436.17 | -4.35 | 0/14 |
| fthigh_mass | a2p75 | 2550.82 +/- 499.29 | -10.32 | 0/14 |
| fthigh_mass | a2p80 | 2977.88 +/- 525.23 | -2.21 | 0/14 |
| fthigh_mass | a3p00 | 2494.97 +/- 457.26 | -3.01 | 0/14 |
| fthigh_mass | TV cap=3.05 | 2562.31 +/- 487.98 | -2.21 | 0/14 |
| fthigh_mass | a3p10 | 2832.98 +/- 543.70 | +0.07 | 3/14 |
| fthigh_mass | a3p20 | 2367.29 +/- 466.77 | -8.39 | 0/14 |
| fthigh_mass | a3p40 | 2778.38 +/- 527.31 | -12.64 | 0/14 |
| fthigh_mass | a3p70 | 2561.30 +/- 482.02 | -2.21 | 0/14 |

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
- `plots/with_variance/bfoot_actuator_gain_return_curve.png`
- `plots/with_variance/bfoot_actuator_gain_return_curve.pdf`
- `plots/with_variance/bfoot_actuator_gain_gain_curve.png`
- `plots/with_variance/bfoot_actuator_gain_gain_curve.pdf`
- `plots/without_variance/bfoot_actuator_gain_return_curve.png`
- `plots/without_variance/bfoot_actuator_gain_return_curve.pdf`
- `plots/without_variance/bfoot_actuator_gain_gain_curve.png`
- `plots/without_variance/bfoot_actuator_gain_gain_curve.pdf`
- `plots/with_variance/bfoot_damping_return_curve.png`
- `plots/with_variance/bfoot_damping_return_curve.pdf`
- `plots/with_variance/bfoot_damping_gain_curve.png`
- `plots/with_variance/bfoot_damping_gain_curve.pdf`
- `plots/without_variance/bfoot_damping_return_curve.png`
- `plots/without_variance/bfoot_damping_return_curve.pdf`
- `plots/without_variance/bfoot_damping_gain_curve.png`
- `plots/without_variance/bfoot_damping_gain_curve.pdf`
- `plots/with_variance/bfoot_friction_return_curve.png`
- `plots/with_variance/bfoot_friction_return_curve.pdf`
- `plots/with_variance/bfoot_friction_gain_curve.png`
- `plots/with_variance/bfoot_friction_gain_curve.pdf`
- `plots/without_variance/bfoot_friction_return_curve.png`
- `plots/without_variance/bfoot_friction_return_curve.pdf`
- `plots/without_variance/bfoot_friction_gain_curve.png`
- `plots/without_variance/bfoot_friction_gain_curve.pdf`
- `plots/with_variance/bfoot_mass_return_curve.png`
- `plots/with_variance/bfoot_mass_return_curve.pdf`
- `plots/with_variance/bfoot_mass_gain_curve.png`
- `plots/with_variance/bfoot_mass_gain_curve.pdf`
- `plots/without_variance/bfoot_mass_return_curve.png`
- `plots/without_variance/bfoot_mass_return_curve.pdf`
- `plots/without_variance/bfoot_mass_gain_curve.png`
- `plots/without_variance/bfoot_mass_gain_curve.pdf`
- `plots/with_variance/bshin_actuator_gain_return_curve.png`
- `plots/with_variance/bshin_actuator_gain_return_curve.pdf`
- `plots/with_variance/bshin_actuator_gain_gain_curve.png`
- `plots/with_variance/bshin_actuator_gain_gain_curve.pdf`
- `plots/without_variance/bshin_actuator_gain_return_curve.png`
- `plots/without_variance/bshin_actuator_gain_return_curve.pdf`
- `plots/without_variance/bshin_actuator_gain_gain_curve.png`
- `plots/without_variance/bshin_actuator_gain_gain_curve.pdf`
- `plots/with_variance/bshin_damping_return_curve.png`
- `plots/with_variance/bshin_damping_return_curve.pdf`
- `plots/with_variance/bshin_damping_gain_curve.png`
- `plots/with_variance/bshin_damping_gain_curve.pdf`
- `plots/without_variance/bshin_damping_return_curve.png`
- `plots/without_variance/bshin_damping_return_curve.pdf`
- `plots/without_variance/bshin_damping_gain_curve.png`
- `plots/without_variance/bshin_damping_gain_curve.pdf`
- `plots/with_variance/bshin_friction_return_curve.png`
- `plots/with_variance/bshin_friction_return_curve.pdf`
- `plots/with_variance/bshin_friction_gain_curve.png`
- `plots/with_variance/bshin_friction_gain_curve.pdf`
- `plots/without_variance/bshin_friction_return_curve.png`
- `plots/without_variance/bshin_friction_return_curve.pdf`
- `plots/without_variance/bshin_friction_gain_curve.png`
- `plots/without_variance/bshin_friction_gain_curve.pdf`
- `plots/with_variance/bshin_mass_return_curve.png`
- `plots/with_variance/bshin_mass_return_curve.pdf`
- `plots/with_variance/bshin_mass_gain_curve.png`
- `plots/with_variance/bshin_mass_gain_curve.pdf`
- `plots/without_variance/bshin_mass_return_curve.png`
- `plots/without_variance/bshin_mass_return_curve.pdf`
- `plots/without_variance/bshin_mass_gain_curve.png`
- `plots/without_variance/bshin_mass_gain_curve.pdf`
- `plots/with_variance/bthigh_actuator_gain_return_curve.png`
- `plots/with_variance/bthigh_actuator_gain_return_curve.pdf`
- `plots/with_variance/bthigh_actuator_gain_gain_curve.png`
- `plots/with_variance/bthigh_actuator_gain_gain_curve.pdf`
- `plots/without_variance/bthigh_actuator_gain_return_curve.png`
- `plots/without_variance/bthigh_actuator_gain_return_curve.pdf`
- `plots/without_variance/bthigh_actuator_gain_gain_curve.png`
- `plots/without_variance/bthigh_actuator_gain_gain_curve.pdf`
- `plots/with_variance/bthigh_damping_return_curve.png`
- `plots/with_variance/bthigh_damping_return_curve.pdf`
- `plots/with_variance/bthigh_damping_gain_curve.png`
- `plots/with_variance/bthigh_damping_gain_curve.pdf`
- `plots/without_variance/bthigh_damping_return_curve.png`
- `plots/without_variance/bthigh_damping_return_curve.pdf`
- `plots/without_variance/bthigh_damping_gain_curve.png`
- `plots/without_variance/bthigh_damping_gain_curve.pdf`
- `plots/with_variance/bthigh_friction_return_curve.png`
- `plots/with_variance/bthigh_friction_return_curve.pdf`
- `plots/with_variance/bthigh_friction_gain_curve.png`
- `plots/with_variance/bthigh_friction_gain_curve.pdf`
- `plots/without_variance/bthigh_friction_return_curve.png`
- `plots/without_variance/bthigh_friction_return_curve.pdf`
- `plots/without_variance/bthigh_friction_gain_curve.png`
- `plots/without_variance/bthigh_friction_gain_curve.pdf`
- `plots/with_variance/bthigh_mass_return_curve.png`
- `plots/with_variance/bthigh_mass_return_curve.pdf`
- `plots/with_variance/bthigh_mass_gain_curve.png`
- `plots/with_variance/bthigh_mass_gain_curve.pdf`
- `plots/without_variance/bthigh_mass_return_curve.png`
- `plots/without_variance/bthigh_mass_return_curve.pdf`
- `plots/without_variance/bthigh_mass_gain_curve.png`
- `plots/without_variance/bthigh_mass_gain_curve.pdf`
- `plots/with_variance/ffoot_actuator_gain_return_curve.png`
- `plots/with_variance/ffoot_actuator_gain_return_curve.pdf`
- `plots/with_variance/ffoot_actuator_gain_gain_curve.png`
- `plots/with_variance/ffoot_actuator_gain_gain_curve.pdf`
- `plots/without_variance/ffoot_actuator_gain_return_curve.png`
- `plots/without_variance/ffoot_actuator_gain_return_curve.pdf`
- `plots/without_variance/ffoot_actuator_gain_gain_curve.png`
- `plots/without_variance/ffoot_actuator_gain_gain_curve.pdf`
- `plots/with_variance/ffoot_damping_return_curve.png`
- `plots/with_variance/ffoot_damping_return_curve.pdf`
- `plots/with_variance/ffoot_damping_gain_curve.png`
- `plots/with_variance/ffoot_damping_gain_curve.pdf`
- `plots/without_variance/ffoot_damping_return_curve.png`
- `plots/without_variance/ffoot_damping_return_curve.pdf`
- `plots/without_variance/ffoot_damping_gain_curve.png`
- `plots/without_variance/ffoot_damping_gain_curve.pdf`
- `plots/with_variance/ffoot_friction_return_curve.png`
- `plots/with_variance/ffoot_friction_return_curve.pdf`
- `plots/with_variance/ffoot_friction_gain_curve.png`
- `plots/with_variance/ffoot_friction_gain_curve.pdf`
- `plots/without_variance/ffoot_friction_return_curve.png`
- `plots/without_variance/ffoot_friction_return_curve.pdf`
- `plots/without_variance/ffoot_friction_gain_curve.png`
- `plots/without_variance/ffoot_friction_gain_curve.pdf`
- `plots/with_variance/ffoot_mass_return_curve.png`
- `plots/with_variance/ffoot_mass_return_curve.pdf`
- `plots/with_variance/ffoot_mass_gain_curve.png`
- `plots/with_variance/ffoot_mass_gain_curve.pdf`
- `plots/without_variance/ffoot_mass_return_curve.png`
- `plots/without_variance/ffoot_mass_return_curve.pdf`
- `plots/without_variance/ffoot_mass_gain_curve.png`
- `plots/without_variance/ffoot_mass_gain_curve.pdf`
- `plots/with_variance/fshin_actuator_gain_return_curve.png`
- `plots/with_variance/fshin_actuator_gain_return_curve.pdf`
- `plots/with_variance/fshin_actuator_gain_gain_curve.png`
- `plots/with_variance/fshin_actuator_gain_gain_curve.pdf`
- `plots/without_variance/fshin_actuator_gain_return_curve.png`
- `plots/without_variance/fshin_actuator_gain_return_curve.pdf`
- `plots/without_variance/fshin_actuator_gain_gain_curve.png`
- `plots/without_variance/fshin_actuator_gain_gain_curve.pdf`
- `plots/with_variance/fshin_damping_return_curve.png`
- `plots/with_variance/fshin_damping_return_curve.pdf`
- `plots/with_variance/fshin_damping_gain_curve.png`
- `plots/with_variance/fshin_damping_gain_curve.pdf`
- `plots/without_variance/fshin_damping_return_curve.png`
- `plots/without_variance/fshin_damping_return_curve.pdf`
- `plots/without_variance/fshin_damping_gain_curve.png`
- `plots/without_variance/fshin_damping_gain_curve.pdf`
- `plots/with_variance/fshin_friction_return_curve.png`
- `plots/with_variance/fshin_friction_return_curve.pdf`
- `plots/with_variance/fshin_friction_gain_curve.png`
- `plots/with_variance/fshin_friction_gain_curve.pdf`
- `plots/without_variance/fshin_friction_return_curve.png`
- `plots/without_variance/fshin_friction_return_curve.pdf`
- `plots/without_variance/fshin_friction_gain_curve.png`
- `plots/without_variance/fshin_friction_gain_curve.pdf`
- `plots/with_variance/fshin_mass_return_curve.png`
- `plots/with_variance/fshin_mass_return_curve.pdf`
- `plots/with_variance/fshin_mass_gain_curve.png`
- `plots/with_variance/fshin_mass_gain_curve.pdf`
- `plots/without_variance/fshin_mass_return_curve.png`
- `plots/without_variance/fshin_mass_return_curve.pdf`
- `plots/without_variance/fshin_mass_gain_curve.png`
- `plots/without_variance/fshin_mass_gain_curve.pdf`
- `plots/with_variance/fthigh_actuator_gain_return_curve.png`
- `plots/with_variance/fthigh_actuator_gain_return_curve.pdf`
- `plots/with_variance/fthigh_actuator_gain_gain_curve.png`
- `plots/with_variance/fthigh_actuator_gain_gain_curve.pdf`
- `plots/without_variance/fthigh_actuator_gain_return_curve.png`
- `plots/without_variance/fthigh_actuator_gain_return_curve.pdf`
- `plots/without_variance/fthigh_actuator_gain_gain_curve.png`
- `plots/without_variance/fthigh_actuator_gain_gain_curve.pdf`
- `plots/with_variance/fthigh_damping_return_curve.png`
- `plots/with_variance/fthigh_damping_return_curve.pdf`
- `plots/with_variance/fthigh_damping_gain_curve.png`
- `plots/with_variance/fthigh_damping_gain_curve.pdf`
- `plots/without_variance/fthigh_damping_return_curve.png`
- `plots/without_variance/fthigh_damping_return_curve.pdf`
- `plots/without_variance/fthigh_damping_gain_curve.png`
- `plots/without_variance/fthigh_damping_gain_curve.pdf`
- `plots/with_variance/fthigh_friction_return_curve.png`
- `plots/with_variance/fthigh_friction_return_curve.pdf`
- `plots/with_variance/fthigh_friction_gain_curve.png`
- `plots/with_variance/fthigh_friction_gain_curve.pdf`
- `plots/without_variance/fthigh_friction_return_curve.png`
- `plots/without_variance/fthigh_friction_return_curve.pdf`
- `plots/without_variance/fthigh_friction_gain_curve.png`
- `plots/without_variance/fthigh_friction_gain_curve.pdf`
- `plots/with_variance/fthigh_mass_return_curve.png`
- `plots/with_variance/fthigh_mass_return_curve.pdf`
- `plots/with_variance/fthigh_mass_gain_curve.png`
- `plots/with_variance/fthigh_mass_gain_curve.pdf`
- `plots/without_variance/fthigh_mass_return_curve.png`
- `plots/without_variance/fthigh_mass_return_curve.pdf`
- `plots/without_variance/fthigh_mass_gain_curve.png`
- `plots/without_variance/fthigh_mass_gain_curve.pdf`

## Output CSV files

- `outputs/eval_metrics_final.csv`: latest merged per-seed eval rows.
- `outputs/summary_by_scenario.csv`: per-model, per-axis, per-factor aggregate return table.
- `outputs/drop_summary.csv`: nominal-minus-perturbed drop table.
- `outputs/gain_summary.csv`: paired robust-gain table against vanilla.
- `outputs/axis_overview.csv`: compact axis-level overview.
- `outputs/curve_points.csv`: same data used for return plots.
- `outputs/gain_curve_points.csv`: same data used for gain plots.
