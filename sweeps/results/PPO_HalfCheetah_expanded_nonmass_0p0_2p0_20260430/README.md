# PPO_HalfCheetah_expanded_nonmass_0p0_2p0_20260430

Date: 2026-05-15

This folder packages the final pinned robustness evaluation for normalized PPO.

## Contents

- Raw metrics: `/Users/arjunmahesh/Library/CloudStorage/OneDrive-Personal/Duke/Research/RL/cleanrl/sweeps/results/PPO_HalfCheetah_final_presentable_20260430/raw_metrics/nonmass`
- Aggregated outputs: `/Users/arjunmahesh/Library/CloudStorage/OneDrive-Personal/Duke/Research/RL/cleanrl/sweeps/results/PPO_HalfCheetah_expanded_nonmass_0p0_2p0_20260430/outputs`
- Plots: `/Users/arjunmahesh/Library/CloudStorage/OneDrive-Personal/Duke/Research/RL/cleanrl/sweeps/results/PPO_HalfCheetah_expanded_nonmass_0p0_2p0_20260430/plots`

## Evaluation protocol

- Models are compared on the same perturbation grid for the configured axes.
- Nominal reference within each axis is the `factor=1.0` point.
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
| bfoot_actuator_gain | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1366.02 +/- 0.00 | 1335.03 +/- 0.00 | 1379.30 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 |
| bfoot_damping | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1350.85 +/- 0.00 | 1348.02 +/- 0.00 | 1372.01 +/- 0.00 | 1492.24 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 |
| bfoot_friction | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1327.70 +/- 0.00 | 1370.43 +/- 0.00 | 1472.44 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 |
| bshin_actuator_gain | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1267.20 +/- 0.00 | 1389.17 +/- 0.00 | 1363.59 +/- 0.00 | 1489.41 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 |
| bshin_damping | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1332.32 +/- 0.00 | 1378.30 +/- 0.00 | 1475.50 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 |
| bshin_friction | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1340.78 +/- 0.00 | 1382.16 +/- 0.00 | 1463.40 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 |
| bthigh_actuator_gain | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1356.72 +/- 0.00 | 1366.02 +/- 0.00 | 1387.62 +/- 0.00 | 1491.43 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 |
| bthigh_damping | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1558.46 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1384.47 +/- 0.00 | 1377.60 +/- 0.00 | 1476.70 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 |
| bthigh_friction | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1364.40 +/- 0.00 | 1371.16 +/- 0.00 | 1491.94 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 |
| damping | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1385.10 +/- 0.00 | 1368.77 +/- 0.00 | 1475.72 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 |
| ffoot_actuator_gain | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1358.42 +/- 0.00 | 1390.73 +/- 0.00 | 1358.33 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 |
| ffoot_damping | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1361.66 +/- 0.00 | 1400.73 +/- 0.00 | 1377.20 +/- 0.00 | 1492.31 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 |
| ffoot_friction | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1364.45 +/- 0.00 | 1383.80 +/- 0.00 | 1488.41 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 |
| friction | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1383.47 +/- 0.00 | 1370.35 +/- 0.00 | 1474.07 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 |
| friction_damping | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1351.99 +/- 0.00 | 1374.25 +/- 0.00 | 1485.24 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 |
| fshin_actuator_gain | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1367.10 +/- 0.00 | 1326.66 +/- 0.00 | 1363.47 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 |
| fshin_damping | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1183.23 +/- 0.00 | 1340.78 +/- 0.00 | 1383.36 +/- 0.00 | 1484.36 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 |
| fshin_friction | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1341.69 +/- 0.00 | 1372.99 +/- 0.00 | 1463.88 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 |
| fthigh_actuator_gain | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1250.51 +/- 0.00 | 1380.00 +/- 0.00 | 1373.69 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 |
| fthigh_damping | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1346.85 +/- 0.00 | 1376.14 +/- 0.00 | 1384.96 +/- 0.00 | 1493.83 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 |
| fthigh_friction | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1366.01 +/- 0.00 | 1375.94 +/- 0.00 | 1495.55 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 |
| gear | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1381.32 +/- 0.00 | 1373.91 +/- 0.00 | 1477.40 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 |
| gravity | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 896.65 +/- 0.00 | 1044.43 +/- 0.00 | 868.67 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 | 1.00 +/- 0.00 |

## Axis overview

| Axis | Model | Nominal return | Mean gain over perturbed scenarios | Positive gain scenarios |
| --- | --- | --- | --- | --- |
| bfoot_actuator_gain | TV cap=2.20 | 1.00 +/- 0.00 | +nan | 0/0 |
| bfoot_actuator_gain | TV cap=2.40 | 1.00 +/- 0.00 | +nan | 0/0 |
| bfoot_actuator_gain | TV cap=2.55 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| bfoot_actuator_gain | TV cap=2.65 | 1.00 +/- 0.00 | +nan | 0/0 |
| bfoot_actuator_gain | TV cap=2.70 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| bfoot_actuator_gain | TV cap=2.75 | 1.00 +/- 0.00 | +nan | 0/0 |
| bfoot_actuator_gain | TV cap=2.80 | 1366.02 +/- 0.00 | -84.05 | 0/15 |
| bfoot_actuator_gain | TV cap=3.00 | 1335.03 +/- 0.00 | +nan | 0/0 |
| bfoot_actuator_gain | TV cap=3.05 | 1379.30 +/- 0.00 | +nan | 0/0 |
| bfoot_actuator_gain | TV cap=3.10 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| bfoot_actuator_gain | TV cap=3.20 | 1.00 +/- 0.00 | +nan | 0/0 |
| bfoot_actuator_gain | TV cap=3.40 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| bfoot_actuator_gain | TV cap=3.70 | 1.00 +/- 0.00 | +nan | 0/0 |
| bfoot_damping | TV cap=2.20 | 1.00 +/- 0.00 | +nan | 0/0 |
| bfoot_damping | TV cap=2.40 | 1.00 +/- 0.00 | +nan | 0/0 |
| bfoot_damping | TV cap=2.55 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| bfoot_damping | TV cap=2.65 | 1.00 +/- 0.00 | +nan | 0/0 |
| bfoot_damping | TV cap=2.70 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| bfoot_damping | TV cap=2.75 | 1.00 +/- 0.00 | +nan | 0/0 |
| bfoot_damping | TV cap=2.80 | 1350.85 +/- 0.00 | -226.53 | 4/15 |
| bfoot_damping | TV cap=3.00 | 1348.02 +/- 0.00 | +nan | 0/0 |
| bfoot_damping | TV cap=3.05 | 1372.01 +/- 0.00 | +nan | 0/0 |
| bfoot_damping | TV cap=3.10 | 1492.24 +/- 0.00 | -22.89 | 1/15 |
| bfoot_damping | TV cap=3.20 | 1.00 +/- 0.00 | +nan | 0/0 |
| bfoot_damping | TV cap=3.40 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| bfoot_damping | TV cap=3.70 | 1.00 +/- 0.00 | +nan | 0/0 |
| bfoot_friction | TV cap=2.20 | 1.00 +/- 0.00 | +nan | 0/0 |
| bfoot_friction | TV cap=2.40 | 1.00 +/- 0.00 | +nan | 0/0 |
| bfoot_friction | TV cap=2.55 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| bfoot_friction | TV cap=2.65 | 1.00 +/- 0.00 | +nan | 0/0 |
| bfoot_friction | TV cap=2.70 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| bfoot_friction | TV cap=2.75 | 1.00 +/- 0.00 | +nan | 0/0 |
| bfoot_friction | TV cap=2.80 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| bfoot_friction | TV cap=3.00 | 1327.70 +/- 0.00 | +nan | 0/0 |
| bfoot_friction | TV cap=3.05 | 1370.43 +/- 0.00 | +nan | 0/0 |
| bfoot_friction | TV cap=3.10 | 1472.44 +/- 0.00 | +11.53 | 12/15 |
| bfoot_friction | TV cap=3.20 | 1.00 +/- 0.00 | +nan | 0/0 |
| bfoot_friction | TV cap=3.40 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| bfoot_friction | TV cap=3.70 | 1.00 +/- 0.00 | +nan | 0/0 |
| bshin_actuator_gain | TV cap=2.20 | 1.00 +/- 0.00 | +nan | 0/0 |
| bshin_actuator_gain | TV cap=2.40 | 1.00 +/- 0.00 | +nan | 0/0 |
| bshin_actuator_gain | TV cap=2.55 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| bshin_actuator_gain | TV cap=2.65 | 1.00 +/- 0.00 | +nan | 0/0 |
| bshin_actuator_gain | TV cap=2.70 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| bshin_actuator_gain | TV cap=2.75 | 1.00 +/- 0.00 | +nan | 0/0 |
| bshin_actuator_gain | TV cap=2.80 | 1267.20 +/- 0.00 | -536.36 | 4/15 |
| bshin_actuator_gain | TV cap=3.00 | 1389.17 +/- 0.00 | +nan | 0/0 |
| bshin_actuator_gain | TV cap=3.05 | 1363.59 +/- 0.00 | +nan | 0/0 |
| bshin_actuator_gain | TV cap=3.10 | 1489.41 +/- 0.00 | -584.68 | 2/15 |
| bshin_actuator_gain | TV cap=3.20 | 1.00 +/- 0.00 | +nan | 0/0 |
| bshin_actuator_gain | TV cap=3.40 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| bshin_actuator_gain | TV cap=3.70 | 1.00 +/- 0.00 | +nan | 0/0 |
| bshin_damping | TV cap=2.20 | 1.00 +/- 0.00 | +nan | 0/0 |
| bshin_damping | TV cap=2.40 | 1.00 +/- 0.00 | +nan | 0/0 |
| bshin_damping | TV cap=2.55 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| bshin_damping | TV cap=2.65 | 1.00 +/- 0.00 | +nan | 0/0 |
| bshin_damping | TV cap=2.70 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| bshin_damping | TV cap=2.75 | 1.00 +/- 0.00 | +nan | 0/0 |
| bshin_damping | TV cap=2.80 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| bshin_damping | TV cap=3.00 | 1332.32 +/- 0.00 | +nan | 0/0 |
| bshin_damping | TV cap=3.05 | 1378.30 +/- 0.00 | +nan | 0/0 |
| bshin_damping | TV cap=3.10 | 1475.50 +/- 0.00 | -407.38 | 1/15 |
| bshin_damping | TV cap=3.20 | 1.00 +/- 0.00 | +nan | 0/0 |
| bshin_damping | TV cap=3.40 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| bshin_damping | TV cap=3.70 | 1.00 +/- 0.00 | +nan | 0/0 |
| bshin_friction | TV cap=2.20 | 1.00 +/- 0.00 | +nan | 0/0 |
| bshin_friction | TV cap=2.40 | 1.00 +/- 0.00 | +nan | 0/0 |
| bshin_friction | TV cap=2.55 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| bshin_friction | TV cap=2.65 | 1.00 +/- 0.00 | +nan | 0/0 |
| bshin_friction | TV cap=2.70 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| bshin_friction | TV cap=2.75 | 1.00 +/- 0.00 | +nan | 0/0 |
| bshin_friction | TV cap=2.80 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| bshin_friction | TV cap=3.00 | 1340.78 +/- 0.00 | +nan | 0/0 |
| bshin_friction | TV cap=3.05 | 1382.16 +/- 0.00 | +nan | 0/0 |
| bshin_friction | TV cap=3.10 | 1463.40 +/- 0.00 | +16.56 | 14/15 |
| bshin_friction | TV cap=3.20 | 1.00 +/- 0.00 | +nan | 0/0 |
| bshin_friction | TV cap=3.40 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| bshin_friction | TV cap=3.70 | 1.00 +/- 0.00 | +nan | 0/0 |
| bthigh_actuator_gain | TV cap=2.20 | 1.00 +/- 0.00 | +nan | 0/0 |
| bthigh_actuator_gain | TV cap=2.40 | 1.00 +/- 0.00 | +nan | 0/0 |
| bthigh_actuator_gain | TV cap=2.55 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| bthigh_actuator_gain | TV cap=2.65 | 1.00 +/- 0.00 | +nan | 0/0 |
| bthigh_actuator_gain | TV cap=2.70 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| bthigh_actuator_gain | TV cap=2.75 | 1.00 +/- 0.00 | +nan | 0/0 |
| bthigh_actuator_gain | TV cap=2.80 | 1356.72 +/- 0.00 | -542.87 | 0/15 |
| bthigh_actuator_gain | TV cap=3.00 | 1366.02 +/- 0.00 | +nan | 0/0 |
| bthigh_actuator_gain | TV cap=3.05 | 1387.62 +/- 0.00 | +nan | 0/0 |
| bthigh_actuator_gain | TV cap=3.10 | 1491.43 +/- 0.00 | -492.60 | 1/15 |
| bthigh_actuator_gain | TV cap=3.20 | 1.00 +/- 0.00 | +nan | 0/0 |
| bthigh_actuator_gain | TV cap=3.40 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| bthigh_actuator_gain | TV cap=3.70 | 1.00 +/- 0.00 | +nan | 0/0 |
| bthigh_damping | TV cap=2.20 | 1.00 +/- 0.00 | +nan | 0/0 |
| bthigh_damping | TV cap=2.40 | 1.00 +/- 0.00 | +nan | 0/0 |
| bthigh_damping | TV cap=2.55 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| bthigh_damping | TV cap=2.65 | 1.00 +/- 0.00 | +nan | 0/0 |
| bthigh_damping | TV cap=2.70 | 1558.46 +/- 0.00 | -1254.45 | 1/15 |
| bthigh_damping | TV cap=2.75 | 1.00 +/- 0.00 | +nan | 0/0 |
| bthigh_damping | TV cap=2.80 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| bthigh_damping | TV cap=3.00 | 1384.47 +/- 0.00 | +nan | 0/0 |
| bthigh_damping | TV cap=3.05 | 1377.60 +/- 0.00 | +nan | 0/0 |
| bthigh_damping | TV cap=3.10 | 1476.70 +/- 0.00 | -548.70 | 0/15 |
| bthigh_damping | TV cap=3.20 | 1.00 +/- 0.00 | +nan | 0/0 |
| bthigh_damping | TV cap=3.40 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| bthigh_damping | TV cap=3.70 | 1.00 +/- 0.00 | +nan | 0/0 |
| bthigh_friction | TV cap=2.20 | 1.00 +/- 0.00 | +nan | 0/0 |
| bthigh_friction | TV cap=2.40 | 1.00 +/- 0.00 | +nan | 0/0 |
| bthigh_friction | TV cap=2.55 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| bthigh_friction | TV cap=2.65 | 1.00 +/- 0.00 | +nan | 0/0 |
| bthigh_friction | TV cap=2.70 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| bthigh_friction | TV cap=2.75 | 1.00 +/- 0.00 | +nan | 0/0 |
| bthigh_friction | TV cap=2.80 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| bthigh_friction | TV cap=3.00 | 1364.40 +/- 0.00 | +nan | 0/0 |
| bthigh_friction | TV cap=3.05 | 1371.16 +/- 0.00 | +nan | 0/0 |
| bthigh_friction | TV cap=3.10 | 1491.94 +/- 0.00 | -7.50 | 2/15 |
| bthigh_friction | TV cap=3.20 | 1.00 +/- 0.00 | +nan | 0/0 |
| bthigh_friction | TV cap=3.40 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| bthigh_friction | TV cap=3.70 | 1.00 +/- 0.00 | +nan | 0/0 |
| damping | TV cap=2.20 | 1.00 +/- 0.00 | +nan | 0/0 |
| damping | TV cap=2.40 | 1.00 +/- 0.00 | +nan | 0/0 |
| damping | TV cap=2.55 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| damping | TV cap=2.65 | 1.00 +/- 0.00 | +nan | 0/0 |
| damping | TV cap=2.70 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| damping | TV cap=2.75 | 1.00 +/- 0.00 | +nan | 0/0 |
| damping | TV cap=2.80 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| damping | TV cap=3.00 | 1385.10 +/- 0.00 | +nan | 0/0 |
| damping | TV cap=3.05 | 1368.77 +/- 0.00 | +nan | 0/0 |
| damping | TV cap=3.10 | 1475.72 +/- 0.00 | -547.04 | 1/15 |
| damping | TV cap=3.20 | 1.00 +/- 0.00 | +nan | 0/0 |
| damping | TV cap=3.40 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| damping | TV cap=3.70 | 1.00 +/- 0.00 | +nan | 0/0 |
| ffoot_actuator_gain | TV cap=2.20 | 1.00 +/- 0.00 | +nan | 0/0 |
| ffoot_actuator_gain | TV cap=2.40 | 1.00 +/- 0.00 | +nan | 0/0 |
| ffoot_actuator_gain | TV cap=2.55 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| ffoot_actuator_gain | TV cap=2.65 | 1.00 +/- 0.00 | +nan | 0/0 |
| ffoot_actuator_gain | TV cap=2.70 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| ffoot_actuator_gain | TV cap=2.75 | 1.00 +/- 0.00 | +nan | 0/0 |
| ffoot_actuator_gain | TV cap=2.80 | 1358.42 +/- 0.00 | -43.40 | 1/15 |
| ffoot_actuator_gain | TV cap=3.00 | 1390.73 +/- 0.00 | +nan | 0/0 |
| ffoot_actuator_gain | TV cap=3.05 | 1358.33 +/- 0.00 | +nan | 0/0 |
| ffoot_actuator_gain | TV cap=3.10 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| ffoot_actuator_gain | TV cap=3.20 | 1.00 +/- 0.00 | +nan | 0/0 |
| ffoot_actuator_gain | TV cap=3.40 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| ffoot_actuator_gain | TV cap=3.70 | 1.00 +/- 0.00 | +nan | 0/0 |
| ffoot_damping | TV cap=2.20 | 1.00 +/- 0.00 | +nan | 0/0 |
| ffoot_damping | TV cap=2.40 | 1.00 +/- 0.00 | +nan | 0/0 |
| ffoot_damping | TV cap=2.55 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| ffoot_damping | TV cap=2.65 | 1.00 +/- 0.00 | +nan | 0/0 |
| ffoot_damping | TV cap=2.70 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| ffoot_damping | TV cap=2.75 | 1.00 +/- 0.00 | +nan | 0/0 |
| ffoot_damping | TV cap=2.80 | 1361.66 +/- 0.00 | -66.56 | 0/15 |
| ffoot_damping | TV cap=3.00 | 1400.73 +/- 0.00 | +nan | 0/0 |
| ffoot_damping | TV cap=3.05 | 1377.20 +/- 0.00 | +nan | 0/0 |
| ffoot_damping | TV cap=3.10 | 1492.31 +/- 0.00 | -35.66 | 0/15 |
| ffoot_damping | TV cap=3.20 | 1.00 +/- 0.00 | +nan | 0/0 |
| ffoot_damping | TV cap=3.40 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| ffoot_damping | TV cap=3.70 | 1.00 +/- 0.00 | +nan | 0/0 |
| ffoot_friction | TV cap=2.20 | 1.00 +/- 0.00 | +nan | 0/0 |
| ffoot_friction | TV cap=2.40 | 1.00 +/- 0.00 | +nan | 0/0 |
| ffoot_friction | TV cap=2.55 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| ffoot_friction | TV cap=2.65 | 1.00 +/- 0.00 | +nan | 0/0 |
| ffoot_friction | TV cap=2.70 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| ffoot_friction | TV cap=2.75 | 1.00 +/- 0.00 | +nan | 0/0 |
| ffoot_friction | TV cap=2.80 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| ffoot_friction | TV cap=3.00 | 1364.45 +/- 0.00 | +nan | 0/0 |
| ffoot_friction | TV cap=3.05 | 1383.80 +/- 0.00 | +nan | 0/0 |
| ffoot_friction | TV cap=3.10 | 1488.41 +/- 0.00 | -3.96 | 6/15 |
| ffoot_friction | TV cap=3.20 | 1.00 +/- 0.00 | +nan | 0/0 |
| ffoot_friction | TV cap=3.40 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| ffoot_friction | TV cap=3.70 | 1.00 +/- 0.00 | +nan | 0/0 |
| friction | TV cap=2.20 | 1.00 +/- 0.00 | +nan | 0/0 |
| friction | TV cap=2.40 | 1.00 +/- 0.00 | +nan | 0/0 |
| friction | TV cap=2.55 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| friction | TV cap=2.65 | 1.00 +/- 0.00 | +nan | 0/0 |
| friction | TV cap=2.70 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| friction | TV cap=2.75 | 1.00 +/- 0.00 | +nan | 0/0 |
| friction | TV cap=2.80 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| friction | TV cap=3.00 | 1383.47 +/- 0.00 | +nan | 0/0 |
| friction | TV cap=3.05 | 1370.35 +/- 0.00 | +nan | 0/0 |
| friction | TV cap=3.10 | 1474.07 +/- 0.00 | +189.57 | 10/15 |
| friction | TV cap=3.20 | 1.00 +/- 0.00 | +nan | 0/0 |
| friction | TV cap=3.40 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| friction | TV cap=3.70 | 1.00 +/- 0.00 | +nan | 0/0 |
| friction_damping | TV cap=2.20 | 1.00 +/- 0.00 | +nan | 0/0 |
| friction_damping | TV cap=2.40 | 1.00 +/- 0.00 | +nan | 0/0 |
| friction_damping | TV cap=2.55 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| friction_damping | TV cap=2.65 | 1.00 +/- 0.00 | +nan | 0/0 |
| friction_damping | TV cap=2.70 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| friction_damping | TV cap=2.75 | 1.00 +/- 0.00 | +nan | 0/0 |
| friction_damping | TV cap=2.80 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| friction_damping | TV cap=3.00 | 1351.99 +/- 0.00 | +nan | 0/0 |
| friction_damping | TV cap=3.05 | 1374.25 +/- 0.00 | +nan | 0/0 |
| friction_damping | TV cap=3.10 | 1485.24 +/- 0.00 | -670.12 | 2/15 |
| friction_damping | TV cap=3.20 | 1.00 +/- 0.00 | +nan | 0/0 |
| friction_damping | TV cap=3.40 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| friction_damping | TV cap=3.70 | 1.00 +/- 0.00 | +nan | 0/0 |
| fshin_actuator_gain | TV cap=2.20 | 1.00 +/- 0.00 | +nan | 0/0 |
| fshin_actuator_gain | TV cap=2.40 | 1.00 +/- 0.00 | +nan | 0/0 |
| fshin_actuator_gain | TV cap=2.55 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| fshin_actuator_gain | TV cap=2.65 | 1.00 +/- 0.00 | +nan | 0/0 |
| fshin_actuator_gain | TV cap=2.70 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| fshin_actuator_gain | TV cap=2.75 | 1.00 +/- 0.00 | +nan | 0/0 |
| fshin_actuator_gain | TV cap=2.80 | 1367.10 +/- 0.00 | -279.12 | 0/15 |
| fshin_actuator_gain | TV cap=3.00 | 1326.66 +/- 0.00 | +nan | 0/0 |
| fshin_actuator_gain | TV cap=3.05 | 1363.47 +/- 0.00 | +nan | 0/0 |
| fshin_actuator_gain | TV cap=3.10 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| fshin_actuator_gain | TV cap=3.20 | 1.00 +/- 0.00 | +nan | 0/0 |
| fshin_actuator_gain | TV cap=3.40 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| fshin_actuator_gain | TV cap=3.70 | 1.00 +/- 0.00 | +nan | 0/0 |
| fshin_damping | TV cap=2.20 | 1.00 +/- 0.00 | +nan | 0/0 |
| fshin_damping | TV cap=2.40 | 1.00 +/- 0.00 | +nan | 0/0 |
| fshin_damping | TV cap=2.55 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| fshin_damping | TV cap=2.65 | 1.00 +/- 0.00 | +nan | 0/0 |
| fshin_damping | TV cap=2.70 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| fshin_damping | TV cap=2.75 | 1.00 +/- 0.00 | +nan | 0/0 |
| fshin_damping | TV cap=2.80 | 1183.23 +/- 0.00 | +29.54 | 11/15 |
| fshin_damping | TV cap=3.00 | 1340.78 +/- 0.00 | +nan | 0/0 |
| fshin_damping | TV cap=3.05 | 1383.36 +/- 0.00 | +nan | 0/0 |
| fshin_damping | TV cap=3.10 | 1484.36 +/- 0.00 | -93.49 | 2/15 |
| fshin_damping | TV cap=3.20 | 1.00 +/- 0.00 | +nan | 0/0 |
| fshin_damping | TV cap=3.40 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| fshin_damping | TV cap=3.70 | 1.00 +/- 0.00 | +nan | 0/0 |
| fshin_friction | TV cap=2.20 | 1.00 +/- 0.00 | +nan | 0/0 |
| fshin_friction | TV cap=2.40 | 1.00 +/- 0.00 | +nan | 0/0 |
| fshin_friction | TV cap=2.55 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| fshin_friction | TV cap=2.65 | 1.00 +/- 0.00 | +nan | 0/0 |
| fshin_friction | TV cap=2.70 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| fshin_friction | TV cap=2.75 | 1.00 +/- 0.00 | +nan | 0/0 |
| fshin_friction | TV cap=2.80 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| fshin_friction | TV cap=3.00 | 1341.69 +/- 0.00 | +nan | 0/0 |
| fshin_friction | TV cap=3.05 | 1372.99 +/- 0.00 | +nan | 0/0 |
| fshin_friction | TV cap=3.10 | 1463.88 +/- 0.00 | +14.53 | 13/15 |
| fshin_friction | TV cap=3.20 | 1.00 +/- 0.00 | +nan | 0/0 |
| fshin_friction | TV cap=3.40 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| fshin_friction | TV cap=3.70 | 1.00 +/- 0.00 | +nan | 0/0 |
| fthigh_actuator_gain | TV cap=2.20 | 1.00 +/- 0.00 | +nan | 0/0 |
| fthigh_actuator_gain | TV cap=2.40 | 1.00 +/- 0.00 | +nan | 0/0 |
| fthigh_actuator_gain | TV cap=2.55 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| fthigh_actuator_gain | TV cap=2.65 | 1.00 +/- 0.00 | +nan | 0/0 |
| fthigh_actuator_gain | TV cap=2.70 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| fthigh_actuator_gain | TV cap=2.75 | 1.00 +/- 0.00 | +nan | 0/0 |
| fthigh_actuator_gain | TV cap=2.80 | 1250.51 +/- 0.00 | -930.60 | 4/15 |
| fthigh_actuator_gain | TV cap=3.00 | 1380.00 +/- 0.00 | +nan | 0/0 |
| fthigh_actuator_gain | TV cap=3.05 | 1373.69 +/- 0.00 | +nan | 0/0 |
| fthigh_actuator_gain | TV cap=3.10 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| fthigh_actuator_gain | TV cap=3.20 | 1.00 +/- 0.00 | +nan | 0/0 |
| fthigh_actuator_gain | TV cap=3.40 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| fthigh_actuator_gain | TV cap=3.70 | 1.00 +/- 0.00 | +nan | 0/0 |
| fthigh_damping | TV cap=2.20 | 1.00 +/- 0.00 | +nan | 0/0 |
| fthigh_damping | TV cap=2.40 | 1.00 +/- 0.00 | +nan | 0/0 |
| fthigh_damping | TV cap=2.55 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| fthigh_damping | TV cap=2.65 | 1.00 +/- 0.00 | +nan | 0/0 |
| fthigh_damping | TV cap=2.70 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| fthigh_damping | TV cap=2.75 | 1.00 +/- 0.00 | +nan | 0/0 |
| fthigh_damping | TV cap=2.80 | 1346.85 +/- 0.00 | -164.58 | 1/15 |
| fthigh_damping | TV cap=3.00 | 1376.14 +/- 0.00 | +nan | 0/0 |
| fthigh_damping | TV cap=3.05 | 1384.96 +/- 0.00 | +nan | 0/0 |
| fthigh_damping | TV cap=3.10 | 1493.83 +/- 0.00 | -123.71 | 0/15 |
| fthigh_damping | TV cap=3.20 | 1.00 +/- 0.00 | +nan | 0/0 |
| fthigh_damping | TV cap=3.40 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| fthigh_damping | TV cap=3.70 | 1.00 +/- 0.00 | +nan | 0/0 |
| fthigh_friction | TV cap=2.20 | 1.00 +/- 0.00 | +nan | 0/0 |
| fthigh_friction | TV cap=2.40 | 1.00 +/- 0.00 | +nan | 0/0 |
| fthigh_friction | TV cap=2.55 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| fthigh_friction | TV cap=2.65 | 1.00 +/- 0.00 | +nan | 0/0 |
| fthigh_friction | TV cap=2.70 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| fthigh_friction | TV cap=2.75 | 1.00 +/- 0.00 | +nan | 0/0 |
| fthigh_friction | TV cap=2.80 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| fthigh_friction | TV cap=3.00 | 1366.01 +/- 0.00 | +nan | 0/0 |
| fthigh_friction | TV cap=3.05 | 1375.94 +/- 0.00 | +nan | 0/0 |
| fthigh_friction | TV cap=3.10 | 1495.55 +/- 0.00 | -17.55 | 0/15 |
| fthigh_friction | TV cap=3.20 | 1.00 +/- 0.00 | +nan | 0/0 |
| fthigh_friction | TV cap=3.40 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| fthigh_friction | TV cap=3.70 | 1.00 +/- 0.00 | +nan | 0/0 |
| gear | TV cap=2.20 | 1.00 +/- 0.00 | +nan | 0/0 |
| gear | TV cap=2.40 | 1.00 +/- 0.00 | +nan | 0/0 |
| gear | TV cap=2.55 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| gear | TV cap=2.65 | 1.00 +/- 0.00 | +nan | 0/0 |
| gear | TV cap=2.70 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| gear | TV cap=2.75 | 1.00 +/- 0.00 | +nan | 0/0 |
| gear | TV cap=2.80 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| gear | TV cap=3.00 | 1381.32 +/- 0.00 | +nan | 0/0 |
| gear | TV cap=3.05 | 1373.91 +/- 0.00 | +nan | 0/0 |
| gear | TV cap=3.10 | 1477.40 +/- 0.00 | -1051.29 | 2/15 |
| gear | TV cap=3.20 | 1.00 +/- 0.00 | +nan | 0/0 |
| gear | TV cap=3.40 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| gear | TV cap=3.70 | 1.00 +/- 0.00 | +nan | 0/0 |
| gravity | TV cap=2.20 | 1.00 +/- 0.00 | +nan | 0/0 |
| gravity | TV cap=2.40 | 1.00 +/- 0.00 | +nan | 0/0 |
| gravity | TV cap=2.55 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| gravity | TV cap=2.65 | 1.00 +/- 0.00 | +nan | 0/0 |
| gravity | TV cap=2.70 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| gravity | TV cap=2.75 | 1.00 +/- 0.00 | +nan | 0/0 |
| gravity | TV cap=2.80 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| gravity | TV cap=3.00 | 896.65 +/- 0.00 | +nan | 0/0 |
| gravity | TV cap=3.05 | 1044.43 +/- 0.00 | +nan | 0/0 |
| gravity | TV cap=3.10 | 868.67 +/- 0.00 | +85.66 | 9/15 |
| gravity | TV cap=3.20 | 1.00 +/- 0.00 | +nan | 0/0 |
| gravity | TV cap=3.40 | 1.00 +/- 0.00 | +0.00 | 0/15 |
| gravity | TV cap=3.70 | 1.00 +/- 0.00 | +nan | 0/0 |

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
- `plots/with_variance/damping_return_curve.png`
- `plots/with_variance/damping_return_curve.pdf`
- `plots/with_variance/damping_gain_curve.png`
- `plots/with_variance/damping_gain_curve.pdf`
- `plots/without_variance/damping_return_curve.png`
- `plots/without_variance/damping_return_curve.pdf`
- `plots/without_variance/damping_gain_curve.png`
- `plots/without_variance/damping_gain_curve.pdf`
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
- `plots/with_variance/friction_return_curve.png`
- `plots/with_variance/friction_return_curve.pdf`
- `plots/with_variance/friction_gain_curve.png`
- `plots/with_variance/friction_gain_curve.pdf`
- `plots/without_variance/friction_return_curve.png`
- `plots/without_variance/friction_return_curve.pdf`
- `plots/without_variance/friction_gain_curve.png`
- `plots/without_variance/friction_gain_curve.pdf`
- `plots/with_variance/friction_damping_return_curve.png`
- `plots/with_variance/friction_damping_return_curve.pdf`
- `plots/with_variance/friction_damping_gain_curve.png`
- `plots/with_variance/friction_damping_gain_curve.pdf`
- `plots/without_variance/friction_damping_return_curve.png`
- `plots/without_variance/friction_damping_return_curve.pdf`
- `plots/without_variance/friction_damping_gain_curve.png`
- `plots/without_variance/friction_damping_gain_curve.pdf`
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

## Output CSV files

- `outputs/eval_metrics_final.csv`: latest merged per-seed eval rows.
- `outputs/summary_by_scenario.csv`: per-model, per-axis, per-factor aggregate return table.
- `outputs/drop_summary.csv`: nominal-minus-perturbed drop table.
- `outputs/gain_summary.csv`: paired robust-gain table against vanilla.
- `outputs/axis_overview.csv`: compact axis-level overview.
- `outputs/curve_points.csv`: same data used for return plots.
- `outputs/gain_curve_points.csv`: same data used for gain plots.
