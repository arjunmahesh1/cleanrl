# PPO_HalfCheetah_expanded_mass_0p1_2p0_20260430

Date: 2026-05-15

This folder packages the final pinned robustness evaluation for normalized PPO.

## Contents

- Raw metrics: `/Users/arjunmahesh/Library/CloudStorage/OneDrive-Personal/Duke/Research/RL/cleanrl/sweeps/results/PPO_HalfCheetah_final_presentable_20260430/raw_metrics/mass`
- Aggregated outputs: `/Users/arjunmahesh/Library/CloudStorage/OneDrive-Personal/Duke/Research/RL/cleanrl/sweeps/results/PPO_HalfCheetah_expanded_mass_0p1_2p0_20260430/outputs`
- Plots: `/Users/arjunmahesh/Library/CloudStorage/OneDrive-Personal/Duke/Research/RL/cleanrl/sweeps/results/PPO_HalfCheetah_expanded_mass_0p1_2p0_20260430/plots`

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
| bfoot_mass | 1341.65 +/- 0.00 | 1383.03 +/- 0.00 | 1441.52 +/- 0.00 | 1378.86 +/- 0.00 | 1452.70 +/- 0.00 | 1569.26 +/- 0.00 | 1368.33 +/- 0.00 | 1352.16 +/- 0.00 | 1344.37 +/- 0.00 | 1379.14 +/- 0.00 | 1464.38 +/- 0.00 | 1515.81 +/- 0.00 | 1558.01 +/- 0.00 | 1375.30 +/- 0.00 |
| bshin_mass | 1432.74 +/- 0.00 | 1380.04 +/- 0.00 | 1456.23 +/- 0.00 | 1372.19 +/- 0.00 | 1452.97 +/- 0.00 | 1561.06 +/- 0.00 | 1357.08 +/- 0.00 | 1253.74 +/- 0.00 | 1352.75 +/- 0.00 | 1374.67 +/- 0.00 | 1463.08 +/- 0.00 | 1541.74 +/- 0.00 | 1567.98 +/- 0.00 | 1379.81 +/- 0.00 |
| bthigh_mass | 1448.86 +/- 0.00 | 1400.90 +/- 0.00 | 1449.26 +/- 0.00 | 1384.50 +/- 0.00 | 1460.44 +/- 0.00 | 1542.51 +/- 0.00 | 1361.19 +/- 0.00 | 1351.81 +/- 0.00 | 1372.52 +/- 0.00 | 1382.55 +/- 0.00 | 1486.62 +/- 0.00 | 1540.68 +/- 0.00 | 1545.03 +/- 0.00 | 1392.78 +/- 0.00 |
| ffoot_mass | 1442.89 +/- 0.00 | 1376.94 +/- 0.00 | 1456.78 +/- 0.00 | 1384.62 +/- 0.00 | 1459.64 +/- 0.00 | 1548.85 +/- 0.00 | 1377.41 +/- 0.00 | 1364.49 +/- 0.00 | 1363.35 +/- 0.00 | 1378.16 +/- 0.00 | 1461.67 +/- 0.00 | 1510.33 +/- 0.00 | 1561.60 +/- 0.00 | 1373.67 +/- 0.00 |
| friction_mass | 1356.89 +/- 0.00 | 1370.76 +/- 0.00 | 1457.16 +/- 0.00 | 1277.03 +/- 0.00 | 1460.54 +/- 0.00 | 1554.80 +/- 0.00 | 1339.14 +/- 0.00 | 1271.02 +/- 0.00 | 1355.38 +/- 0.00 | 1373.36 +/- 0.00 | 1480.87 +/- 0.00 | 1545.98 +/- 0.00 | 1577.77 +/- 0.00 | 1378.37 +/- 0.00 |
| friction_mass_damping | 1406.87 +/- 0.00 | 1394.22 +/- 0.00 | 1459.52 +/- 0.00 | 1376.96 +/- 0.00 | 1460.67 +/- 0.00 | 1569.28 +/- 0.00 | 1352.69 +/- 0.00 | 1261.41 +/- 0.00 | 1333.75 +/- 0.00 | 1383.78 +/- 0.00 | 1479.17 +/- 0.00 | 1517.69 +/- 0.00 | 1558.40 +/- 0.00 | 1370.88 +/- 0.00 |
| fshin_mass | 1435.55 +/- 0.00 | 1313.13 +/- 0.00 | 1444.78 +/- 0.00 | 1366.55 +/- 0.00 | 1464.39 +/- 0.00 | 1552.38 +/- 0.00 | 1380.58 +/- 0.00 | 1362.41 +/- 0.00 | 1330.63 +/- 0.00 | 1376.12 +/- 0.00 | 1475.75 +/- 0.00 | 1520.24 +/- 0.00 | 1576.86 +/- 0.00 | 1396.98 +/- 0.00 |
| fthigh_mass | 1434.41 +/- 0.00 | 1384.73 +/- 0.00 | 1444.70 +/- 0.00 | 1373.05 +/- 0.00 | 1460.54 +/- 0.00 | 1575.35 +/- 0.00 | 1380.56 +/- 0.00 | 1252.41 +/- 0.00 | 1321.46 +/- 0.00 | 1376.93 +/- 0.00 | 1485.43 +/- 0.00 | 1523.36 +/- 0.00 | 1553.54 +/- 0.00 | 1390.76 +/- 0.00 |
| mass | 1342.49 +/- 0.00 | 1390.85 +/- 0.00 | 1430.71 +/- 0.00 | 1273.95 +/- 0.00 | 1469.50 +/- 0.00 | 1576.67 +/- 0.00 | 1371.79 +/- 0.00 | 1261.61 +/- 0.00 | 1366.42 +/- 0.00 | 1373.33 +/- 0.00 | 1467.84 +/- 0.00 | 1538.67 +/- 0.00 | 1559.62 +/- 0.00 | 1377.17 +/- 0.00 |
| mass_damping | 1435.19 +/- 0.00 | 1381.88 +/- 0.00 | 1435.51 +/- 0.00 | 1297.09 +/- 0.00 | 1469.37 +/- 0.00 | 1563.47 +/- 0.00 | 1366.86 +/- 0.00 | 1269.10 +/- 0.00 | 1398.23 +/- 0.00 | 1380.36 +/- 0.00 | 1484.16 +/- 0.00 | 1521.10 +/- 0.00 | 1550.61 +/- 0.00 | 1387.98 +/- 0.00 |

## Axis overview

| Axis | Model | Nominal return | Mean gain over perturbed scenarios | Positive gain scenarios |
| --- | --- | --- | --- | --- |
| bfoot_mass | TV cap=2.20 | 1383.03 +/- 0.00 | +nan | 0/0 |
| bfoot_mass | TV cap=2.40 | 1441.52 +/- 0.00 | +nan | 0/0 |
| bfoot_mass | TV cap=2.55 | 1378.86 +/- 0.00 | -95.29 | 1/14 |
| bfoot_mass | TV cap=2.65 | 1452.70 +/- 0.00 | +nan | 0/0 |
| bfoot_mass | TV cap=2.70 | 1569.26 +/- 0.00 | -82.63 | 1/14 |
| bfoot_mass | TV cap=2.75 | 1368.33 +/- 0.00 | +nan | 0/0 |
| bfoot_mass | TV cap=2.80 | 1352.16 +/- 0.00 | -112.71 | 1/14 |
| bfoot_mass | TV cap=3.00 | 1344.37 +/- 0.00 | +nan | 0/0 |
| bfoot_mass | TV cap=3.05 | 1379.14 +/- 0.00 | +nan | 0/0 |
| bfoot_mass | TV cap=3.10 | 1464.38 +/- 0.00 | -57.81 | 2/14 |
| bfoot_mass | TV cap=3.20 | 1515.81 +/- 0.00 | +nan | 0/0 |
| bfoot_mass | TV cap=3.40 | 1558.01 +/- 0.00 | -65.51 | 2/14 |
| bfoot_mass | TV cap=3.70 | 1375.30 +/- 0.00 | +nan | 0/0 |
| bshin_mass | TV cap=2.20 | 1380.04 +/- 0.00 | +nan | 0/0 |
| bshin_mass | TV cap=2.40 | 1456.23 +/- 0.00 | +nan | 0/0 |
| bshin_mass | TV cap=2.55 | 1372.19 +/- 0.00 | +6.30 | 11/14 |
| bshin_mass | TV cap=2.65 | 1452.97 +/- 0.00 | +nan | 0/0 |
| bshin_mass | TV cap=2.70 | 1561.06 +/- 0.00 | +9.90 | 6/14 |
| bshin_mass | TV cap=2.75 | 1357.08 +/- 0.00 | +nan | 0/0 |
| bshin_mass | TV cap=2.80 | 1253.74 +/- 0.00 | +74.09 | 14/14 |
| bshin_mass | TV cap=3.00 | 1352.75 +/- 0.00 | +nan | 0/0 |
| bshin_mass | TV cap=3.05 | 1374.67 +/- 0.00 | +nan | 0/0 |
| bshin_mass | TV cap=3.10 | 1463.08 +/- 0.00 | +36.60 | 13/14 |
| bshin_mass | TV cap=3.20 | 1541.74 +/- 0.00 | +nan | 0/0 |
| bshin_mass | TV cap=3.40 | 1567.98 +/- 0.00 | +11.29 | 7/14 |
| bshin_mass | TV cap=3.70 | 1379.81 +/- 0.00 | +nan | 0/0 |
| bthigh_mass | TV cap=2.20 | 1400.90 +/- 0.00 | +nan | 0/0 |
| bthigh_mass | TV cap=2.40 | 1449.26 +/- 0.00 | +nan | 0/0 |
| bthigh_mass | TV cap=2.55 | 1384.50 +/- 0.00 | -13.89 | 6/14 |
| bthigh_mass | TV cap=2.65 | 1460.44 +/- 0.00 | +nan | 0/0 |
| bthigh_mass | TV cap=2.70 | 1542.51 +/- 0.00 | +53.23 | 14/14 |
| bthigh_mass | TV cap=2.75 | 1361.19 +/- 0.00 | +nan | 0/0 |
| bthigh_mass | TV cap=2.80 | 1351.81 +/- 0.00 | -37.69 | 7/14 |
| bthigh_mass | TV cap=3.00 | 1372.52 +/- 0.00 | +nan | 0/0 |
| bthigh_mass | TV cap=3.05 | 1382.55 +/- 0.00 | +nan | 0/0 |
| bthigh_mass | TV cap=3.10 | 1486.62 +/- 0.00 | +29.33 | 12/14 |
| bthigh_mass | TV cap=3.20 | 1540.68 +/- 0.00 | +nan | 0/0 |
| bthigh_mass | TV cap=3.40 | 1545.03 +/- 0.00 | +51.73 | 14/14 |
| bthigh_mass | TV cap=3.70 | 1392.78 +/- 0.00 | +nan | 0/0 |
| ffoot_mass | TV cap=2.20 | 1376.94 +/- 0.00 | +nan | 0/0 |
| ffoot_mass | TV cap=2.40 | 1456.78 +/- 0.00 | +nan | 0/0 |
| ffoot_mass | TV cap=2.55 | 1384.62 +/- 0.00 | -27.50 | 4/14 |
| ffoot_mass | TV cap=2.65 | 1459.64 +/- 0.00 | +nan | 0/0 |
| ffoot_mass | TV cap=2.70 | 1548.85 +/- 0.00 | +45.99 | 14/14 |
| ffoot_mass | TV cap=2.75 | 1377.41 +/- 0.00 | +nan | 0/0 |
| ffoot_mass | TV cap=2.80 | 1364.49 +/- 0.00 | -45.94 | 2/14 |
| ffoot_mass | TV cap=3.00 | 1363.35 +/- 0.00 | +nan | 0/0 |
| ffoot_mass | TV cap=3.05 | 1378.16 +/- 0.00 | +nan | 0/0 |
| ffoot_mass | TV cap=3.10 | 1461.67 +/- 0.00 | +44.43 | 12/14 |
| ffoot_mass | TV cap=3.20 | 1510.33 +/- 0.00 | +nan | 0/0 |
| ffoot_mass | TV cap=3.40 | 1561.60 +/- 0.00 | +36.31 | 12/14 |
| ffoot_mass | TV cap=3.70 | 1373.67 +/- 0.00 | +nan | 0/0 |
| friction_mass | TV cap=2.20 | 1370.76 +/- 0.00 | +nan | 0/0 |
| friction_mass | TV cap=2.40 | 1457.16 +/- 0.00 | +nan | 0/0 |
| friction_mass | TV cap=2.55 | 1277.03 +/- 0.00 | -97.23 | 6/14 |
| friction_mass | TV cap=2.65 | 1460.54 +/- 0.00 | +nan | 0/0 |
| friction_mass | TV cap=2.70 | 1554.80 +/- 0.00 | -39.94 | 7/14 |
| friction_mass | TV cap=2.75 | 1339.14 +/- 0.00 | +nan | 0/0 |
| friction_mass | TV cap=2.80 | 1271.02 +/- 0.00 | -245.30 | 3/14 |
| friction_mass | TV cap=3.00 | 1355.38 +/- 0.00 | +nan | 0/0 |
| friction_mass | TV cap=3.05 | 1373.36 +/- 0.00 | +nan | 0/0 |
| friction_mass | TV cap=3.10 | 1480.87 +/- 0.00 | -5.45 | 4/14 |
| friction_mass | TV cap=3.20 | 1545.98 +/- 0.00 | +nan | 0/0 |
| friction_mass | TV cap=3.40 | 1577.77 +/- 0.00 | -176.55 | 4/14 |
| friction_mass | TV cap=3.70 | 1378.37 +/- 0.00 | +nan | 0/0 |
| friction_mass_damping | TV cap=2.20 | 1394.22 +/- 0.00 | +nan | 0/0 |
| friction_mass_damping | TV cap=2.40 | 1459.52 +/- 0.00 | +nan | 0/0 |
| friction_mass_damping | TV cap=2.55 | 1376.96 +/- 0.00 | -490.99 | 7/14 |
| friction_mass_damping | TV cap=2.65 | 1460.67 +/- 0.00 | +nan | 0/0 |
| friction_mass_damping | TV cap=2.70 | 1569.28 +/- 0.00 | -249.52 | 4/14 |
| friction_mass_damping | TV cap=2.75 | 1352.69 +/- 0.00 | +nan | 0/0 |
| friction_mass_damping | TV cap=2.80 | 1261.41 +/- 0.00 | -270.51 | 8/14 |
| friction_mass_damping | TV cap=3.00 | 1333.75 +/- 0.00 | +nan | 0/0 |
| friction_mass_damping | TV cap=3.05 | 1383.78 +/- 0.00 | +nan | 0/0 |
| friction_mass_damping | TV cap=3.10 | 1479.17 +/- 0.00 | -60.01 | 9/14 |
| friction_mass_damping | TV cap=3.20 | 1517.69 +/- 0.00 | +nan | 0/0 |
| friction_mass_damping | TV cap=3.40 | 1558.40 +/- 0.00 | -261.53 | 6/14 |
| friction_mass_damping | TV cap=3.70 | 1370.88 +/- 0.00 | +nan | 0/0 |
| fshin_mass | TV cap=2.20 | 1313.13 +/- 0.00 | +nan | 0/0 |
| fshin_mass | TV cap=2.40 | 1444.78 +/- 0.00 | +nan | 0/0 |
| fshin_mass | TV cap=2.55 | 1366.55 +/- 0.00 | +3.95 | 9/14 |
| fshin_mass | TV cap=2.65 | 1464.39 +/- 0.00 | +nan | 0/0 |
| fshin_mass | TV cap=2.70 | 1552.38 +/- 0.00 | +24.08 | 11/14 |
| fshin_mass | TV cap=2.75 | 1380.58 +/- 0.00 | +nan | 0/0 |
| fshin_mass | TV cap=2.80 | 1362.41 +/- 0.00 | -23.42 | 4/14 |
| fshin_mass | TV cap=3.00 | 1330.63 +/- 0.00 | +nan | 0/0 |
| fshin_mass | TV cap=3.05 | 1376.12 +/- 0.00 | +nan | 0/0 |
| fshin_mass | TV cap=3.10 | 1475.75 +/- 0.00 | +12.99 | 9/14 |
| fshin_mass | TV cap=3.20 | 1520.24 +/- 0.00 | +nan | 0/0 |
| fshin_mass | TV cap=3.40 | 1576.86 +/- 0.00 | -0.25 | 4/14 |
| fshin_mass | TV cap=3.70 | 1396.98 +/- 0.00 | +nan | 0/0 |
| fthigh_mass | TV cap=2.20 | 1384.73 +/- 0.00 | +nan | 0/0 |
| fthigh_mass | TV cap=2.40 | 1444.70 +/- 0.00 | +nan | 0/0 |
| fthigh_mass | TV cap=2.55 | 1373.05 +/- 0.00 | -9.06 | 7/14 |
| fthigh_mass | TV cap=2.65 | 1460.54 +/- 0.00 | +nan | 0/0 |
| fthigh_mass | TV cap=2.70 | 1575.35 +/- 0.00 | -0.04 | 3/14 |
| fthigh_mass | TV cap=2.75 | 1380.56 +/- 0.00 | +nan | 0/0 |
| fthigh_mass | TV cap=2.80 | 1252.41 +/- 0.00 | +58.04 | 12/14 |
| fthigh_mass | TV cap=3.00 | 1321.46 +/- 0.00 | +nan | 0/0 |
| fthigh_mass | TV cap=3.05 | 1376.93 +/- 0.00 | +nan | 0/0 |
| fthigh_mass | TV cap=3.10 | 1485.43 +/- 0.00 | +13.13 | 8/14 |
| fthigh_mass | TV cap=3.20 | 1523.36 +/- 0.00 | +nan | 0/0 |
| fthigh_mass | TV cap=3.40 | 1553.54 +/- 0.00 | +24.72 | 10/14 |
| fthigh_mass | TV cap=3.70 | 1390.76 +/- 0.00 | +nan | 0/0 |
| mass | TV cap=2.20 | 1390.85 +/- 0.00 | +nan | 0/0 |
| mass | TV cap=2.40 | 1430.71 +/- 0.00 | +nan | 0/0 |
| mass | TV cap=2.55 | 1273.95 +/- 0.00 | -71.74 | 5/14 |
| mass | TV cap=2.65 | 1469.50 +/- 0.00 | +nan | 0/0 |
| mass | TV cap=2.70 | 1576.67 +/- 0.00 | -197.88 | 1/14 |
| mass | TV cap=2.75 | 1371.79 +/- 0.00 | +nan | 0/0 |
| mass | TV cap=2.80 | 1261.61 +/- 0.00 | -190.95 | 3/14 |
| mass | TV cap=3.00 | 1366.42 +/- 0.00 | +nan | 0/0 |
| mass | TV cap=3.05 | 1373.33 +/- 0.00 | +nan | 0/0 |
| mass | TV cap=3.10 | 1467.84 +/- 0.00 | -82.84 | 4/14 |
| mass | TV cap=3.20 | 1538.67 +/- 0.00 | +nan | 0/0 |
| mass | TV cap=3.40 | 1559.62 +/- 0.00 | -231.02 | 1/14 |
| mass | TV cap=3.70 | 1377.17 +/- 0.00 | +nan | 0/0 |
| mass_damping | TV cap=2.20 | 1381.88 +/- 0.00 | +nan | 0/0 |
| mass_damping | TV cap=2.40 | 1435.51 +/- 0.00 | +nan | 0/0 |
| mass_damping | TV cap=2.55 | 1297.09 +/- 0.00 | -18.25 | 8/14 |
| mass_damping | TV cap=2.65 | 1469.37 +/- 0.00 | +nan | 0/0 |
| mass_damping | TV cap=2.70 | 1563.47 +/- 0.00 | -200.93 | 4/14 |
| mass_damping | TV cap=2.75 | 1366.86 +/- 0.00 | +nan | 0/0 |
| mass_damping | TV cap=2.80 | 1269.10 +/- 0.00 | -181.91 | 7/14 |
| mass_damping | TV cap=3.00 | 1398.23 +/- 0.00 | +nan | 0/0 |
| mass_damping | TV cap=3.05 | 1380.36 +/- 0.00 | +nan | 0/0 |
| mass_damping | TV cap=3.10 | 1484.16 +/- 0.00 | -5.78 | 9/14 |
| mass_damping | TV cap=3.20 | 1521.10 +/- 0.00 | +nan | 0/0 |
| mass_damping | TV cap=3.40 | 1550.61 +/- 0.00 | -247.73 | 5/14 |
| mass_damping | TV cap=3.70 | 1387.98 +/- 0.00 | +nan | 0/0 |

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
- `plots/with_variance/bfoot_mass_return_curve.png`
- `plots/with_variance/bfoot_mass_return_curve.pdf`
- `plots/with_variance/bfoot_mass_gain_curve.png`
- `plots/with_variance/bfoot_mass_gain_curve.pdf`
- `plots/without_variance/bfoot_mass_return_curve.png`
- `plots/without_variance/bfoot_mass_return_curve.pdf`
- `plots/without_variance/bfoot_mass_gain_curve.png`
- `plots/without_variance/bfoot_mass_gain_curve.pdf`
- `plots/with_variance/bshin_mass_return_curve.png`
- `plots/with_variance/bshin_mass_return_curve.pdf`
- `plots/with_variance/bshin_mass_gain_curve.png`
- `plots/with_variance/bshin_mass_gain_curve.pdf`
- `plots/without_variance/bshin_mass_return_curve.png`
- `plots/without_variance/bshin_mass_return_curve.pdf`
- `plots/without_variance/bshin_mass_gain_curve.png`
- `plots/without_variance/bshin_mass_gain_curve.pdf`
- `plots/with_variance/bthigh_mass_return_curve.png`
- `plots/with_variance/bthigh_mass_return_curve.pdf`
- `plots/with_variance/bthigh_mass_gain_curve.png`
- `plots/with_variance/bthigh_mass_gain_curve.pdf`
- `plots/without_variance/bthigh_mass_return_curve.png`
- `plots/without_variance/bthigh_mass_return_curve.pdf`
- `plots/without_variance/bthigh_mass_gain_curve.png`
- `plots/without_variance/bthigh_mass_gain_curve.pdf`
- `plots/with_variance/ffoot_mass_return_curve.png`
- `plots/with_variance/ffoot_mass_return_curve.pdf`
- `plots/with_variance/ffoot_mass_gain_curve.png`
- `plots/with_variance/ffoot_mass_gain_curve.pdf`
- `plots/without_variance/ffoot_mass_return_curve.png`
- `plots/without_variance/ffoot_mass_return_curve.pdf`
- `plots/without_variance/ffoot_mass_gain_curve.png`
- `plots/without_variance/ffoot_mass_gain_curve.pdf`
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
- `plots/with_variance/fshin_mass_return_curve.png`
- `plots/with_variance/fshin_mass_return_curve.pdf`
- `plots/with_variance/fshin_mass_gain_curve.png`
- `plots/with_variance/fshin_mass_gain_curve.pdf`
- `plots/without_variance/fshin_mass_return_curve.png`
- `plots/without_variance/fshin_mass_return_curve.pdf`
- `plots/without_variance/fshin_mass_gain_curve.png`
- `plots/without_variance/fshin_mass_gain_curve.pdf`
- `plots/with_variance/fthigh_mass_return_curve.png`
- `plots/with_variance/fthigh_mass_return_curve.pdf`
- `plots/with_variance/fthigh_mass_gain_curve.png`
- `plots/with_variance/fthigh_mass_gain_curve.pdf`
- `plots/without_variance/fthigh_mass_return_curve.png`
- `plots/without_variance/fthigh_mass_return_curve.pdf`
- `plots/without_variance/fthigh_mass_gain_curve.png`
- `plots/without_variance/fthigh_mass_gain_curve.pdf`
- `plots/with_variance/mass_return_curve.png`
- `plots/with_variance/mass_return_curve.pdf`
- `plots/with_variance/mass_gain_curve.png`
- `plots/with_variance/mass_gain_curve.pdf`
- `plots/without_variance/mass_return_curve.png`
- `plots/without_variance/mass_return_curve.pdf`
- `plots/without_variance/mass_gain_curve.png`
- `plots/without_variance/mass_gain_curve.pdf`
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
