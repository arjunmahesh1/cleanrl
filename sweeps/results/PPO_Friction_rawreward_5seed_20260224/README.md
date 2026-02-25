# PPO Friction Robustness (Raw Reward, 5 Seeds)

This folder contains the final corrected evaluation snapshot for the PPO friction robustness run.

## Experiment configuration
- Environment: `Walker2d-v4`
- Models: `vanilla`, `tv` (fixed cap from 200k AUC selection)
- Selected TV cap: `3500`
- Seeds: `1..5`
- Evaluation episodes per condition: `100`
- Perturbation grid:
  - `xml_body_mass_scale = 1.0`
  - `xml_joint_damping_scale = 1.0`
  - `xml_geom_friction_scale in {1.0, 1.1, 1.2, 1.3}`
- Reward reporting: raw rewards (`eval_raw_rewards=1`)

## Folder contents
- `inputs/selected_cap_from_auc_200k.txt`: selected cap used for TV final model.
- `outputs/eval_metrics_rawreward_friction_only.csv`: merged 50-row eval table.
- `outputs/summary_rawreward_friction_only.csv`: aggregated summary (mean/median/IQM + CIs).
- `outputs/gain_stats_rawreward_friction_only.csv`: robust gain table.
- `outputs/per_seed/seed_*.csv`: per-seed eval rows (10 rows per seed).

## Quick headline numbers (mean +/- 95% CI)
- Nominal: TV `3137.24 +/- 683.34`, Vanilla `3155.19 +/- 730.20`
- Friction 1.0: TV `3153.78 +/- 804.18`, Vanilla `3049.49 +/- 707.00`
- Friction 1.1: TV `3162.17 +/- 757.72`, Vanilla `3225.74 +/- 841.56`
- Friction 1.2: TV `2732.18 +/- 585.37`, Vanilla `2700.01 +/- 573.32`
- Friction 1.3: TV `2400.87 +/- 751.03`, Vanilla `2317.54 +/- 723.63`

## Robust gain (vanilla drop - TV drop)
- 1.0: mean `+122.25 +/- 341.88`, median `-21.70 +/- 267.14`, IQM `+74.19 +/- 239.27`
- 1.1: mean `-45.61 +/- 247.91`, median `-53.11 +/- 377.98`, IQM `-49.05 +/- 87.66`
- 1.2: mean `+50.12 +/- 224.70`, median `-134.45 +/- 279.15`, IQM `-32.85 +/- 114.72`
- 1.3: mean `+101.28 +/- 134.18`, median `-111.19 +/- 225.54`, IQM `+106.19 +/- 42.14`