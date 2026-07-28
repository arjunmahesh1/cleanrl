# Seed-Level Analysis Plots

Source result directory: `/home/users/am1015/cleanrl/sweeps/results/TD3_TVCap_Walker_v5_full30_20260722_walker_v5_full30`

Outputs:
- `seed_spaghetti_by_axis/`: one figure per environment/axis. Each model panel shows all seed return curves over perturbation level, plus a thick median curve.
- `fixed_seed_all_caps/`: one figure per environment/seed. Each panel is a selected perturbation axis, with all models shown together.
- `seed_scatter/`: one-point-per-seed scatter plots at a stress factor, both raw return and same-seed vanilla-subtracted return.
- `reliability_curves/`: reliability survival curves, `P(return >= threshold)`, where threshold is normalized by vanilla nominal median return.
- `seed_conditioned_effect/`: vanilla nominal quality versus robust curve-average gain, shown both ex post over the cap menu and for one preselected cap.
- `stress_scenario_summary.csv`: model-level stress-factor summary with reliability/failure/win-rate statistics.
- `best_caps_by_axis.csv`: best model by median stress-factor return for each axis.
- `best_caps_by_reliability_auc.csv`: best model by area under the reliability curve between 0 and 1x vanilla nominal median return.
- `per_seed_best_caps.csv`: per-seed best model and number of robust models beating same-seed vanilla at the stress factor.

Default catastrophe threshold fraction: `0.5`.
Scatter/reliability axis mode: `all axes`.
Requested fixed model for seed-conditioned analysis: `tvc400`.

ClipFraction note:
`robust/tv_return_clip_fraction` is logged during training, not evaluation. These eval CSVs are enough for reliability and seed-return analyses, but ClipFraction trajectory/scatter plots require the training TensorBoard event files or exported W&B scalars.
