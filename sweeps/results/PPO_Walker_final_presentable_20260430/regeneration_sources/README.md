# Regeneration Sources

This directory stores raw-metric snapshots copied from the packaged source experiment folders used to build the presentation bundle.

Each subfolder contains a `raw_metrics/` directory that can be fed back into `sweeps/package_alpha_robust_eval.py` to regenerate category-level packaged plots.

Snapshots included:
- `single_axis_nonmass` <- `sweeps/results/PPO_Walker_single_axis_physical_nonmass_0p0_2p0_20260430`
- `single_axis_mass` <- `sweeps/results/PPO_Walker_single_axis_mass_0p1_2p0_20260430`
- `targeted_nonmass` <- `sweeps/results/PPO_Walker_targeted_nonmass_0p0_2p0_20260430`
- `targeted_mass` <- `sweeps/results/PPO_Walker_targeted_mass_0p1_2p0_20260430`
- `combo_nonmass` <- `sweeps/results/PPO_Walker_fmd_nonmass_0p0_2p0_20260430`
- `combo_mass` <- `sweeps/results/PPO_Walker_fmd_mass_0p1_2p0_20260430`
- `gaussian` <- `sweeps/results/PPO_Adversarial_action_noise_20260406_repack_20260430`
- `bernoulli` <- `sweeps/results/PPO_Adversarial_action_bernoulli_20260408_clean_v3_repack_20260430`
