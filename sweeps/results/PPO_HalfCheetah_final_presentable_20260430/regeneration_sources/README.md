# Regeneration Sources

This directory stores raw-metric snapshots copied from the packaged source experiment folders used to build the presentation bundle.

Each subfolder contains a `raw_metrics/` directory that can be fed back into `sweeps/package_alpha_robust_eval.py` to regenerate category-level packaged plots.

Snapshots included:
- `nonmass` <- `sweeps/results/PPO_HalfCheetah_expanded_nonmass_0p0_2p0_20260430`
- `mass` <- `sweeps/results/PPO_HalfCheetah_expanded_mass_0p1_2p0_20260430`
- `signal` <- `sweeps/results/PPO_HalfCheetah_signal_all_caps_20260430`
