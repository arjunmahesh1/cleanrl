# Archive

This directory holds historical or generated artifacts that are useful to keep
for provenance but should not sit at the repository root.

## Layout

- `bundles/`: compressed experiment bundles copied or packaged during earlier
  phases of the PPO robustness project.
- `legacy_experiments/`: unpacked early experiment bundles, mostly February and
  March mass/damping/percentile-cap sweeps.
- `cluster_snapshots/`: copied cluster run directories, cluster artifacts, and
  cluster log snapshots.
- `slurm_logs/`: loose root-level `slurm-*.out` files.
- `local_outputs/`: generated local run outputs, W&B cache, videos, temporary
  model runs, generated XML files, and root Python cache.
- `sweeps_backups/`: old root-level `sweeps_backup_*` snapshots.
- `misc_quarantine/`: odd root files that looked like terminal paste artifacts.

Active code, current result bundles, and project handoff documents remain at the
repository root or under `sweeps/results/`.
