# PPO Percentile Fixed-Cap Setup (Phase 1)

Artifacts for percentile-based fixed-cap TV clipping.

## Purpose

Define paper-defensible shared clipping caps from converged vanilla PPO returns before retraining TV models.

Method:
- source runs: `ppo_cont_vanilla_final`
- environment: `Walker2d-v4`
- seeds: `1..5`
- signal: `charts/episodic_return`
- convergence window: last `25%` of training
- candidate caps: `75th`, `85th`, `90th`, `95th` percentile
- pooling: percentile per seed, then median across seeds

## Shared caps

| Percentile | Shared cap | Tag |
| --- | ---: | --- |
| p75 | `4196.3230` | `4196p323` |
| p85 | `4267.1008` | `4267p101` |
| p90 | `4292.8940` | `4292p894` |
| p95 | `4324.8652` | `4324p865` |

## Files

- `shared_percentile_caps.csv`: shared cap table used for training.
- `per_seed_percentile_caps.csv`: per-seed percentile cap values from vanilla runs.
- `phase1_train_manifest.csv`: `(percentile, seed)` training manifest with ready-to-run commands.
- `cap_p75.txt`, `cap_p85.txt`, `cap_p90.txt`, `cap_p95.txt`: single-cap text files for cluster use.
- `shared_caps.env`: shell-friendly exported cap values.

## Phase 1 outcome

Phase 1 retraining + single-axis evaluation completed in:
- `sweeps/results/PPO_PercentileFixedCap_phase1_eval_20260302/README.md`

Selected candidate to carry forward:
- `p95 = 4324.8652`
