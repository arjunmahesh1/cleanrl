# PPO Alpha Grid — Normalized Coarse Sweep

**Date:** 2026-03-26
**WandB group:** `ppo-alpha-norm-coarse`
**Project:** `fixed-alpha-randomness`

## Setup

| Parameter | Value |
|---|---|
| Environment | Walker2d-v4 |
| Seeds | 1 2 3 4 5 |
| Total timesteps | 1,000,000 |
| NormalizeReward | True |
| NormalizeObservation | True |
| Hardware | RTX 2080 Ti (linux[51-60]) |
| Determinism | torch_deterministic=True, enforce_full_determinism=True, CUBLAS_WORKSPACE_CONFIG=:4096:8, PYTHONHASHSEED=seed |

## Variants

| Variant | tv_fixed_cap | Description |
|---|---|---|
| vanilla | — | Standard PPO, no clipping |
| noop | 1e9 | Clipping enabled but cap unreachably high — confirms no-op |
| 2.85 | 2.85 | Aggressive: below median p95 of value targets |
| 2.95 | 2.95 | Moderate: near median p99 |
| 3.05 | 3.05 | Conservative: above median max |
| 3.20 | 3.20 | Very conservative |
| 3.50 | 3.50 | Near no-op at convergence — tests early-training-only clipping |

## Key Findings

### Nominal performance (IQM across 5 seeds)

| Variant | IQM return | vs vanilla |
|---|---|---|
| vanilla | 3628 | — |
| noop | 3628 | 0% (identical — confirms noop works) |
| 3.05 | 3238 | -11% |
| 2.95 | 2944 | -19% |
| 3.50 | 2378 | -34% |
| 3.20 | 1842 | -49% |
| 2.85 | 1431 | -61% |

### Clipping activity (IQM clip_fraction)

| Variant | IQM clip_fraction | Activates? |
|---|---|---|
| noop | 0.000 | No — confirmed no-op |
| 3.50 | 0.000 | No — too loose |
| 2.95 | 0.013 | Yes — lightly |
| 3.05 | 0.011 | Yes — lightly |
| 3.20 | 0.065 | Yes — but erratic (bimodal across seeds) |
| 2.85 | 0.097 | Yes — aggressively |

### Alpha selection

**3.05 is the best candidate:** clips lightly (IQM=1.1%), nominal cost only -11%, and value learning is meaningfully different from vanilla (EV 0.856 vs 0.680).

**2.95** is the second candidate: clips slightly less, -19% nominal cost.

**3.20 and 3.50** are disqualified: 3.20 has erratic clip_fraction across seeds (0 for 3 seeds, 0.19-0.21 for 2); 3.50 never clips at convergence.

**2.85** is too aggressive: -61% nominal cost.

## Files

```
outputs/
  alpha_grid_summary.txt   — full tables from extract_alpha_results.py
  alpha_grid_summary.png   — bar chart (return / clip_fraction / explained_variance)
inputs/
  (sbatch command logged below)
```

## Launch command

```bash
EXPORTS="ALL,PROJECT=fixed-alpha-randomness,GROUP=ppo-alpha-norm-coarse,\
RUN_DIR=$HOME/rl_runs_alpha_grid_norm_coarse,\
ALPHA_VALUES_STR=vanilla noop 2.85 2.95 3.05 3.20 3.50,\
SEED_VALUES_STR=1 2 3 4 5,TOTAL_TIMESTEPS=1000000,NORMALIZE_REWARD=true"

sbatch -p compsci-gpu --gres=gpu:1 --constraint=rtx_2080 --exclude=linux[41-55] \
  --mem=32G --cpus-per-task=8 --time=08:00:00 \
  --array=0-34%5 -J ppo_alpha_coarse \
  -o ~/slurm_logs/ppo_alpha_coarse_%A_%a.out \
  --export="$EXPORTS" \
  slurm/train_ppo_alpha_grid.sh
```

## Next steps

1. Run robustness evals (friction / mass / damping) for vanilla + 3.05 + 2.95
2. Compare degradation under perturbation between vanilla and tv variants
3. Select final alpha based on robustness gain vs nominal cost tradeoff
