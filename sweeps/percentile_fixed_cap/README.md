# Percentile-Based Fixed-Cap TV Clipping

## Goal

Replace the current AUC-selected absolute cap with a more defensible shared cap derived from converged vanilla PPO returns:

- source: vanilla PPO training returns,
- window: last 25% of training,
- candidates: 75th / 85th / 90th / 95th percentiles,
- pooling: percentile per seed, then median across seeds,
- output: one shared cap per percentile.

## Current shared caps

- `p75 = 4196.3230`
- `p85 = 4267.1008`
- `p90 = 4292.8940`
- `p95 = 4324.8652`
