# PPO Full 30-Seed All-Caps No-Noop - 2026-05-24

Full 30-seed PPO robustness evaluation for Walker2d-v4 and HalfCheetah-v4,
excluding `noop`/`a1e9` controls.

## Contents

- `outputs/combined_metrics.csv`: combined eval metrics from all shard CSVs.
- `raw_metrics/shards/`: raw eval shard CSVs copied from the cluster run.
- `manifests/`: eval manifests used for the global array.
- `Walker2d/`: category-level packaged Walker2d outputs.
- `HalfCheetah/`: category-level packaged HalfCheetah outputs.
- `analysis_plots/`: seed-level and reliability plots built from
  `outputs/combined_metrics.csv`.

## Category Folders

Each environment has:

- `single_axis_perturbations/`
- `targeted_localized_perturbations/`
- `combos/`
- `gaussian_action_noise/`
- `bernoulli_action_noise/`

HalfCheetah also has:

- `observation_noise/`

Within each category:

- `raw_metrics/metrics.csv`: raw metrics for that category.
- `outputs/`: packaged summary CSVs.
- `plots/without_variance/`: PNG/PDF return and gain curves without variance
  whiskers.

## Row Count

- Total rows: `320520`
- Walker2d-v4 rows: `89100`
- HalfCheetah-v4 rows: `231420`
- Seeds: `1..30`
- No `noop` or `a1e9` rows.

## Analysis Plots

Generated with:

```bash
python sweeps/plot_full30_seed_reliability.py \
  --result-dir sweeps/results/PPO_full30_allcaps_no_noop_20260524
```

See `analysis_plots/README.md` for the seed-line, seed-scatter, and
reliability-curve layout.

ClipFraction trajectory plots require training TensorBoard/W&B scalars, because
`robust/tv_return_clip_fraction` is logged during training rather than eval.
