# Expanded 30-seed PPO confirmation

This folder contains expanded 30-seed confirmatory PPO robustness results for Walker2d and HalfCheetah.

Each environment includes:
- `outputs/combined_metrics.csv`: all evaluations merged
- `outputs/summary_by_model.csv`: aggregate return by perturbation and cap
- `outputs/paired_gain_over_vanilla.csv`: paired seed-level gain summaries
- `plots/paired_seed_gains/`: per-seed gain-over-vanilla plots

Interpretation:
The main diagnostic is paired seed gain over vanilla. A cap is convincing only if its gains are mostly positive across seeds, not merely positive on average due to a few favorable seeds.
