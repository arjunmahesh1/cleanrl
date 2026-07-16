# Walker2d TD3 TV-Cap Full 30-Seed Evaluation

- Models: vanilla, tvc100, tvc150, tvc200, tvc225, tvc250, tvc275, tvc300
- Seeds: 1--30
- Evaluation episodes per row: 20
- Evaluation rows: 71520
- Fixed deployment-style cap in seed-conditioned analysis: tvc250
- Variance whiskers are disabled in category return/gain figures.

Layout:
- `raw_metrics/shards/`: chunk-level source CSVs.
- `outputs/combined_metrics.csv`: validated combined evaluation table.
- `Walker2d/`: PPO-style category folders with raw metrics, tables, and plots.
- `analysis_plots/`: seed spaghetti, fixed-seed cap comparisons, seed scatter, reliability/AUC, and seed-conditioned gain analyses.

Paper-facing additions:
- `PAPER_INTERPRETATION.md`: main findings, limitations, and proposed next experiment.
- `analysis_plots/paper_figures/cap_activity_dynamics.pdf`: cap-activity phase transition used in the TD3 paper section.
