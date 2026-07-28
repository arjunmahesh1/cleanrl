# Walker2d TD3 TV-Cap 30-Seed Evaluation

- Environment: Walker2d-v5
- Method: TD3 TV-Cap
- Models: vanilla, tvc100, tvc150, tvc200, tvc225, tvc250, tvc275, tvc300, tvc400, tvc500
- Seeds: 1--30
- Evaluation episodes per row: 20
- Evaluation rows: 89400
- Total evaluation episodes: 1788000
- Fixed deployment-style model in seed-conditioned analysis: tvc400
- Variance whiskers are disabled in category return/gain figures.

Layout:
- `raw_metrics/shards/`: chunk-level source CSVs.
- `outputs/combined_metrics.csv`: validated combined evaluation table.
- `Walker2d/`: PPO-style category folders with raw metrics, tables, and plots.
- `analysis_plots/`: seed-level plots, reliability/AUC, catastrophe risk,
  failure-channel decomposition, training dynamics, environment-version
  comparisons, nominal-reliability Pareto plots, and seed-by-perturbation
  variance decomposition.

## Result

`tvc400` has a small favorable point estimate: reliability AUC 0.7141 versus
0.7064 for vanilla, and catastrophe probability at half nominal return
0.2790 versus 0.2822. The whole-training-seed bootstrap AUC interval is
[-0.0564, 0.0694], so this is not a confirmed robustness improvement.
Stronger caps are active but reduce policy quality; `tvc500` is inactive.

See `PAPER_INTERPRETATION.md` for the complete interpretation, uncertainty
analysis, v4/v5 comparison, failure-channel decomposition, and recommended
figures.
