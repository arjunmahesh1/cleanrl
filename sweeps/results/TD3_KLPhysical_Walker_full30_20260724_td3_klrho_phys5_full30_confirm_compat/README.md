# Walker2d TD3 Physical KL Radius 30-Seed Evaluation

- Environment: Walker2d-v5
- Method: TD3 Physical KL Radius
- Models: vanilla, klprho0, klprho0p05
- Seeds: 6--35
- Evaluation episodes per row: 20
- Evaluation rows: 26820
- Total evaluation episodes: 536400
- Fixed deployment-style model in seed-conditioned analysis: klprho0p05
- Variance whiskers are disabled in category return/gain figures.

Layout:
- `raw_metrics/shards/`: chunk-level source CSVs.
- `outputs/combined_metrics.csv`: validated combined evaluation table.
- `Walker2d/`: PPO-style category folders with raw metrics, tables, and plots.
- `analysis_plots/`: seed-level plots, reliability/AUC, catastrophe risk,
  nominal-reliability Pareto plots, failure-channel diagnostics, support
  transfer, and seed-by-perturbation variance decomposition.
- `PAPER_INTERPRETATION.md`: confirmatory result, failure diagnosis, and
  paper-level conclusion.

## Result

The selected explicit radius `rho=0.05` did not confirm on independent
seeds. Reliability AUC is 0.6615 versus 0.7291 for vanilla, a difference of
-0.0677 with whole-seed bootstrap interval [-0.1411, -0.0050].
Catastrophe probability at half nominal return increases from 0.2557 to
0.3347. The matched `rho=0` ensemble control is statistically
indistinguishable from vanilla.

The operator attained the requested radius without saturation. Its principal
observed cost is reduced nominal training reliability, not a missing-support
or solver failure. See `PAPER_INTERPRETATION.md` for the full diagnosis.
