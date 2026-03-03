# PPO Percentile Fixed-Cap Phase 1 Eval

Goal:
- replace the old AUC-selected fixed cap with a percentile-based fixed-cap rule derived from converged vanilla PPO returns;
- retrain TV PPO at candidate caps `p75/p85/p90/p95`;
- evaluate each candidate on single-axis friction, mass, and damping sweeps;
- select one clipped PPO variant to carry into the next robustness phases.

## Metrics

- `nominal return`: performance in the unperturbed environment; this guards against picking a clipped policy that is simply weaker everywhere.
- `perturbed return`: absolute performance under a perturbation; this is the cleanest practical robustness metric.
- `drop = nominal - perturbed`: how much performance the perturbation removed.
- `robust gain = vanilla_drop - tv_drop`: positive means the clipped policy degraded less than vanilla, but this can be confounded if TV starts from a much lower nominal return.

## Candidate caps

| Cap | Shared cap | TV nominal | Vanilla nominal | Nominal gap (TV - vanilla) |
| --- | ---: | ---: | ---: | ---: |
| p75 | `4196.32` | `2744.57 +/- 291.21` | `3095.46 +/- 705.79` | `-350.89` |
| p85 | `4267.10` | `2457.80 +/- 578.32` | `3114.47 +/- 586.59` | `-656.67` |
| p90 | `4292.89` | `2624.24 +/- 475.70` | `3168.00 +/- 670.83` | `-543.76` |
| p95 | `4324.87` | `2802.50 +/- 431.43` | `3191.57 +/- 697.85` | `-389.08` |

All four percentile-cap variants have lower nominal return than vanilla. This is the main interpretation caveat for the gain metric.

## Axis-level summary

| Cap | Avg gain: friction | Avg gain: mass | Avg gain: damping | Summary |
| --- | ---: | ---: | ---: | --- |
| p75 | `+166.24` | `-48.19` | `-19.10` | Mixed; smallest nominal deficit, but not strong enough overall |
| p85 | `+272.23` | `+46.54` | `+89.17` | Good drop-based gains, but very weak nominal baseline |
| p90 | `+145.53` | `+12.69` | `+16.57` | Mild positive shift relative to the old cap, but weaker than `p95` |
| p95 | `+308.62` | `+98.47` | `+82.01` | Best overall candidate |

## Main findings

- `p95` is the best candidate to carry forward.
- The clearest genuine TV wins are on high-friction perturbations, where TV beats vanilla in absolute perturbed return despite starting from a lower nominal baseline:
  - `friction_1p3`: TV `2957.06` vs vanilla `2334.69`
  - `friction_1p5`: TV `2045.85` vs vanilla `1623.43`
  - `friction_1p7`: TV `1362.07` vs vanilla `1058.08`
  - `friction_2p0`: TV `854.79` vs vanilla `550.69`
- Mass improves in drop-space under `p95`, with several positive gain intervals, but TV still remains below vanilla in absolute perturbed return at all tested mass settings.
- Damping also improves substantially relative to the earlier `cap=3500` setup, but those gains are mostly drop-based; TV still remains below vanilla in absolute perturbed return at all tested damping settings.

## Interpretation

Phase 1 succeeded in finding a better clipped PPO candidate than the earlier fixed-cap selection method. The strongest evidence is that `p95` produces real absolute-return robustness wins in the harsher friction regimes and no longer collapses on damping the way the earlier `cap=3500` policy did.

The remaining issue is nominal performance: the percentile-cap variants all start below vanilla. That means some positive robust-gain results can come from lower starting performance rather than a true absolute robustness advantage. For the next phase, `p95` is still the correct choice, but the nominal-gap confound should be kept explicit in reporting.

## Files

- `inputs/shared_percentile_caps.csv`: shared cap table used for retraining.
- `inputs/per_seed_percentile_caps.csv`: per-seed percentile values from vanilla runs.
- `inputs/phase1_train_manifest.csv`: training manifest for the 20 percentile-cap retraining jobs.
- `outputs/ppo_robust_eval_metrics_pctcap_p75_axes.csv`
- `outputs/ppo_robust_eval_metrics_pctcap_p85_axes.csv`
- `outputs/ppo_robust_eval_metrics_pctcap_p90_axes.csv`
- `outputs/ppo_robust_eval_metrics_pctcap_p95_axes.csv`
- `outputs/ppo_robust_summary_pctcap_*_{friction,mass,damping}.csv`: per-cap per-axis summaries.
