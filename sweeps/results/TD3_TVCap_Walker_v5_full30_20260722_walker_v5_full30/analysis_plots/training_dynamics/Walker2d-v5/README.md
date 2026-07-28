# Walker2d TD3 TV-Cap Training Dynamics

Event coverage: 300 runs; 13/15 requested tags present in every run where applicable.
Training episodic return and episode length are unavailable: neither tag was emitted by any of the 240 runs. The current Gymnasium completion-info format did not enter the training loop's `final_info` logging branch.
Critic, target, loss, throughput, and TV-cap activity tags are intact across the sweep.
The final deterministic nominal evaluation is available: vanilla median 4113.1; highest cap-level median TV c=400.

## Per-Cap Summary

| Model | Q1 median | Pre-cap Q p95 | Critic loss | Final clip frac. | Mean clip frac. | Q p95 / cap | Active final | Active ever |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Vanilla | 289.080 | 380.473 | 18.232 | NA | NA | NA | NA | NA |
| TV c=100 | 93.908 | 102.262 | 3.340 | 0.695 | 0.525 | 1.023 | 1.000 | 1.000 |
| TV c=150 | 135.405 | 152.671 | 8.655 | 0.441 | 0.275 | 1.018 | 1.000 | 1.000 |
| TV c=200 | 177.607 | 202.604 | 12.706 | 0.361 | 0.169 | 1.013 | 1.000 | 1.000 |
| TV c=225 | 195.814 | 227.293 | 11.922 | 0.311 | 0.144 | 1.010 | 1.000 | 1.000 |
| TV c=250 | 221.140 | 252.067 | 13.644 | 0.287 | 0.098 | 1.008 | 1.000 | 1.000 |
| TV c=275 | 237.594 | 276.586 | 13.574 | 0.236 | 0.067 | 1.006 | 0.967 | 0.967 |
| TV c=300 | 251.188 | 301.041 | 15.281 | 0.188 | 0.044 | 1.003 | 0.967 | 1.000 |
| TV c=400 | 287.780 | 381.870 | 19.972 | 0.000 | 0.000 | 0.955 | 0.333 | 0.467 |
| TV c=500 | 267.530 | 353.316 | 17.115 | 0.000 | 0.000 | 0.707 | 0.000 | 0.000 |

These diagnostics establish whether the cap changed training, not whether it improved deployment robustness. Interpret them jointly with the perturbation evaluation.

## Files

- `return_length_dynamics.*`: explicit record that return/length tags were unavailable.
- `critic_actor_dynamics.*`: Q scale, pre-cap Q p95, critic loss, and actor loss.
- `effective_pessimism_dynamics.*`: clip fraction, clipped excess, Q-to-cap ratio, and clip/endpoint scatter.
- `final_nominal_eval_return_by_cap.*` or `final_q_scale_by_cap.*`: 30-seed endpoint distributions.
- `run_summary.csv`, `model_summary.csv`, `training_scalars_binned.csv`: analysis source tables.
