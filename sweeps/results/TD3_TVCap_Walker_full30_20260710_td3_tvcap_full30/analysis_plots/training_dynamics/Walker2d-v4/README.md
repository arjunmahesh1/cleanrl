# Walker2d TD3 TV-Cap Training Dynamics

Event coverage: 240 runs; 10/15 requested tags present in every run where applicable.
Training episodic return and episode length are unavailable: neither tag was emitted by any of the 240 runs. The current Gymnasium completion-info format did not enter the training loop's `final_info` logging branch.
Critic, target, loss, throughput, and TV-cap activity tags are intact across the sweep.
The final deterministic nominal evaluation is available: vanilla median nan; highest cap-level median TV c=300.

## Per-Cap Summary

| Model | Q1 median | Pre-cap Q p95 | Critic loss | Final clip frac. | Mean clip frac. | Q p95 / cap | Active final | Active ever |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Vanilla | 168.510 | 225.606 | 46.928 | NA | NA | NA | NA | NA |
| TV c=100 | 93.151 | 101.827 | 16.243 | 0.494 | 0.359 | 1.018 | 1.000 | 1.000 |
| TV c=150 | 129.314 | 151.411 | 38.979 | 0.193 | 0.114 | 1.009 | 1.000 | 1.000 |
| TV c=200 | 158.772 | 199.802 | 46.956 | 0.047 | 0.011 | 0.999 | 0.600 | 0.833 |
| TV c=225 | 166.012 | 214.651 | 46.020 | 0.004 | 0.001 | 0.954 | 0.400 | 0.700 |
| TV c=250 | 165.255 | 221.914 | 44.185 | 0.000 | 0.000 | 0.888 | 0.200 | 0.433 |
| TV c=275 | 169.413 | 224.212 | 47.915 | 0.000 | 0.000 | 0.815 | 0.133 | 0.300 |
| TV c=300 | 170.082 | 228.218 | 47.658 | 0.000 | 0.000 | 0.761 | 0.000 | 0.033 |

These diagnostics establish whether the cap changed training, not whether it improved deployment robustness. Interpret them jointly with the perturbation evaluation.

## Files

- `return_length_dynamics.*`: explicit record that return/length tags were unavailable.
- `critic_actor_dynamics.*`: Q scale, pre-cap Q p95, critic loss, and actor loss.
- `effective_pessimism_dynamics.*`: clip fraction, clipped excess, Q-to-cap ratio, and clip/endpoint scatter.
- `final_nominal_eval_return_by_cap.*` or `final_q_scale_by_cap.*`: 30-seed endpoint distributions.
- `run_summary.csv`, `model_summary.csv`, `training_scalars_binned.csv`: analysis source tables.
