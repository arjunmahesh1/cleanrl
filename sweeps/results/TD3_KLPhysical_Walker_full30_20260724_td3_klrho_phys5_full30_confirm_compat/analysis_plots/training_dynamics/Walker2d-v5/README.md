# Physical-support TD3-KL diagnostic

Promotion requires:
- median same-seed nominal retention >= 0.70
- median same-seed final critic-loss ratio <= 10.0
- finite diagnostics and a measured realized KL radius

Promoted by the automated health gate: klprho0p05

The health gate is necessary but not sufficient. Promoted settings must still
beat vanilla on predeclared perturbation and catastrophe-risk metrics.

## Scalar coverage

- `kl_physical/effective_beta_mean`: 33.3%
- `kl_physical/effective_beta_median`: 33.3%
- `kl_physical/effective_num_dynamics_mean`: 66.7%
- `kl_physical/implicit_kl_radius_mean`: 66.7%
- `kl_physical/implicit_kl_radius_p95`: 66.7%
- `kl_physical/nominal_reward_abs_error`: 66.7%
- `kl_physical/nominal_obs_max_abs_error`: 66.7%
- `kl_physical/joint_return_std_across_dynamics`: 66.7%
- `kl_physical/pessimism_gap_p95`: 66.7%
- `kl_physical/reference_target_mean`: 66.7%
- `kl_physical/requested_radius`: 66.7%
- `kl_physical/pessimism_gap_mean`: 66.7%
- `kl_physical/robust_target_mean`: 66.7%
- `kl_physical/worst_member_target_mean`: 66.7%
- `kl_physical/worst_member_adversarial_weight_mean`: 66.7%
- `kl_physical/worst_case_saturation_fraction`: 66.7%
- `charts/SPS`: 100.0%
- `eval/episodic_return`: 100.0%
- `charts/episodic_return`: 100.0%
- `losses/actor_loss`: 100.0%
- `losses/qf_loss`: 100.0%
