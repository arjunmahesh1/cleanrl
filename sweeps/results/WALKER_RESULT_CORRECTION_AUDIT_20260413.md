# Walker Result Correction Audit

This audit records which packaged Walker robustness folders are still valid, which are partially invalid due to earlier XML perturbation bugs, and how to correct them.

## Code fixes now applied locally

- `mass` / localized `*_mass` now scale effective mass through geom `density` rather than nonexistent body `mass` attributes.
- `actuator_gain` / localized `*_actuator_gain` now scale motor `gear` (and still scale `gainprm` when present in other MuJoCo models).
- localized damping now injects explicit `damping` on the targeted joint using the inherited default joint damping before scaling.
- the packager supports `--exclude-axes ...` so permanently invalid axes can be omitted from corrected plots.

## Trustworthy as-is

- `PPO_Adversarial_action_noise_20260406`
- `PPO_Adversarial_action_bernoulli_20260408_clean_v3`

## Partially invalid and needs rerun into the same folder

### `PPO_AlphaGridNorm_final_eval_20260329`

- rerun axis: `mass`
- keep as-is: `friction`, `damping`

### `PPO_AlphaGridNorm_multicap_eval_20260330`

- rerun axis: `mass`
- keep as-is: `friction`, `damping`

### `PPO_AlphaGridNorm_expanded_eval_20260404`

- rerun axis: `mass`
- keep as-is: `friction`, `damping`

### `PPO_Adversarial_single_axis_20260406`

- rerun axes: `mass`, `actuator_gain`
- keep as-is: `friction`, `damping`, `obs_noise`, `reward_noise`
- exclude from corrected packaging: `actuator_bias`

### `PPO_Adversarial_fmd_combo_20260406`

- rerun axes: `friction_mass`, `mass_damping`, `friction_mass_damping`
- keep as-is: `friction_damping`

### `PPO_Adversarial_walker_targeted_xml_20260413`

- rerun axes:
  - `thigh_left_mass`
  - `leg_left_mass`
  - `foot_left_mass`
  - `thigh_left_damping`
  - `leg_left_damping`
  - `foot_left_damping`
  - `thigh_left_actuator_gain`
  - `leg_left_actuator_gain`
  - `foot_left_actuator_gain`
- keep as-is: `foot_left_friction`

## Not the current paper focus, but also suspect if used later

- `PPO_Mass_rawreward_5seed_0p5_2p0_20260226`
- `PPO_FixedAlpha_mass_20260307`

These were built on the old mass implementation and should not be used without a corrected rerun.

## Correction strategy

1. Push the fixed XML/eval/package files to cluster.
2. Rerun only the invalid axes into the same `raw_metrics/` folders.
3. Repackage the folders.
4. Use `--exclude-axes actuator_bias` for the corrected `single_axis` package.

Because packaging keeps the latest timestamped row per `(model_label, scenario_label, seed)`, rerunning into the same CSV files is safe and avoids rewriting trustworthy axes.
