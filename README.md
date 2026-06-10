# PPO Robustness via Value-Target Clipping

Research code for investigating whether upper-tail value-target clipping in PPO instantiates a TV-ball distributionally robust MDP, and whether the theoretical robustness guarantee transfers empirically under MuJoCo dynamics shifts.

**Advisor:** Prof. Xu | **Institution:** Duke University

---

## Overview

This codebase trains PPO with a fixed upper-tail cap on normalized value targets:

```
R̂_t = min(R̂_t, α)
```

where `α` is chosen in normalized return space. The hypothesis is that this corresponds to training under a TV-distance perturbation ball around the nominal MDP, making the learned policy more robust to deployment-time dynamics shifts.

Environments: **Walker2d-v4**, **HalfCheetah-v4** (MuJoCo continuous control)

---

## Repository Structure

```
cleanrl/
  ppo_continuous_action.py    # Main PPO training script (supports --alpha-cap)
  td3_continuous_action.py    # TD3 (future comparison baseline)
  sac_continuous_action.py    # SAC (comparison baseline)
  ppo.py                      # Discrete PPO (used by evaluate_ppo_robust.py)

cleanrl_utils/
  mujoco_xml_utils.py         # MuJoCo XML perturbation engine
  perturbation_config.py      # Perturbation argument builder
  perturbation_wrappers.py    # Gym wrappers for obs/action/reward noise
  buffers.py                  # Replay buffer (TD3/SAC)
  huggingface.py              # Optional model upload (--upload-model)
  evals/
    ppo_eval.py               # PPO eval utility (deterministic, frozen obs norm)
    td3_eval.py               # TD3 eval utility

sweeps/
  package_alpha_robust_eval.py         # Package raw_metrics → outputs + plots
  assemble_presentable_results.py      # Assemble publication-ready plot bundles
  plot_checkpoint_learning_dynamics.py # Plot eval return across training checkpoints
  extract_alpha_results.py             # Extract W&B alpha-grid summaries
  results/                             # All packaged experiment results

slurm/
  train_ppo_alpha_grid.sh              # SLURM training array
  eval_ppo_alpha_grid.sh               # SLURM eval array (all perturbation axes)
  submit_ppo_final_pinned_and_eval.sh  # Pinned train + eval
  submit_ppo_vanilla_noop_paired_nodes.sh

evaluate_ppo_robust.py        # Canonical robustness eval script
AGENTS.md                     # Agent/session onboarding notes
PROJECT_STATUS.md             # Current experiment status and findings
```

---

## Quickstart

**Train PPO with value-target cap:**
```bash
python cleanrl/ppo_continuous_action.py \
  --env-id Walker2d-v4 \
  --alpha-cap 2.85 \
  --total-timesteps 1000000 \
  --seed 1
```

**Evaluate under perturbation:**
```bash
python evaluate_ppo_robust.py \
  --algo ppo_cont \
  --model-path runs/Walker2d-v4__ppo_alpha_a2p85__1/ppo_alpha_a2p85.cleanrl_model \
  --env-id Walker2d-v4 \
  --xml-perturb \
  --xml-body-mass-scale 0.5 \
  --eval-episodes 30 \
  --deterministic-eval
```

**Run SLURM eval sweep (cluster):**
```bash
sbatch slurm/eval_ppo_alpha_grid.sh
```

**Package results:**
```bash
python sweeps/package_alpha_robust_eval.py \
  --raw-metrics-dir sweeps/results/<folder>/raw_metrics \
  --out-dir sweeps/results/<folder>
```

---

## Key Findings

- TV-capping yields **perturbation-dependent** robustness benefits — strongest on actuator gain and mass reduction, absent on friction and damping.
- The mechanism is **failure-rate reduction**, not mean-return improvement: 30-seed analysis shows Walker2d catastrophic failure rate drops 37% → 23% under mass perturbation at the optimal cap.
- Optimal α is environment-specific: ~2.85 for Walker2d, ~2.20–2.55 for HalfCheetah.
- Both environments exhibit **bimodal return distributions** under perturbation (walk vs. fall), making failure-rate the appropriate primary metric rather than mean return.

See `PROJECT_STATUS.md` for full experiment timeline and current status.

---

## Perturbation Axes Covered

| Family | Axes |
|---|---|
| Global dynamics | mass, friction, damping, actuator gain, gravity |
| Localized | per-body mass, per-joint damping, per-actuator gain, per-geom friction |
| Combined | friction+mass, friction+damping, friction+mass+damping |
| Signal | observation noise, action noise (Gaussian), action replacement (Bernoulli), reward noise |

---

## Based On

Forked from [CleanRL](https://github.com/vwxyzjn/cleanrl) (MIT License). Core PPO implementation retained; extended with value-target clipping, MuJoCo XML perturbation engine, and robustness evaluation infrastructure.
