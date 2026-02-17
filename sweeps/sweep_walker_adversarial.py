import argparse
import csv
import itertools
import os
import pathlib
import sys
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List

import gymnasium as gym
import numpy as np
import torch

# Ensure repository root is on sys.path when running as `python sweeps/sweep_walker_adversarial.py`.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cleanrl.ppo_continuous_action import Agent as PPOContinuousAgent
from cleanrl.ppo_continuous_action import make_env as make_ppo_continuous_env


def parse_float_list(spec: str) -> List[float]:
    values = [x.strip() for x in spec.split(",") if x.strip()]
    if not values:
        raise ValueError("Expected non-empty comma-separated float list.")
    return [float(v) for v in values]


def find_wrapper(env: gym.Env, wrapper_cls):
    current = env
    while True:
        if isinstance(current, wrapper_cls):
            return current
        if not hasattr(current, "env"):
            return None
        current = current.env


def load_norm_stats(envs, norm_stats_path: str) -> None:
    if not os.path.exists(norm_stats_path):
        raise FileNotFoundError(
            f"Normalization stats not found: {norm_stats_path}. "
            "Use checkpoints generated after normalization-stats patch."
        )
    stats = np.load(norm_stats_path)
    env = envs.envs[0]
    obs_norm = find_wrapper(env, gym.wrappers.NormalizeObservation)
    rew_norm = find_wrapper(env, gym.wrappers.NormalizeReward)
    if obs_norm is None or rew_norm is None:
        raise RuntimeError("Could not find NormalizeObservation/NormalizeReward wrappers during eval.")

    obs_norm.obs_rms.mean = np.asarray(stats["obs_mean"], dtype=np.float64)
    obs_norm.obs_rms.var = np.asarray(stats["obs_var"], dtype=np.float64)
    obs_norm.obs_rms.count = float(np.asarray(stats["obs_count"], dtype=np.float64))
    rew_norm.return_rms.mean = float(np.asarray(stats["ret_mean"], dtype=np.float64))
    rew_norm.return_rms.var = float(np.asarray(stats["ret_var"], dtype=np.float64))
    rew_norm.return_rms.count = float(np.asarray(stats["ret_count"], dtype=np.float64))


def evaluate_checkpoint(
    *,
    model_path: str,
    norm_stats_path: str,
    env_id: str,
    eval_episodes: int,
    seed: int,
    device: str,
    gamma: float,
    xml_out_dir: str,
    mass_scale: float,
    friction_scale: float,
    damping_scale: float,
    xml_perturb: bool = True,
) -> np.ndarray:
    perturb = SimpleNamespace(
        obs_noise_std=0.0,
        obs_noise_clip=None,
        reward_noise_std=0.0,
        action_noise_std=0.0,
        action_noise_clip=None,
        param_override="",
        param_randomize="",
        param_strict=True,
        xml_perturb=xml_perturb,
        xml_out_dir=xml_out_dir,
        xml_path_override=None,
        xml_body_mass_scale=mass_scale,
        xml_geom_friction_scale=friction_scale,
        xml_joint_damping_scale=damping_scale,
        xml_actuator_gain_scale=1.0,
        xml_actuator_bias_scale=1.0,
    )
    run_name = f"sweep_{env_id}_{mass_scale}_{friction_scale}_{damping_scale}_{int(time.time())}"
    envs = gym.vector.SyncVectorEnv(
        [make_ppo_continuous_env(env_id, 0, False, run_name, gamma, perturb, seed)]
    )
    try:
        load_norm_stats(envs, norm_stats_path)
        torch_device = torch.device(device)
        agent = PPOContinuousAgent(envs).to(torch_device)
        agent.load_state_dict(torch.load(model_path, map_location=torch_device))
        agent.eval()

        obs, _ = envs.reset(seed=seed)
        episodic_returns = []
        while len(episodic_returns) < eval_episodes:
            with torch.no_grad():
                actions, _, _, _ = agent.get_action_and_value(torch.Tensor(obs).to(torch_device))
            obs, _, _, _, infos = envs.step(actions.cpu().numpy())
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        episodic_returns.append(float(np.asarray(info["episode"]["r"]).item()))
                        if len(episodic_returns) >= eval_episodes:
                            break
        return np.asarray(episodic_returns, dtype=np.float64)
    finally:
        envs.close()


def summarize(returns: np.ndarray) -> Dict[str, float]:
    return {
        "mean_return": float(returns.mean()),
        "std_return": float(returns.std(ddof=1) if len(returns) > 1 else 0.0),
        "median_return": float(np.median(returns)),
        "min_return": float(returns.min()),
        "max_return": float(returns.max()),
    }


@dataclass
class ModelSpec:
    name: str
    model_path: str
    norm_stats_path: str


def auto_norm_stats_path(model_path: str) -> str:
    return f"{model_path}.norm_stats.npz"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Sweep Walker2d XML perturbations and select an adversarial perturbation "
            "using degradation from nominal performance."
        )
    )
    parser.add_argument("--env-id", default="Walker2d-v4")
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--xml-out-dir", default="perturbed_xml")
    parser.add_argument("--mass-scales", default="0.8,0.9,1.0,1.1,1.2")
    parser.add_argument("--friction-scales", default="0.8,0.9,1.0,1.1,1.2")
    parser.add_argument("--damping-scales", default="0.8,0.9,1.0,1.1,1.2")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--out-csv", default="")
    parser.add_argument("--selection-target", choices=["vanilla", "tv90"], default="vanilla")
    parser.add_argument(
        "--selection-metric",
        choices=[
            "drop_mean",
            "drop_median",
            "delta_mean",
            "delta_median",
            "mean_return",
            "median_return",
        ],
        default="drop_mean",
    )

    parser.add_argument("--vanilla-model-path", required=True)
    parser.add_argument("--vanilla-norm-stats-path", default="")
    parser.add_argument("--tv90-model-path", default="")
    parser.add_argument("--tv90-norm-stats-path", default="")

    parser.add_argument("--track", action="store_true")
    parser.add_argument("--wandb-project-name", default="cleanRL")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-group", default="walker2d_adversarial_sweep")
    args = parser.parse_args()

    mass_scales = parse_float_list(args.mass_scales)
    friction_scales = parse_float_list(args.friction_scales)
    damping_scales = parse_float_list(args.damping_scales)

    vanilla_stats = args.vanilla_norm_stats_path or auto_norm_stats_path(args.vanilla_model_path)
    models = [
        ModelSpec(name="vanilla", model_path=args.vanilla_model_path, norm_stats_path=vanilla_stats),
    ]
    if args.tv90_model_path:
        tv90_stats = args.tv90_norm_stats_path or auto_norm_stats_path(args.tv90_model_path)
        models.append(ModelSpec(name="tv90", model_path=args.tv90_model_path, norm_stats_path=tv90_stats))

    combos = list(itertools.product(mass_scales, friction_scales, damping_scales))
    combos = [c for c in combos if not (c[0] == 1.0 and c[1] == 1.0 and c[2] == 1.0)]
    if not combos:
        raise ValueError("No perturbation combos after excluding nominal (1.0,1.0,1.0).")

    nominal_stats: Dict[str, Dict[str, float]] = {}
    print(f"evaluating nominal reference on {args.env_id} ...")
    for model in models:
        nominal_returns = evaluate_checkpoint(
            model_path=model.model_path,
            norm_stats_path=model.norm_stats_path,
            env_id=args.env_id,
            eval_episodes=args.eval_episodes,
            seed=args.seed,
            device=args.device,
            gamma=args.gamma,
            xml_out_dir=args.xml_out_dir,
            mass_scale=1.0,
            friction_scale=1.0,
            damping_scale=1.0,
            xml_perturb=False,
        )
        nominal_stats[model.name] = summarize(nominal_returns)
        print(
            f"{model.name} nominal: mean={nominal_stats[model.name]['mean_return']:.4f}, "
            f"median={nominal_stats[model.name]['median_return']:.4f}"
        )

    wandb_run = None
    if args.track:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            group=args.wandb_group,
            job_type="sweep",
            config=vars(args),
            name=f"sweep__{args.env_id}__{int(time.time())}",
        )

    rows = []
    print(f"sweeping {len(combos)} perturbation combos on {args.env_id} ...")
    for idx, (m, f, d) in enumerate(combos, start=1):
        print(f"[{idx}/{len(combos)}] mass={m:.3f} friction={f:.3f} damping={d:.3f}")
        row = {"mass_scale": m, "friction_scale": f, "damping_scale": d}
        for model in models:
            returns = evaluate_checkpoint(
                model_path=model.model_path,
                norm_stats_path=model.norm_stats_path,
                env_id=args.env_id,
                eval_episodes=args.eval_episodes,
                seed=args.seed,
                device=args.device,
                gamma=args.gamma,
                xml_out_dir=args.xml_out_dir,
                mass_scale=m,
                friction_scale=f,
                damping_scale=d,
                xml_perturb=True,
            )
            stats = summarize(returns)
            for k, v in stats.items():
                row[f"{model.name}_{k}"] = v
            row[f"{model.name}_nominal_mean_return"] = nominal_stats[model.name]["mean_return"]
            row[f"{model.name}_nominal_median_return"] = nominal_stats[model.name]["median_return"]
            row[f"{model.name}_delta_mean_from_nominal"] = (
                row[f"{model.name}_mean_return"] - nominal_stats[model.name]["mean_return"]
            )
            row[f"{model.name}_delta_median_from_nominal"] = (
                row[f"{model.name}_median_return"] - nominal_stats[model.name]["median_return"]
            )
            row[f"{model.name}_drop_mean_from_nominal"] = (
                nominal_stats[model.name]["mean_return"] - row[f"{model.name}_mean_return"]
            )
            row[f"{model.name}_drop_median_from_nominal"] = (
                nominal_stats[model.name]["median_return"] - row[f"{model.name}_median_return"]
            )
        rows.append(row)

        if wandb_run is not None:
            import wandb

            payload = {
                "sweep/mass_scale": m,
                "sweep/friction_scale": f,
                "sweep/damping_scale": d,
            }
            for k, v in row.items():
                if isinstance(v, (int, float)):
                    payload[f"sweep/{k}"] = v
            wandb.log(payload)

    target = args.selection_target
    metric_key_map = {
        "drop_mean": f"{target}_drop_mean_from_nominal",
        "drop_median": f"{target}_drop_median_from_nominal",
        "delta_mean": f"{target}_delta_mean_from_nominal",
        "delta_median": f"{target}_delta_median_from_nominal",
        "mean_return": f"{target}_mean_return",
        "median_return": f"{target}_median_return",
    }
    score_key = metric_key_map[args.selection_metric]
    reverse = args.selection_metric.startswith("drop_")
    rows_sorted = sorted(rows, key=lambda r: r[score_key], reverse=reverse)
    adversarial = rows_sorted[0]
    print(
        f"\nselected perturbation by target={args.selection_target}, "
        f"metric={args.selection_metric} ({score_key}):"
    )
    print(
        f"mass={adversarial['mass_scale']:.3f}, "
        f"friction={adversarial['friction_scale']:.3f}, "
        f"damping={adversarial['damping_scale']:.3f}, "
        f"score={adversarial[score_key]:.4f}"
    )
    if "tv90_mean_return" in adversarial:
        print(
            f"vanilla_mean={adversarial['vanilla_mean_return']:.4f}, "
            f"tv90_mean={adversarial['tv90_mean_return']:.4f}"
        )
    if args.selection_metric.startswith("drop_") and adversarial[score_key] <= 0:
        print("warning: selected perturbation does not reduce target performance (not strictly adversarial).")
    if args.selection_metric.startswith("delta_") and adversarial[score_key] >= 0:
        print("warning: selected perturbation has non-negative delta vs nominal (not strictly adversarial).")

    print(
        f"\ntop {min(args.top_k, len(rows_sorted))} combos by "
        f"{args.selection_target}:{args.selection_metric}:"
    )
    for i, row in enumerate(rows_sorted[: args.top_k], start=1):
        line = (
            f"{i:2d}. mass={row['mass_scale']:.3f}, friction={row['friction_scale']:.3f}, "
            f"damping={row['damping_scale']:.3f}, score={row[score_key]:.4f}, "
            f"vanilla_mean={row['vanilla_mean_return']:.4f}, "
            f"vanilla_drop={row['vanilla_drop_mean_from_nominal']:.4f}"
        )
        if "tv90_mean_return" in row:
            line += (
                f", tv90_mean={row['tv90_mean_return']:.4f}, "
                f"tv90_drop={row['tv90_drop_mean_from_nominal']:.4f}"
            )
        print(line)

    out_csv = args.out_csv or os.path.join(
        "sweeps",
        f"{args.env_id.replace('/', '_')}__adversarial_sweep__{int(time.time())}.csv",
    )
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    fieldnames = sorted(rows[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_sorted)
    print(f"\nsweep results written to: {out_csv}")

    print("\nfinal-eval perturbation flags:")
    print(
        f"--xml-body-mass-scale {adversarial['mass_scale']} "
        f"--xml-geom-friction-scale {adversarial['friction_scale']} "
        f"--xml-joint-damping-scale {adversarial['damping_scale']}"
    )

    if wandb_run is not None:
        wandb_run.summary["selection/target"] = args.selection_target
        wandb_run.summary["selection/metric"] = args.selection_metric
        wandb_run.summary["selection/score_key"] = score_key
        wandb_run.summary["selection/score"] = adversarial[score_key]
        wandb_run.summary["selected/mass_scale"] = adversarial["mass_scale"]
        wandb_run.summary["selected/friction_scale"] = adversarial["friction_scale"]
        wandb_run.summary["selected/damping_scale"] = adversarial["damping_scale"]
        wandb_run.summary["selected/vanilla_mean_return"] = adversarial["vanilla_mean_return"]
        wandb_run.summary["selected/vanilla_drop_mean_from_nominal"] = adversarial["vanilla_drop_mean_from_nominal"]
        if "tv90_mean_return" in adversarial:
            wandb_run.summary["selected/tv90_mean_return"] = adversarial["tv90_mean_return"]
            wandb_run.summary["selected/tv90_drop_mean_from_nominal"] = adversarial["tv90_drop_mean_from_nominal"]
        wandb_run.finish()


if __name__ == "__main__":
    main()
