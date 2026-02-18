import argparse
import csv
import os
import time
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch

from cleanrl.ppo import Agent as PPOAgent
from cleanrl.ppo import make_env as make_ppo_env
from cleanrl.ppo_continuous_action import Agent as PPOContinuousAgent
from cleanrl.ppo_continuous_action import make_env as make_ppo_continuous_env


def interquartile_mean(x: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    lower = np.quantile(x, 0.25)
    upper = np.quantile(x, 0.75)
    middle = x[(x >= lower) & (x <= upper)]
    if middle.size == 0:
        return float(np.median(x))
    return float(np.mean(middle))


def find_wrapper(env: gym.Env, wrapper_cls):
    current = env
    while True:
        if isinstance(current, wrapper_cls):
            return current
        if not hasattr(current, "env"):
            return None
        current = current.env


def load_normalization_stats_if_available(envs, args: argparse.Namespace) -> bool:
    norm_stats_path = args.norm_stats_path or f"{args.model_path}.norm_stats.npz"
    if not os.path.exists(norm_stats_path):
        print(f"warning: normalization stats file not found: {norm_stats_path}")
        print("warning: ppo_cont evaluation without training normalization stats can severely under-report performance.")
        return False

    stats = np.load(norm_stats_path)
    env = envs.envs[0]
    obs_norm = find_wrapper(env, gym.wrappers.NormalizeObservation)
    rew_norm = find_wrapper(env, gym.wrappers.NormalizeReward)
    if obs_norm is None or rew_norm is None:
        print("warning: normalization wrappers not found in eval env; cannot load normalization stats.")
        return False

    obs_norm.obs_rms.mean = np.asarray(stats["obs_mean"], dtype=np.float64)
    obs_norm.obs_rms.var = np.asarray(stats["obs_var"], dtype=np.float64)
    obs_norm.obs_rms.count = float(np.asarray(stats["obs_count"], dtype=np.float64))
    rew_norm.return_rms.mean = float(np.asarray(stats["ret_mean"], dtype=np.float64))
    rew_norm.return_rms.var = float(np.asarray(stats["ret_var"], dtype=np.float64))
    rew_norm.return_rms.count = float(np.asarray(stats["ret_count"], dtype=np.float64))
    print(f"loaded normalization stats from {norm_stats_path}")
    return True


def is_perturbed_eval(args: argparse.Namespace) -> bool:
    return any(
        [
            args.obs_noise_std > 0.0,
            args.reward_noise_std > 0.0,
            args.action_noise_std > 0.0,
            bool(args.param_override),
            bool(args.param_randomize),
            args.xml_perturb,
            args.xml_body_mass_scale != 1.0,
            args.xml_geom_friction_scale != 1.0,
            args.xml_joint_damping_scale != 1.0,
            args.xml_actuator_gain_scale != 1.0,
            args.xml_actuator_bias_scale != 1.0,
        ]
    )


def build_perturbation_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        obs_noise_std=args.obs_noise_std,
        obs_noise_clip=args.obs_noise_clip,
        reward_noise_std=args.reward_noise_std,
        action_noise_std=args.action_noise_std,
        action_noise_clip=args.action_noise_clip,
        param_override=args.param_override,
        param_randomize=args.param_randomize,
        param_strict=args.param_strict,
        xml_perturb=args.xml_perturb,
        xml_out_dir=args.xml_out_dir,
        xml_path_override=args.xml_path_override,
        xml_body_mass_scale=args.xml_body_mass_scale,
        xml_geom_friction_scale=args.xml_geom_friction_scale,
        xml_joint_damping_scale=args.xml_joint_damping_scale,
        xml_actuator_gain_scale=args.xml_actuator_gain_scale,
        xml_actuator_bias_scale=args.xml_actuator_bias_scale,
    )


def evaluate_discrete(args: argparse.Namespace):
    perturb = build_perturbation_args(args)
    run_name = args.run_name or f"eval_{args.env_id}"
    envs = gym.vector.SyncVectorEnv([make_ppo_env(args.env_id, 0, args.capture_video, run_name, perturb, args.seed)])
    device = torch.device(args.device)
    agent = PPOAgent(envs).to(device)
    agent.load_state_dict(torch.load(args.model_path, map_location=device))
    agent.eval()

    obs, _ = envs.reset(seed=args.seed)
    episodic_returns = []
    while len(episodic_returns) < args.eval_episodes:
        with torch.no_grad():
            actions, _, _, _ = agent.get_action_and_value(torch.Tensor(obs).to(device))
        obs, _, _, _, infos = envs.step(actions.cpu().numpy())
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    episodic_returns.append(float(np.asarray(info["episode"]["r"]).item()))
                    print(f"eval_episode={len(episodic_returns)-1}, episodic_return={info['episode']['r']}")
                    if len(episodic_returns) >= args.eval_episodes:
                        break
    envs.close()
    return episodic_returns


def evaluate_continuous(args: argparse.Namespace):
    perturb = build_perturbation_args(args)
    run_name = args.run_name or f"eval_{args.env_id}"
    envs = gym.vector.SyncVectorEnv(
        [make_ppo_continuous_env(args.env_id, 0, args.capture_video, run_name, args.gamma, perturb, args.seed)]
    )
    load_normalization_stats_if_available(envs, args)
    device = torch.device(args.device)
    agent = PPOContinuousAgent(envs).to(device)
    agent.load_state_dict(torch.load(args.model_path, map_location=device))
    agent.eval()

    obs, _ = envs.reset(seed=args.seed)
    episodic_returns = []
    while len(episodic_returns) < args.eval_episodes:
        with torch.no_grad():
            actions, _, _, _ = agent.get_action_and_value(torch.Tensor(obs).to(device))
        obs, _, _, _, infos = envs.step(actions.cpu().numpy())
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    episodic_returns.append(float(np.asarray(info["episode"]["r"]).item()))
                    print(f"eval_episode={len(episodic_returns)-1}, episodic_return={info['episode']['r']}")
                    if len(episodic_returns) >= args.eval_episodes:
                        break
    envs.close()
    return episodic_returns


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained PPO checkpoints under nominal or perturbed environments.")
    parser.add_argument("--algo", choices=["ppo", "ppo_cont"], required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--env-id", required=True)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--capture-video", action="store_true")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--norm-stats-path", default="")
    parser.add_argument("--track", action="store_true")
    parser.add_argument("--wandb-project-name", default="cleanRL")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-group", default="")
    parser.add_argument("--metrics-out-csv", default="")
    parser.add_argument("--model-label", default="")
    parser.add_argument("--scenario-label", default="")

    parser.add_argument("--obs-noise-std", type=float, default=0.0)
    parser.add_argument("--obs-noise-clip", type=float, default=None)
    parser.add_argument("--reward-noise-std", type=float, default=0.0)
    parser.add_argument("--action-noise-std", type=float, default=0.0)
    parser.add_argument("--action-noise-clip", type=float, default=None)
    parser.add_argument("--param-override", default="")
    parser.add_argument("--param-randomize", default="")
    parser.add_argument("--param-strict", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--xml-perturb", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--xml-out-dir", default="perturbed_xml")
    parser.add_argument("--xml-path-override", default=None)
    parser.add_argument("--xml-body-mass-scale", type=float, default=1.0)
    parser.add_argument("--xml-geom-friction-scale", type=float, default=1.0)
    parser.add_argument("--xml-joint-damping-scale", type=float, default=1.0)
    parser.add_argument("--xml-actuator-gain-scale", type=float, default=1.0)
    parser.add_argument("--xml-actuator-bias-scale", type=float, default=1.0)

    args = parser.parse_args()
    eval_name = args.run_name or f"eval__{args.env_id}__{args.algo}__{int(time.time())}"
    is_perturbed = is_perturbed_eval(args)

    wandb_run = None
    if args.track:
        import wandb

        config = vars(args).copy()
        config["eval_is_perturbed"] = is_perturbed
        config["model_basename"] = os.path.basename(args.model_path)
        wandb_run = wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=config,
            name=eval_name,
            group=args.wandb_group or None,
            job_type="evaluation",
        )

    if args.algo == "ppo":
        episodic_returns = evaluate_discrete(args)
    else:
        episodic_returns = evaluate_continuous(args)

    returns = np.array(episodic_returns, dtype=np.float64)
    mean_return = returns.mean()
    std_return = returns.std(ddof=1) if len(returns) > 1 else 0.0
    median_return = np.median(returns)
    iqm_return = interquartile_mean(returns)
    min_return = returns.min()
    max_return = returns.max()
    print(f"mean_return={mean_return:.4f}")
    print(f"std_return={std_return:.4f}")
    print(f"median_return={median_return:.4f}")
    print(f"iqm_return={iqm_return:.4f}")
    print(f"min_return={min_return:.4f}")
    print(f"max_return={max_return:.4f}")

    if wandb_run is not None:
        import wandb

        for i, r in enumerate(episodic_returns):
            wandb.log({"eval/episode": i, "eval/episodic_return": r})
        wandb.log(
            {
                "eval/mean_return": mean_return,
                "eval/std_return": std_return,
                "eval/median_return": median_return,
                "eval/iqm_return": iqm_return,
                "eval/min_return": min_return,
                "eval/max_return": max_return,
                "eval/is_perturbed": int(is_perturbed),
                "eval/eval_episodes": len(episodic_returns),
            }
        )
        wandb_run.summary["eval/mean_return"] = mean_return
        wandb_run.summary["eval/std_return"] = std_return
        wandb_run.summary["eval/median_return"] = median_return
        wandb_run.summary["eval/iqm_return"] = iqm_return
        wandb_run.summary["eval/min_return"] = min_return
        wandb_run.summary["eval/max_return"] = max_return
        wandb_run.summary["eval/is_perturbed"] = int(is_perturbed)
        wandb_run.finish()

    if args.metrics_out_csv:
        os.makedirs(os.path.dirname(args.metrics_out_csv) or ".", exist_ok=True)
        row = {
            "timestamp": int(time.time()),
            "run_name": eval_name,
            "algo": args.algo,
            "env_id": args.env_id,
            "seed": args.seed,
            "eval_episodes": len(episodic_returns),
            "is_perturbed": int(is_perturbed),
            "model_label": args.model_label,
            "scenario_label": args.scenario_label,
            "model_path": args.model_path,
            "norm_stats_path": args.norm_stats_path or f"{args.model_path}.norm_stats.npz",
            "xml_perturb": int(args.xml_perturb),
            "xml_body_mass_scale": args.xml_body_mass_scale,
            "xml_geom_friction_scale": args.xml_geom_friction_scale,
            "xml_joint_damping_scale": args.xml_joint_damping_scale,
            "mean_return": float(mean_return),
            "std_return": float(std_return),
            "median_return": float(median_return),
            "iqm_return": float(iqm_return),
            "min_return": float(min_return),
            "max_return": float(max_return),
        }
        header = list(row.keys())
        write_header = not os.path.exists(args.metrics_out_csv)
        with open(args.metrics_out_csv, "a", newline="", encoding="utf-8") as fobj:
            writer = csv.DictWriter(fobj, fieldnames=header)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        print(f"metrics appended to {args.metrics_out_csv}")


if __name__ == "__main__":
    main()
