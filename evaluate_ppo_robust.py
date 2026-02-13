import argparse
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch

from cleanrl.ppo import Agent as PPOAgent
from cleanrl.ppo import make_env as make_ppo_env
from cleanrl.ppo_continuous_action import Agent as PPOContinuousAgent
from cleanrl.ppo_continuous_action import make_env as make_ppo_continuous_env


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

    if args.algo == "ppo":
        episodic_returns = evaluate_discrete(args)
    else:
        episodic_returns = evaluate_continuous(args)

    returns = np.array(episodic_returns, dtype=np.float64)
    print(f"mean_return={returns.mean():.4f}")
    print(f"std_return={returns.std(ddof=1) if len(returns) > 1 else 0.0:.4f}")
    print(f"min_return={returns.min():.4f}")
    print(f"max_return={returns.max():.4f}")


if __name__ == "__main__":
    main()
