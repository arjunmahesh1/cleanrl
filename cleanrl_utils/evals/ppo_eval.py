import os
from types import SimpleNamespace
from typing import Callable

import gymnasium as gym
import numpy as np
import torch


def extract_episode_returns_from_infos(infos: dict) -> list[float]:
    episodic_returns: list[float] = []

    final_infos = infos.get("final_info")
    if final_infos is not None:
        for info in final_infos:
            if info and "episode" in info:
                episodic_returns.append(float(np.asarray(info["episode"]["r"]).item()))

    if episodic_returns:
        return episodic_returns

    ep = infos.get("episode")
    if isinstance(ep, dict) and "r" in ep:
        rs = np.asarray(ep["r"])
        mask = np.asarray(infos.get("_episode", np.ones_like(rs, dtype=bool)), dtype=bool)
        if rs.shape == ():
            if bool(mask.item() if mask.shape == () else True):
                episodic_returns.append(float(rs.item()))
        else:
            for r, m in zip(rs.reshape(-1), mask.reshape(-1)):
                if bool(m):
                    episodic_returns.append(float(np.asarray(r).item()))

    return episodic_returns


def find_wrapper(env: gym.Env, wrapper_type: type) -> gym.Wrapper | None:
    current = env
    while current is not None:
        if isinstance(current, wrapper_type):
            return current
        if not hasattr(current, "env"):
            return None
        current = current.env
    return None


def load_normalization_stats_if_available(
    envs: gym.vector.VectorEnv,
    norm_stats_path: str,
    eval_raw_rewards: bool,
) -> bool:
    if not os.path.exists(norm_stats_path):
        print(f"warning: normalization stats file not found: {norm_stats_path}")
        return False

    stats = np.load(norm_stats_path)
    env = envs.envs[0]
    obs_norm = find_wrapper(env, gym.wrappers.NormalizeObservation)
    if obs_norm is None:
        print("warning: NormalizeObservation wrapper not found in eval env; cannot load observation stats.")
        return False

    obs_norm.obs_rms.mean = np.asarray(stats["obs_mean"], dtype=np.float64)
    obs_norm.obs_rms.var = np.asarray(stats["obs_var"], dtype=np.float64)
    obs_norm.obs_rms.count = float(np.asarray(stats["obs_count"], dtype=np.float64))

    if eval_raw_rewards:
        print(f"loaded observation normalization stats from {norm_stats_path} (raw rewards enabled)")
        return True

    rew_norm = find_wrapper(env, gym.wrappers.NormalizeReward)
    if rew_norm is None:
        print("warning: NormalizeReward wrapper not found in eval env; cannot load reward stats.")
        return False
    if not {"ret_mean", "ret_var", "ret_count"}.issubset(set(stats.files)):
        print("warning: checkpoint normalization stats do not include reward normalization fields.")
        return False

    rew_norm.return_rms.mean = float(np.asarray(stats["ret_mean"], dtype=np.float64))
    rew_norm.return_rms.var = float(np.asarray(stats["ret_var"], dtype=np.float64))
    rew_norm.return_rms.count = float(np.asarray(stats["ret_count"], dtype=np.float64))
    print(f"loaded observation+reward normalization stats from {norm_stats_path}")
    return True


def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    capture_video: bool = True,
    gamma: float = 0.99,
    norm_stats_path: str = "",
    eval_raw_rewards: bool = False,
    normalize_reward: bool = True,
    seed: int = 0,
):
    eval_args = SimpleNamespace(
        xml_perturb=False,
        xml_out_dir="perturbed_xml",
        xml_body_mass_scale=1.0,
        xml_geom_friction_scale=1.0,
        xml_joint_damping_scale=1.0,
        xml_actuator_gain_scale=1.0,
        xml_actuator_bias_scale=1.0,
        xml_path_override=None,
        eval_raw_rewards=eval_raw_rewards,
        normalize_reward=normalize_reward,
        obs_noise_std=0.0,
        obs_noise_clip=None,
        reward_noise_std=0.0,
        action_noise_std=0.0,
        action_noise_clip=None,
        param_override="",
        param_randomize="",
        param_strict=True,
    )
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, capture_video, run_name, gamma, eval_args, seed)])
    load_normalization_stats_if_available(
        envs,
        norm_stats_path or f"{model_path}.norm_stats.npz",
        eval_raw_rewards=eval_raw_rewards,
    )
    agent = Model(envs).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()

    obs, _ = envs.reset(seed=seed)
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        actions, _, _, _ = agent.get_action_and_value(torch.Tensor(obs).to(device))
        next_obs, _, _, _, infos = envs.step(actions.cpu().numpy())
        for ep_return in extract_episode_returns_from_infos(infos):
            print(f"eval_episode={len(episodic_returns)}, episodic_return={ep_return}")
            episodic_returns += [ep_return]
        obs = next_obs

    return episodic_returns


if __name__ == "__main__":
    from huggingface_hub import hf_hub_download

    from cleanrl.ppo_continuous_action import Agent, make_env

    model_path = hf_hub_download(
        repo_id="sdpkjc/Hopper-v4-ppo_continuous_action-seed1", filename="ppo_continuous_action.cleanrl_model"
    )
    evaluate(
        model_path,
        make_env,
        "Hopper-v4",
        eval_episodes=10,
        run_name=f"eval",
        Model=Agent,
        device="cpu",
        capture_video=False,
        eval_raw_rewards=True,
    )
