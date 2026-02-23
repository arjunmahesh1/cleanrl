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
):
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, capture_video, run_name, gamma)])
    agent = Model(envs).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()

    obs, _ = envs.reset()
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
    )
