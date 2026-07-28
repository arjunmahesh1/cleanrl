# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/td3/#td3_continuous_actionpy
import copy
import os
import random
import time
from dataclasses import dataclass
import math

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from cleanrl_utils.buffers import PhysicalEnsembleReplayBuffer, ReplayBuffer
from cleanrl_utils.mujoco_xml_utils import make_mujoco_env
from cleanrl_utils.perturbation_config import apply_env_perturbations


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    wandb_group: str = ""
    """the wandb run group"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""
    run_dir: str = "runs"
    """base directory for TensorBoard logs and saved models"""

    # Algorithm specific arguments
    env_id: str = "Hopper-v4"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    policy_noise: float = 0.2
    """the scale of policy noise"""
    exploration_noise: float = 0.1
    """the scale of exploration noise"""
    learning_starts: int = 25e3
    """timestep to start learning"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""
    robust_target_mode: str = "none"
    """robust critic target mode: none, tv_cap, kl_moment, kl_physical, kl_physical_radius"""
    tv_clip_q_targets: bool = False
    """if true, apply an upper cap to the bootstrapped TD3 target Q value"""
    tv_fixed_cap: float | None = None
    """fixed one-sided cap for min target Q before constructing the TD target"""
    kl_beta: float = 100.0
    """KL moment beta; larger values are closer to risk-neutral TD3"""
    kl_log_moment_exp_min: float = -80.0
    """minimum KL log-moment value before exponentiating in the moment loss"""
    kl_log_moment_exp_max: float = 20.0
    """maximum KL log-moment value before exponentiating in the moment loss"""
    kl_log_moment_target_clip: float | None = None
    """deprecated compatibility alias for symmetric KL target clipping; prefer exp min/max"""
    kl_rescale_moment_loss: bool = True
    """multiply KL moment MSE by (gamma * beta)^2 to restore its local Q-error scale"""
    reward_scale: float = 1.0
    """multiply training rewards before replay/updates; episode logs remain in raw env units"""
    kl_next_state_samples: int = 1
    """number of synthetic next-state samples in the KL exponential moment target"""
    kl_next_obs_noise_std: float = 0.0
    """standard deviation for synthetic next-observation samples; 0 disables the ensemble perturbation"""
    kl_next_obs_noise_relative: bool = True
    """scale synthetic next-observation noise by per-batch next-observation standard deviation"""
    kl_next_obs_noise_clip: float = 3.0
    """clip synthetic next-observation noise to this many stds; <=0 disables clipping"""
    kl_physical_dynamics: str = "nominal"
    """comma-separated finite dynamics support, e.g. nominal,mass:0.8,mass:1.2,actuator_gain:0.8"""
    kl_physical_weights: str = ""
    """optional comma-separated reference probabilities; empty means uniform"""
    kl_physical_verify_nominal: bool = True
    """verify that a nominal branch reproduces each live MuJoCo transition"""
    kl_physical_verify_tolerance: float = 1e-5
    """maximum observation/reward discrepancy allowed by nominal branch verification"""
    kl_radius: float = 0.1
    """fixed KL radius for kl_physical_radius"""
    kl_radius_bisection_steps: int = 40
    """dual-temperature bisection steps for the constrained physical KL target"""


def _needs_xml_perturbation(perturb) -> bool:
    if perturb is None:
        return False
    return any(
        [
            getattr(perturb, "xml_perturb", False),
            getattr(perturb, "xml_total_mass_scale", 1.0) != 1.0,
            getattr(perturb, "xml_body_mass_scale", 1.0) != 1.0,
            getattr(perturb, "xml_geom_friction_scale", 1.0) != 1.0,
            getattr(perturb, "xml_gravity_component_index", -1) >= 0
            and getattr(perturb, "xml_gravity_component_value", None) is not None,
            getattr(perturb, "xml_joint_damping_scale", 1.0) != 1.0,
            getattr(perturb, "xml_actuator_gain_scale", 1.0) != 1.0,
            getattr(perturb, "xml_actuator_bias_scale", 1.0) != 1.0,
            bool(getattr(perturb, "xml_body_name_selector", "")),
            bool(getattr(perturb, "xml_geom_name_selector", "")),
            bool(getattr(perturb, "xml_joint_name_selector", "")),
            bool(getattr(perturb, "xml_actuator_joint_selector", "")),
        ]
    )


class ScaleReward(gym.RewardWrapper):
    def __init__(self, env, scale: float):
        super().__init__(env)
        self.scale = float(scale)

    def reward(self, reward):
        return reward * self.scale


def make_env(env_id, seed, idx, capture_video, run_name, perturb=None, reward_scale: float = 1.0):
    def thunk():
        if _needs_xml_perturbation(perturb):
            env = make_mujoco_env(
                env_id,
                xml_out_dir=getattr(perturb, "xml_out_dir", "perturbed_xml"),
                run_name=run_name,
                total_mass_scale=getattr(perturb, "xml_total_mass_scale", 1.0),
                body_mass_scale=getattr(perturb, "xml_body_mass_scale", 1.0),
                body_name_selector=getattr(perturb, "xml_body_name_selector", ""),
                geom_friction_scale=getattr(perturb, "xml_geom_friction_scale", 1.0),
                geom_friction_component=getattr(perturb, "xml_geom_friction_component", -1),
                geom_name_selector=getattr(perturb, "xml_geom_name_selector", ""),
                joint_damping_scale=getattr(perturb, "xml_joint_damping_scale", 1.0),
                joint_name_selector=getattr(perturb, "xml_joint_name_selector", ""),
                actuator_gain_scale=getattr(perturb, "xml_actuator_gain_scale", 1.0),
                actuator_bias_scale=getattr(perturb, "xml_actuator_bias_scale", 1.0),
                gravity_component_index=getattr(perturb, "xml_gravity_component_index", -1),
                gravity_component_value=getattr(perturb, "xml_gravity_component_value", None),
                actuator_joint_selector=getattr(perturb, "xml_actuator_joint_selector", ""),
                xml_path_override=getattr(perturb, "xml_path_override", None),
                render_mode="rgb_array" if capture_video and idx == 0 else None,
            )
        else:
            if capture_video and idx == 0:
                env = gym.make(env_id, render_mode="rgb_array")
            else:
                env = gym.make(env_id)
        env = apply_env_perturbations(
            env,
            obs_noise_std=getattr(perturb, "obs_noise_std", 0.0),
            obs_noise_clip=getattr(perturb, "obs_noise_clip", None),
            reward_noise_std=getattr(perturb, "reward_noise_std", 0.0),
            action_noise_std=getattr(perturb, "action_noise_std", 0.0),
            action_noise_clip=getattr(perturb, "action_noise_clip", None),
            action_replace_prob=getattr(perturb, "action_replace_prob", 0.0),
            param_override_spec=getattr(perturb, "param_override", ""),
            param_randomize_spec=getattr(perturb, "param_randomize", ""),
            param_strict=getattr(perturb, "param_strict", True),
            seed=seed,
        )
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if reward_scale != 1.0:
            env = ScaleReward(env, reward_scale)
        env.action_space.seed(seed)
        return env

    return thunk


def make_sync_vector_env(env_fns):
    """Use same-step autoreset on modern Gymnasium while retaining 0.29 compatibility."""
    autoreset_mode = getattr(gym.vector, "AutoresetMode", None)
    if autoreset_mode is None:
        return gym.vector.SyncVectorEnv(env_fns)
    return gym.vector.SyncVectorEnv(env_fns, autoreset_mode=autoreset_mode.SAME_STEP)


def extract_episode_statistics(infos: dict) -> list[tuple[float, int]]:
    """Normalize Gymnasium 0.29 and 1.x vector episode-info layouts."""

    def unpack_episode(episode: dict, mask=None) -> list[tuple[float, int]]:
        rewards = np.asarray(episode["r"]).reshape(-1)
        lengths = np.asarray(episode["l"]).reshape(-1)
        if mask is None:
            mask = episode.get("_r")
        mask_array = (
            np.ones(rewards.shape, dtype=bool)
            if mask is None
            else np.asarray(mask, dtype=bool).reshape(-1)
        )
        return [
            (float(rewards[index]), int(lengths[index]))
            for index in range(min(len(rewards), len(lengths), len(mask_array)))
            if mask_array[index]
        ]

    final_infos = infos.get("final_info")
    if isinstance(final_infos, dict):
        episode = final_infos.get("episode")
        if episode is not None:
            return unpack_episode(episode, final_infos.get("_episode"))
    elif final_infos is not None:
        completed = []
        for info in final_infos:
            if info is not None and "episode" in info:
                completed.extend(unpack_episode(info["episode"]))
        if completed:
            return completed

    episode = infos.get("episode")
    if episode is not None:
        return unpack_episode(episode, infos.get("_episode"))
    return []


def final_observations_from_infos(infos: dict):
    """Return terminal observations and validity mask across Gymnasium versions."""
    for key in ("final_observation", "final_obs"):
        observations = infos.get(key)
        if observations is not None:
            mask = infos.get(f"_{key}")
            if mask is None:
                mask = np.ones(len(observations), dtype=bool)
            return observations, np.asarray(mask, dtype=bool)
    return None, None


def parse_physical_dynamics_spec(spec: str):
    """Parse a finite MuJoCo dynamics support for KL-regularized targets."""
    members = []
    for raw_token in spec.split(","):
        token = raw_token.strip()
        if not token:
            continue
        if token == "nominal":
            members.append(("nominal", {}))
            continue
        if ":" not in token:
            raise ValueError(
                f"invalid KL physical dynamics token {token!r}; expected nominal or kind:factor"
            )
        kind, raw_factor = token.split(":", 1)
        factor = float(raw_factor)
        if factor <= 0:
            raise ValueError(f"KL physical dynamics factors must be positive, got {token!r}")
        argument_by_kind = {
            "mass": "total_mass_scale",
            "friction": "geom_friction_scale",
            "damping": "joint_damping_scale",
            "actuator_gain": "actuator_gain_scale",
        }
        if kind not in argument_by_kind:
            raise ValueError(
                f"unsupported KL physical dynamics kind {kind!r}; "
                f"choose from {sorted(argument_by_kind)}"
            )
        members.append((f"{kind}:{factor:g}", {argument_by_kind[kind]: factor}))
    if not members:
        raise ValueError("--kl-physical-dynamics must contain at least one member")
    return members


def parse_physical_dynamics_weights(raw_weights: str, num_dynamics: int) -> np.ndarray:
    if raw_weights.strip():
        weights = np.asarray([float(value.strip()) for value in raw_weights.split(",")], dtype=np.float64)
        if len(weights) != num_dynamics:
            raise ValueError(
                f"--kl-physical-weights contains {len(weights)} values for {num_dynamics} dynamics members"
            )
        if np.any(weights <= 0):
            raise ValueError("--kl-physical-weights must all be strictly positive")
        weights = weights / weights.sum()
    else:
        weights = np.full(num_dynamics, 1.0 / num_dynamics, dtype=np.float64)
    return weights


def make_physical_dynamics_ensemble(args: Args, run_path: str):
    members = parse_physical_dynamics_spec(args.kl_physical_dynamics)
    weights = parse_physical_dynamics_weights(args.kl_physical_weights, len(members))
    ensemble = []
    xml_dir = os.path.join(run_path, "kl_physical_xml")
    for index, (label, xml_kwargs) in enumerate(members):
        if label == "nominal":
            env = gym.make(args.env_id)
        else:
            safe_label = label.replace(":", "_").replace(".", "p")
            env = make_mujoco_env(
                args.env_id,
                xml_out_dir=xml_dir,
                run_name=f"{safe_label}_{index}",
                **xml_kwargs,
            )
        env.reset(seed=args.seed + 10_000 + index)
        ensemble.append(env)
    return ensemble, [label for label, _ in members], weights


def step_physical_dynamics_ensemble(envs, ensemble, actions: np.ndarray, reward_scale: float):
    """Branch one live MuJoCo state through every fixed dynamics member."""
    if envs.num_envs != 1:
        raise ValueError("KL physical dynamics currently requires --num-envs 1")
    source = envs.envs[0].unwrapped
    qpos = source.data.qpos.copy()
    qvel = source.data.qvel.copy()
    source_time = float(source.data.time)
    next_observations = []
    rewards = []
    dones = []
    for candidate in ensemble:
        target = candidate.unwrapped
        target.set_state(qpos, qvel)
        target.data.time = source_time
        next_obs, reward, terminated, _, _ = target.step(actions[0])
        next_observations.append(np.asarray(next_obs))
        rewards.append(float(reward) * reward_scale)
        dones.append(float(terminated))
    return (
        np.asarray(next_observations)[None, ...],
        np.asarray(rewards, dtype=np.float32)[None, ...],
        np.asarray(dones, dtype=np.float32)[None, ...],
    )


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape),
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ValueNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def clamp_log_moment_for_exp(log_moment: torch.Tensor, args: Args) -> torch.Tensor:
    return torch.clamp(log_moment, min=args.kl_log_moment_exp_min, max=args.kl_log_moment_exp_max)


def kl_regularized_discrete_target(
    joint_returns: torch.Tensor,
    beta: float,
    log_reference_weights: torch.Tensor,
):
    """Return the finite-support KL value and its optimal adversarial weights."""
    if joint_returns.shape[1] == 1:
        return joint_returns[:, 0], torch.ones_like(joint_returns)
    reference_weights = torch.exp(log_reference_weights)
    reference_target = torch.sum(reference_weights * joint_returns, dim=1, keepdim=True)
    centered_returns = joint_returns - reference_target
    adversarial_logits = log_reference_weights - centered_returns / beta
    adversarial_weights = torch.softmax(adversarial_logits, dim=1)
    robust_target = reference_target.view(-1) - beta * torch.logsumexp(adversarial_logits, dim=1)
    return robust_target, adversarial_weights


def kl_constrained_discrete_target(
    joint_returns: torch.Tensor,
    radius: float,
    log_reference_weights: torch.Tensor,
    bisection_steps: int,
):
    """Solve the finite-support KL-ball inner problem for every batch row."""
    batch_size, num_dynamics = joint_returns.shape
    reference_weights = torch.exp(log_reference_weights).expand(batch_size, num_dynamics)
    if num_dynamics == 1:
        zeros = torch.zeros(batch_size, dtype=joint_returns.dtype, device=joint_returns.device)
        return joint_returns[:, 0], torch.ones_like(joint_returns), zeros, zeros
    if radius == 0:
        target = torch.sum(reference_weights * joint_returns, dim=1)
        zeros = torch.zeros(batch_size, dtype=joint_returns.dtype, device=joint_returns.device)
        return target, reference_weights, torch.full_like(zeros, float("inf")), zeros

    minimum_returns = torch.min(joint_returns, dim=1, keepdim=True).values
    worst_mask = torch.isclose(
        joint_returns,
        minimum_returns,
        rtol=1e-6,
        atol=1e-7,
    )
    worst_reference_weights = reference_weights * worst_mask
    worst_reference_mass = worst_reference_weights.sum(dim=1).clamp_min(1e-12)
    maximum_useful_radius = -torch.log(worst_reference_mass)
    value_range = torch.max(joint_returns, dim=1).values - torch.min(joint_returns, dim=1).values
    saturated = radius >= maximum_useful_radius
    indistinguishable = value_range <= 1e-8

    scale = value_range.clamp_min(1e-6)
    beta_low = scale * 1e-7
    beta_high = scale * 1e7
    centered_returns = joint_returns - torch.sum(
        reference_weights * joint_returns,
        dim=1,
        keepdim=True,
    )

    for _ in range(bisection_steps):
        beta_mid = torch.sqrt(beta_low * beta_high)
        logits = log_reference_weights - centered_returns / beta_mid.unsqueeze(1)
        weights_mid = torch.softmax(logits, dim=1)
        kl_mid = torch.sum(
            weights_mid
            * (torch.log(weights_mid.clamp_min(1e-12)) - log_reference_weights),
            dim=1,
        )
        beta_low = torch.where(kl_mid > radius, beta_mid, beta_low)
        beta_high = torch.where(kl_mid > radius, beta_high, beta_mid)

    effective_beta = beta_high
    adversarial_logits = log_reference_weights - centered_returns / effective_beta.unsqueeze(1)
    adversarial_weights = torch.softmax(adversarial_logits, dim=1)

    # Among tied minimizers, this is the minimum-KL distribution that attains
    # the worst value. It avoids reporting a fictitious extra radius for a
    # one-hot choice when several support members have the same return.
    worst_weights = worst_reference_weights / worst_reference_mass.unsqueeze(1)
    adversarial_weights = torch.where(
        saturated.unsqueeze(1),
        worst_weights,
        adversarial_weights,
    )
    adversarial_weights = torch.where(
        indistinguishable.unsqueeze(1),
        reference_weights,
        adversarial_weights,
    )
    effective_beta = torch.where(saturated, torch.zeros_like(effective_beta), effective_beta)
    effective_beta = torch.where(
        indistinguishable,
        torch.full_like(effective_beta, float("inf")),
        effective_beta,
    )
    achieved_radius = torch.sum(
        adversarial_weights
        * (torch.log(adversarial_weights.clamp_min(1e-12)) - log_reference_weights),
        dim=1,
    )
    robust_target = torch.sum(adversarial_weights * joint_returns, dim=1)
    return robust_target, adversarial_weights, effective_beta, achieved_radius


def build_kl_next_observation_samples(next_observations: torch.Tensor, args: Args) -> torch.Tensor:
    """Build a small empirical next-state distribution around replay next observations.

    The first sample is always the replay next observation. Additional samples are
    local perturbations, giving the KL exponential moment a non-degenerate support
    for diagnostic experiments without requiring a learned dynamics model.
    """

    num_samples = int(args.kl_next_state_samples)
    if num_samples <= 1:
        return next_observations.unsqueeze(1)

    samples = next_observations.unsqueeze(1).repeat(1, num_samples, 1)
    if args.kl_next_obs_noise_std <= 0:
        return samples

    if args.kl_next_obs_noise_relative:
        scale = next_observations.detach().std(dim=0, unbiased=False).clamp_min(1e-3).view(1, 1, -1)
    else:
        scale = 1.0

    noise = torch.randn_like(samples[:, 1:, :]) * args.kl_next_obs_noise_std * scale
    if args.kl_next_obs_noise_clip > 0:
        noise_limit = args.kl_next_obs_noise_clip * args.kl_next_obs_noise_std * scale
        noise = torch.clamp(noise, min=-noise_limit, max=noise_limit)
    samples[:, 1:, :] = samples[:, 1:, :] + noise
    return samples


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.single_action_space.high - env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.single_action_space.high + env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias


if __name__ == "__main__":

    args = tyro.cli(Args)
    if args.tv_clip_q_targets and args.robust_target_mode == "none":
        args.robust_target_mode = "tv_cap"
    if args.robust_target_mode not in {
        "none",
        "tv_cap",
        "kl_moment",
        "kl_physical",
        "kl_physical_radius",
    }:
        raise ValueError(
            "--robust-target-mode must be one of: "
            "none, tv_cap, kl_moment, kl_physical, kl_physical_radius"
        )
    if args.robust_target_mode == "tv_cap" and args.tv_fixed_cap is None:
        raise ValueError("--tv-fixed-cap is required when --robust-target-mode tv_cap")
    if args.robust_target_mode != "tv_cap" and args.tv_clip_q_targets:
        raise ValueError("--tv-clip-q-targets is only valid with --robust-target-mode tv_cap")
    if args.robust_target_mode in {"kl_moment", "kl_physical"} and args.kl_beta <= 0:
        raise ValueError("--kl-beta must be positive for KL robust target modes")
    if args.reward_scale <= 0:
        raise ValueError("--reward-scale must be positive")
    if args.kl_next_state_samples < 1:
        raise ValueError("--kl-next-state-samples must be at least 1")
    if args.kl_next_obs_noise_std < 0:
        raise ValueError("--kl-next-obs-noise-std must be non-negative")
    if args.kl_log_moment_target_clip is not None:
        clip = abs(float(args.kl_log_moment_target_clip))
        args.kl_log_moment_exp_min = -clip
        args.kl_log_moment_exp_max = min(clip, args.kl_log_moment_exp_max)
    if args.robust_target_mode == "kl_moment" and args.kl_log_moment_exp_min >= args.kl_log_moment_exp_max:
        raise ValueError("--kl-log-moment-exp-min must be smaller than --kl-log-moment-exp-max")
    if args.robust_target_mode in {"kl_physical", "kl_physical_radius"} and args.num_envs != 1:
        raise ValueError("physical KL robust target modes currently require --num-envs 1")
    if args.kl_physical_verify_tolerance <= 0:
        raise ValueError("--kl-physical-verify-tolerance must be positive")
    if args.kl_radius < 0:
        raise ValueError("--kl-radius must be non-negative")
    if args.kl_radius_bisection_steps < 1:
        raise ValueError("--kl-radius-bisection-steps must be at least 1")
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    run_path = os.path.join(args.run_dir, run_name)
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            group=args.wandb_group or None,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(run_path)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = make_sync_vector_env(
        [
            make_env(
                args.env_id,
                args.seed + i,
                i,
                args.capture_video,
                run_name,
                reward_scale=args.reward_scale,
            )
            for i in range(args.num_envs)
        ]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    physical_dynamics_ensemble = []
    physical_dynamics_labels = []
    physical_dynamics_weights = None
    physical_dynamics_weights_tensor = None
    physical_log_weights = None
    if args.robust_target_mode in {"kl_physical", "kl_physical_radius"}:
        physical_dynamics_ensemble, physical_dynamics_labels, physical_dynamics_weights = (
            make_physical_dynamics_ensemble(args, run_path)
        )
        physical_dynamics_weights_tensor = torch.as_tensor(
            physical_dynamics_weights,
            dtype=torch.float32,
            device=device,
        )
        physical_log_weights = torch.log(physical_dynamics_weights_tensor)
        writer.add_text("kl_physical/dynamics_labels", ", ".join(physical_dynamics_labels))
        writer.add_text(
            "kl_physical/dynamics_weights",
            ", ".join(f"{weight:.8g}" for weight in physical_dynamics_weights),
        )

    actor = Actor(envs).to(device)
    qf1 = QNetwork(envs).to(device)
    qf2 = QNetwork(envs).to(device)
    qf1_target = QNetwork(envs).to(device)
    qf2_target = QNetwork(envs).to(device)
    vf = ValueNetwork(envs).to(device) if args.robust_target_mode == "kl_moment" else None
    vf_target = ValueNetwork(envs).to(device) if args.robust_target_mode == "kl_moment" else None
    target_actor = Actor(envs).to(device)
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    if vf_target is not None:
        vf_target.load_state_dict(vf.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.learning_rate)
    vf_optimizer = optim.Adam(vf.parameters(), lr=args.learning_rate) if vf is not None else None
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate)

    replay_observation_space = copy.deepcopy(envs.single_observation_space)
    replay_observation_space.dtype = np.float32
    if args.robust_target_mode in {"kl_physical", "kl_physical_radius"}:
        rb = PhysicalEnsembleReplayBuffer(
            args.buffer_size,
            replay_observation_space,
            envs.single_action_space,
            device,
            num_dynamics=len(physical_dynamics_ensemble),
            n_envs=args.num_envs,
            handle_timeout_termination=False,
        )
    else:
        rb = ReplayBuffer(
            args.buffer_size,
            replay_observation_space,
            envs.single_action_space,
            device,
            n_envs=args.num_envs,
            handle_timeout_termination=False,
        )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            with torch.no_grad():
                actions = actor(torch.Tensor(obs).to(device))
                actions += torch.normal(0, actor.action_scale * args.exploration_noise)
                actions = actions.cpu().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)

        physical_outcomes = None
        if args.robust_target_mode in {"kl_physical", "kl_physical_radius"}:
            physical_outcomes = step_physical_dynamics_ensemble(
                envs,
                physical_dynamics_ensemble,
                actions,
                args.reward_scale,
            )

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        for episodic_return, episodic_length in extract_episode_statistics(infos):
            print(f"global_step={global_step}, episodic_return={episodic_return}")
            writer.add_scalar("charts/episodic_return", episodic_return, global_step)
            writer.add_scalar("charts/episodic_length", episodic_length, global_step)

        # Save the true terminal observation when same-step autoreset returns reset observations.
        real_next_obs = next_obs.copy()
        final_observations, final_observation_mask = final_observations_from_infos(infos)
        for idx, trunc in enumerate(truncations):
            if (
                (terminations[idx] or trunc)
                and final_observations is not None
                and final_observation_mask[idx]
                and final_observations[idx] is not None
            ):
                real_next_obs[idx] = final_observations[idx]
        if args.robust_target_mode in {"kl_physical", "kl_physical_radius"}:
            ensemble_next_observations, ensemble_rewards, ensemble_dones = physical_outcomes
            if args.kl_physical_verify_nominal and "nominal" in physical_dynamics_labels:
                nominal_index = physical_dynamics_labels.index("nominal")
                obs_error = float(
                    np.max(np.abs(ensemble_next_observations[0, nominal_index] - real_next_obs[0]))
                )
                reward_error = abs(float(ensemble_rewards[0, nominal_index] - rewards[0]))
                done_matches = bool(ensemble_dones[0, nominal_index] == float(terminations[0]))
                if (
                    obs_error > args.kl_physical_verify_tolerance
                    or reward_error > args.kl_physical_verify_tolerance
                    or not done_matches
                ):
                    raise RuntimeError(
                        "nominal KL physical branch did not reproduce the live transition: "
                        f"obs_error={obs_error}, reward_error={reward_error}, done_matches={done_matches}"
                    )
                if global_step % 100 == 0:
                    writer.add_scalar("kl_physical/nominal_obs_max_abs_error", obs_error, global_step)
                    writer.add_scalar("kl_physical/nominal_reward_abs_error", reward_error, global_step)
            rb.add(
                obs,
                real_next_obs,
                actions,
                rewards,
                terminations,
                infos,
                ensemble_next_observations=ensemble_next_observations,
                ensemble_rewards=ensemble_rewards,
                ensemble_dones=ensemble_dones,
            )
        else:
            rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                clipped_noise = (torch.randn_like(data.actions, device=device) * args.policy_noise).clamp(
                    -args.noise_clip, args.noise_clip
                ) * target_actor.action_scale

                next_state_actions = (target_actor(data.next_observations) + clipped_noise).clamp(
                    envs.single_action_space.low[0], envs.single_action_space.high[0]
                )
                if args.robust_target_mode == "kl_moment":
                    # KL mode critics output ell=log f, where implied Q=-gamma*beta*ell.
                    # The Bellman residual is fit in f-space. With kl_next_state_samples>1,
                    # the target estimates a local empirical exponential moment over
                    # plausible next observations rather than a single deterministic sample.
                    kl_next_obs_samples = build_kl_next_observation_samples(data.next_observations, args)
                    batch_size, num_next_samples, obs_dim = kl_next_obs_samples.shape
                    g_next_target = vf_target(kl_next_obs_samples.reshape(batch_size * num_next_samples, obs_dim)).view(
                        batch_size, num_next_samples
                    )
                    done_mask = (1 - data.dones.flatten()).view(-1, 1)
                    kl_next_log_terms = done_mask * g_next_target / args.kl_beta
                    kl_next_log_moment = torch.logsumexp(kl_next_log_terms, dim=1) - math.log(num_next_samples)
                    kl_log_moment_target = -data.rewards.flatten() / (args.gamma * args.kl_beta)
                    kl_log_moment_target = kl_log_moment_target + kl_next_log_moment
                    kl_log_moment_target_pre_clip = kl_log_moment_target
                    kl_log_moment_target_for_exp = clamp_log_moment_for_exp(kl_log_moment_target, args)
                    kl_moment_target = torch.exp(kl_log_moment_target_for_exp)
                    min_qf_next_target_pre_clip = None
                    min_qf_next_target_post_clip = None
                    target_q_clip_fraction = None
                    target_q_clip_count = None
                    target_q_excess_mean = None
                    physical_joint_returns = None
                    physical_reference_target = None
                    physical_worst_target = None
                    physical_pessimism_gap = None
                elif args.robust_target_mode in {"kl_physical", "kl_physical_radius"}:
                    batch_size, num_dynamics, obs_dim = data.ensemble_next_observations.shape
                    flat_next_observations = data.ensemble_next_observations.reshape(
                        batch_size * num_dynamics,
                        obs_dim,
                    )
                    common_target_noise = clipped_noise.unsqueeze(1).expand(
                        batch_size,
                        num_dynamics,
                        clipped_noise.shape[-1],
                    )
                    flat_next_actions = (
                        target_actor(flat_next_observations)
                        + common_target_noise.reshape(batch_size * num_dynamics, -1)
                    ).clamp(
                        envs.single_action_space.low[0],
                        envs.single_action_space.high[0],
                    )
                    physical_qf1_next = qf1_target(flat_next_observations, flat_next_actions).view(
                        batch_size,
                        num_dynamics,
                    )
                    physical_qf2_next = qf2_target(flat_next_observations, flat_next_actions).view(
                        batch_size,
                        num_dynamics,
                    )
                    physical_min_q_next = torch.min(physical_qf1_next, physical_qf2_next)
                    physical_joint_returns = data.ensemble_rewards + (
                        1 - data.ensemble_dones
                    ) * args.gamma * physical_min_q_next
                    physical_reference_target = torch.sum(
                        physical_dynamics_weights_tensor * physical_joint_returns,
                        dim=1,
                    )
                    physical_worst_target = torch.min(physical_joint_returns, dim=1).values
                    if args.robust_target_mode == "kl_physical_radius":
                        (
                            next_q_value,
                            physical_adversarial_weights,
                            physical_effective_beta,
                            physical_implicit_kl_radius,
                        ) = kl_constrained_discrete_target(
                            physical_joint_returns,
                            args.kl_radius,
                            physical_log_weights,
                            args.kl_radius_bisection_steps,
                        )
                    else:
                        next_q_value, physical_adversarial_weights = kl_regularized_discrete_target(
                            physical_joint_returns,
                            args.kl_beta,
                            physical_log_weights,
                        )
                        physical_effective_beta = torch.full(
                            (batch_size,),
                            args.kl_beta,
                            dtype=physical_joint_returns.dtype,
                            device=physical_joint_returns.device,
                        )
                        physical_implicit_kl_radius = torch.sum(
                            physical_adversarial_weights
                            * (
                                torch.log(physical_adversarial_weights.clamp_min(1e-12))
                                - physical_log_weights
                            ),
                            dim=1,
                        )
                    physical_pessimism_gap = physical_reference_target - next_q_value
                    physical_worst_member_index = torch.argmin(physical_joint_returns, dim=1, keepdim=True)
                    physical_worst_member_weight = torch.gather(
                        physical_adversarial_weights,
                        dim=1,
                        index=physical_worst_member_index,
                    ).view(-1)
                    physical_adversarial_entropy = -torch.sum(
                        physical_adversarial_weights
                        * torch.log(physical_adversarial_weights.clamp_min(1e-12)),
                        dim=1,
                    )
                    min_qf_next_target_pre_clip = None
                    min_qf_next_target_post_clip = None
                    target_q_clip_fraction = None
                    target_q_clip_count = None
                    target_q_excess_mean = None
                    kl_log_moment_target_pre_clip = None
                    kl_log_moment_target_for_exp = None
                    kl_moment_target = None
                else:
                    qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                    qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                    min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
                    min_qf_next_target_pre_clip = min_qf_next_target.view(-1)
                    target_q_clip_fraction = None
                    target_q_clip_count = None
                    target_q_excess_mean = None
                    if args.robust_target_mode == "tv_cap":
                        cap = float(args.tv_fixed_cap)
                        target_q_clip_fraction = (min_qf_next_target_pre_clip > cap).float().mean()
                        target_q_clip_count = (min_qf_next_target_pre_clip > cap).float().sum()
                        target_q_excess_mean = torch.clamp(min_qf_next_target_pre_clip - cap, min=0.0).mean()
                        min_qf_next_target = torch.clamp(min_qf_next_target, max=cap)
                    min_qf_next_target_post_clip = min_qf_next_target.view(-1)
                    next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)
                    kl_log_moment_target_pre_clip = None
                    kl_log_moment_target_for_exp = None
                    kl_moment_target = None

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            if args.robust_target_mode == "kl_moment":
                kl_log_moment1_for_exp = clamp_log_moment_for_exp(qf1_a_values, args)
                kl_log_moment2_for_exp = clamp_log_moment_for_exp(qf2_a_values, args)
                kl_moment1 = torch.exp(kl_log_moment1_for_exp)
                kl_moment2 = torch.exp(kl_log_moment2_for_exp)
                qf1_moment_mse = F.mse_loss(kl_moment1, kl_moment_target)
                qf2_moment_mse = F.mse_loss(kl_moment2, kl_moment_target)
                kl_moment_loss_scale = (args.gamma * args.kl_beta) ** 2 if args.kl_rescale_moment_loss else 1.0
                qf1_loss = kl_moment_loss_scale * qf1_moment_mse
                qf2_loss = kl_moment_loss_scale * qf2_moment_mse
            else:
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            vf_loss = None
            vf_values = None
            vf_target_values = None
            if args.robust_target_mode == "kl_moment":
                with torch.no_grad():
                    policy_actions_for_value = actor(data.observations)
                    ell1_policy = qf1(data.observations, policy_actions_for_value).view(-1)
                    ell2_policy = qf2(data.observations, policy_actions_for_value).view(-1)
                    ell_policy = torch.max(ell1_policy, ell2_policy)
                    vf_target_values = args.gamma * args.kl_beta * ell_policy
                vf_values = vf(data.observations).view(-1)
                vf_loss = F.mse_loss(vf_values, vf_target_values)
                vf_optimizer.zero_grad()
                vf_loss.backward()
                vf_optimizer.step()

            if global_step % args.policy_frequency == 0:
                if args.robust_target_mode == "kl_moment":
                    actor_actions = actor(data.observations)
                    actor_ell1 = qf1(data.observations, actor_actions)
                    actor_ell2 = qf2(data.observations, actor_actions)
                    # Larger ell means smaller implied Q. Multiplying by gamma*beta
                    # preserves the optimum and restores Q-scale actor gradients.
                    actor_loss = args.gamma * args.kl_beta * torch.max(actor_ell1, actor_ell2).mean()
                else:
                    actor_loss = -qf1(data.observations, actor(data.observations)).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # update the target network
                for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                if args.robust_target_mode == "kl_moment":
                    for param, target_param in zip(vf.parameters(), vf_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                if args.robust_target_mode == "kl_moment":
                    implied_q1 = -args.gamma * args.kl_beta * qf1_a_values
                    implied_q2 = -args.gamma * args.kl_beta * qf2_a_values
                    implied_q_target = -args.gamma * args.kl_beta * kl_log_moment_target_pre_clip
                    writer.add_scalar("kl/beta", args.kl_beta, global_step)
                    writer.add_scalar("kl/moment_loss_scale", kl_moment_loss_scale, global_step)
                    writer.add_scalar("kl/qf1_moment_mse_unscaled", qf1_moment_mse.item(), global_step)
                    writer.add_scalar("kl/qf2_moment_mse_unscaled", qf2_moment_mse.item(), global_step)
                    writer.add_scalar("kl/log_moment1_mean", qf1_a_values.mean().item(), global_step)
                    writer.add_scalar("kl/log_moment2_mean", qf2_a_values.mean().item(), global_step)
                    writer.add_scalar("kl/log_moment1_mean_for_exp", kl_log_moment1_for_exp.mean().item(), global_step)
                    writer.add_scalar("kl/log_moment2_mean_for_exp", kl_log_moment2_for_exp.mean().item(), global_step)
                    writer.add_scalar("kl/log_moment_target_mean_raw", kl_log_moment_target_pre_clip.mean().item(), global_step)
                    writer.add_scalar("kl/log_moment_target_mean_for_exp", kl_log_moment_target_for_exp.mean().item(), global_step)
                    writer.add_scalar("kl/log_moment_target_min_for_exp", kl_log_moment_target_for_exp.min().item(), global_step)
                    writer.add_scalar("kl/log_moment_target_max_for_exp", kl_log_moment_target_for_exp.max().item(), global_step)
                    writer.add_scalar(
                        "kl/log_moment1_clamp_fraction",
                        (kl_log_moment1_for_exp != qf1_a_values).float().mean().item(),
                        global_step,
                    )
                    writer.add_scalar(
                        "kl/log_moment2_clamp_fraction",
                        (kl_log_moment2_for_exp != qf2_a_values).float().mean().item(),
                        global_step,
                    )
                    writer.add_scalar(
                        "kl/log_moment_target_clamp_fraction",
                        (kl_log_moment_target_for_exp != kl_log_moment_target_pre_clip).float().mean().item(),
                        global_step,
                    )
                    writer.add_scalar("kl/moment1_mean", kl_moment1.mean().item(), global_step)
                    writer.add_scalar("kl/moment2_mean", kl_moment2.mean().item(), global_step)
                    writer.add_scalar("kl/moment_target_mean", kl_moment_target.mean().item(), global_step)
                    writer.add_scalar("kl/moment_target_min", kl_moment_target.min().item(), global_step)
                    writer.add_scalar("kl/moment_target_max", kl_moment_target.max().item(), global_step)
                    writer.add_scalar("kl/moment_target_gt_1e_minus_12_frac", (kl_moment_target > 1e-12).float().mean().item(), global_step)
                    writer.add_scalar("kl/moment_target_gt_1e_minus_6_frac", (kl_moment_target > 1e-6).float().mean().item(), global_step)
                    writer.add_scalar("kl/log_moment_exp_min", args.kl_log_moment_exp_min, global_step)
                    writer.add_scalar("kl/log_moment_exp_max", args.kl_log_moment_exp_max, global_step)
                    writer.add_scalar("kl/next_state_samples", args.kl_next_state_samples, global_step)
                    writer.add_scalar("kl/next_obs_noise_std", args.kl_next_obs_noise_std, global_step)
                    writer.add_scalar("kl/next_log_moment_mean", kl_next_log_moment.mean().item(), global_step)
                    writer.add_scalar("kl/g_next_target_mean", g_next_target.mean().item(), global_step)
                    writer.add_scalar("kl/g_next_target_std", g_next_target.std(unbiased=False).item(), global_step)
                    writer.add_scalar("kl/reward_scale", args.reward_scale, global_step)
                    writer.add_scalar("kl/scaled_reward_mean", data.rewards.mean().item(), global_step)
                    writer.add_scalar("kl/value_g_mean", vf_values.mean().item(), global_step)
                    writer.add_scalar("kl/value_g_target_mean", vf_target_values.mean().item(), global_step)
                    writer.add_scalar("kl/value_g_loss", vf_loss.item(), global_step)
                    writer.add_scalar("kl/implied_q1_mean", implied_q1.mean().item(), global_step)
                    writer.add_scalar("kl/implied_q2_mean", implied_q2.mean().item(), global_step)
                    writer.add_scalar("kl/implied_q_target_mean", implied_q_target.mean().item(), global_step)
                    writer.add_scalar("kl/implied_q1_target_mse", F.mse_loss(implied_q1, implied_q_target).item(), global_step)
                    writer.add_scalar("kl/implied_q2_target_mse", F.mse_loss(implied_q2, implied_q_target).item(), global_step)
                    writer.add_scalar("kl/implied_q1_p95", torch.quantile(implied_q1, 0.95).item(), global_step)
                    writer.add_scalar("kl/implied_q1_p99", torch.quantile(implied_q1, 0.99).item(), global_step)
                elif args.robust_target_mode in {"kl_physical", "kl_physical_radius"}:
                    if args.robust_target_mode == "kl_physical":
                        writer.add_scalar("kl_physical/beta", args.kl_beta, global_step)
                    else:
                        writer.add_scalar("kl_physical/requested_radius", args.kl_radius, global_step)
                        finite_beta = physical_effective_beta[torch.isfinite(physical_effective_beta)]
                        if finite_beta.numel() > 0:
                            writer.add_scalar(
                                "kl_physical/effective_beta_median",
                                torch.median(finite_beta).item(),
                                global_step,
                            )
                            writer.add_scalar(
                                "kl_physical/effective_beta_mean",
                                finite_beta.mean().item(),
                                global_step,
                            )
                        writer.add_scalar(
                            "kl_physical/worst_case_saturation_fraction",
                            (physical_effective_beta == 0).float().mean().item(),
                            global_step,
                        )
                    writer.add_scalar("kl_physical/num_dynamics", len(physical_dynamics_labels), global_step)
                    writer.add_scalar(
                        "kl_physical/joint_return_mean",
                        physical_joint_returns.mean().item(),
                        global_step,
                    )
                    writer.add_scalar(
                        "kl_physical/joint_return_std_across_dynamics",
                        physical_joint_returns.std(dim=1, unbiased=False).mean().item(),
                        global_step,
                    )
                    writer.add_scalar(
                        "kl_physical/reference_target_mean",
                        physical_reference_target.mean().item(),
                        global_step,
                    )
                    writer.add_scalar(
                        "kl_physical/robust_target_mean",
                        next_q_value.mean().item(),
                        global_step,
                    )
                    writer.add_scalar(
                        "kl_physical/worst_member_target_mean",
                        physical_worst_target.mean().item(),
                        global_step,
                    )
                    writer.add_scalar(
                        "kl_physical/pessimism_gap_mean",
                        physical_pessimism_gap.mean().item(),
                        global_step,
                    )
                    writer.add_scalar(
                        "kl_physical/pessimism_gap_p95",
                        torch.quantile(physical_pessimism_gap, 0.95).item(),
                        global_step,
                    )
                    writer.add_scalar(
                        "kl_physical/implicit_kl_radius_mean",
                        physical_implicit_kl_radius.mean().item(),
                        global_step,
                    )
                    writer.add_scalar(
                        "kl_physical/implicit_kl_radius_p95",
                        torch.quantile(physical_implicit_kl_radius, 0.95).item(),
                        global_step,
                    )
                    writer.add_scalar(
                        "kl_physical/worst_member_adversarial_weight_mean",
                        physical_worst_member_weight.mean().item(),
                        global_step,
                    )
                    for member_index, member_label in enumerate(
                        physical_dynamics_labels
                    ):
                        metric_label = (
                            member_label.replace(":", "_")
                            .replace(".", "p")
                            .replace("-", "m")
                        )
                        writer.add_scalar(
                            f"kl_physical/member_weight/{metric_label}",
                            physical_adversarial_weights[
                                :, member_index
                            ].mean().item(),
                            global_step,
                        )
                        writer.add_scalar(
                            f"kl_physical/member_worst_fraction/{metric_label}",
                            (
                                physical_worst_member_index.squeeze(1)
                                == member_index
                            )
                            .float()
                            .mean()
                            .item(),
                            global_step,
                        )
                    writer.add_scalar(
                        "kl_physical/effective_num_dynamics_mean",
                        torch.exp(physical_adversarial_entropy).mean().item(),
                        global_step,
                    )
                else:
                    writer.add_scalar("targets/min_q_next_mean_pre_clip", min_qf_next_target_pre_clip.mean().item(), global_step)
                    writer.add_scalar("targets/min_q_next_max_pre_clip", min_qf_next_target_pre_clip.max().item(), global_step)
                    writer.add_scalar("targets/min_q_next_p95_pre_clip", torch.quantile(min_qf_next_target_pre_clip, 0.95).item(), global_step)
                    writer.add_scalar("targets/min_q_next_p99_pre_clip", torch.quantile(min_qf_next_target_pre_clip, 0.99).item(), global_step)
                    writer.add_scalar("targets/min_q_next_mean_post_clip", min_qf_next_target_post_clip.mean().item(), global_step)
                    writer.add_scalar("targets/td_target_mean", next_q_value.mean().item(), global_step)
                    writer.add_scalar("targets/td_target_max", next_q_value.max().item(), global_step)
                    if args.robust_target_mode == "tv_cap":
                        writer.add_scalar("robust/td3_q_target_fixed_cap", float(args.tv_fixed_cap), global_step)
                        writer.add_scalar("robust/td3_q_target_clip_fraction", target_q_clip_fraction.item(), global_step)
                        writer.add_scalar("robust/td3_q_target_clip_count", target_q_clip_count.item(), global_step)
                        writer.add_scalar("robust/td3_q_target_excess_mean", target_q_excess_mean.item(), global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )

    if args.save_model:
        model_path = os.path.join(run_path, f"{args.exp_name}.cleanrl_model")
        if args.robust_target_mode == "kl_moment":
            torch.save((actor.state_dict(), qf1.state_dict(), qf2.state_dict(), vf.state_dict()), model_path)
        else:
            torch.save((actor.state_dict(), qf1.state_dict(), qf2.state_dict()), model_path)
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.td3_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=(Actor, QNetwork),
            device=device,
            exploration_noise=0.0,
            capture_video=False,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(
                args,
                episodic_returns,
                repo_id,
                "TD3",
                run_path,
                f"videos/{run_name}-eval",
            )

    for physical_env in physical_dynamics_ensemble:
        physical_env.close()
    envs.close()
    writer.close()
