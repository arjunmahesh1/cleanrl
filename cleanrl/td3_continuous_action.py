# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/td3/#td3_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from cleanrl_utils.buffers import ReplayBuffer
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
    """robust critic target mode: none, tv_cap, kl_moment"""
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
    reward_scale: float = 1.0
    """multiply training rewards before replay/updates; episode logs remain in raw env units"""


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
    if args.robust_target_mode not in {"none", "tv_cap", "kl_moment"}:
        raise ValueError("--robust-target-mode must be one of: none, tv_cap, kl_moment")
    if args.robust_target_mode == "tv_cap" and args.tv_fixed_cap is None:
        raise ValueError("--tv-fixed-cap is required when --robust-target-mode tv_cap")
    if args.robust_target_mode != "tv_cap" and args.tv_clip_q_targets:
        raise ValueError("--tv-clip-q-targets is only valid with --robust-target-mode tv_cap")
    if args.robust_target_mode == "kl_moment" and args.kl_beta <= 0:
        raise ValueError("--kl-beta must be positive for --robust-target-mode kl_moment")
    if args.reward_scale <= 0:
        raise ValueError("--reward-scale must be positive")
    if args.kl_log_moment_target_clip is not None:
        clip = abs(float(args.kl_log_moment_target_clip))
        args.kl_log_moment_exp_min = -clip
        args.kl_log_moment_exp_max = min(clip, args.kl_log_moment_exp_max)
    if args.robust_target_mode == "kl_moment" and args.kl_log_moment_exp_min >= args.kl_log_moment_exp_max:
        raise ValueError("--kl-log-moment-exp-min must be smaller than --kl-log-moment-exp-max")
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
    envs = gym.vector.SyncVectorEnv(
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

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
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

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        final_observations = infos.get("final_observation")
        for idx, trunc in enumerate(truncations):
            if trunc and final_observations is not None and final_observations[idx] is not None:
                real_next_obs[idx] = final_observations[idx]
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
                    # The Bellman residual is fit in f-space so this single transition
                    # is an unbiased sample of the exponential moment.
                    g_next_target = vf_target(data.next_observations).view(-1)
                    kl_log_moment_target = -data.rewards.flatten() / (args.gamma * args.kl_beta)
                    kl_log_moment_target = kl_log_moment_target + (1 - data.dones.flatten()) * g_next_target / args.kl_beta
                    kl_log_moment_target_pre_clip = kl_log_moment_target
                    kl_log_moment_target_for_exp = clamp_log_moment_for_exp(kl_log_moment_target, args)
                    kl_moment_target = torch.exp(kl_log_moment_target_for_exp)
                    min_qf_next_target_pre_clip = None
                    min_qf_next_target_post_clip = None
                    target_q_clip_fraction = None
                    target_q_clip_count = None
                    target_q_excess_mean = None
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
                qf1_loss = F.mse_loss(kl_moment1, kl_moment_target)
                qf2_loss = F.mse_loss(kl_moment2, kl_moment_target)
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
                    writer.add_scalar("kl/beta", args.kl_beta, global_step)
                    writer.add_scalar("kl/log_moment1_mean", qf1_a_values.mean().item(), global_step)
                    writer.add_scalar("kl/log_moment2_mean", qf2_a_values.mean().item(), global_step)
                    writer.add_scalar("kl/log_moment1_mean_for_exp", kl_log_moment1_for_exp.mean().item(), global_step)
                    writer.add_scalar("kl/log_moment2_mean_for_exp", kl_log_moment2_for_exp.mean().item(), global_step)
                    writer.add_scalar("kl/log_moment_target_mean_raw", kl_log_moment_target_pre_clip.mean().item(), global_step)
                    writer.add_scalar("kl/log_moment_target_mean_for_exp", kl_log_moment_target_for_exp.mean().item(), global_step)
                    writer.add_scalar("kl/log_moment_target_min_for_exp", kl_log_moment_target_for_exp.min().item(), global_step)
                    writer.add_scalar("kl/log_moment_target_max_for_exp", kl_log_moment_target_for_exp.max().item(), global_step)
                    writer.add_scalar("kl/moment1_mean", kl_moment1.mean().item(), global_step)
                    writer.add_scalar("kl/moment2_mean", kl_moment2.mean().item(), global_step)
                    writer.add_scalar("kl/moment_target_mean", kl_moment_target.mean().item(), global_step)
                    writer.add_scalar("kl/moment_target_min", kl_moment_target.min().item(), global_step)
                    writer.add_scalar("kl/moment_target_max", kl_moment_target.max().item(), global_step)
                    writer.add_scalar("kl/moment_target_gt_1e_minus_12_frac", (kl_moment_target > 1e-12).float().mean().item(), global_step)
                    writer.add_scalar("kl/moment_target_gt_1e_minus_6_frac", (kl_moment_target > 1e-6).float().mean().item(), global_step)
                    writer.add_scalar("kl/log_moment_exp_min", args.kl_log_moment_exp_min, global_step)
                    writer.add_scalar("kl/log_moment_exp_max", args.kl_log_moment_exp_max, global_step)
                    writer.add_scalar("kl/reward_scale", args.reward_scale, global_step)
                    writer.add_scalar("kl/scaled_reward_mean", data.rewards.mean().item(), global_step)
                    writer.add_scalar("kl/value_g_mean", vf_values.mean().item(), global_step)
                    writer.add_scalar("kl/value_g_target_mean", vf_target_values.mean().item(), global_step)
                    writer.add_scalar("kl/value_g_loss", vf_loss.item(), global_step)
                    writer.add_scalar("kl/implied_q1_mean", implied_q1.mean().item(), global_step)
                    writer.add_scalar("kl/implied_q2_mean", implied_q2.mean().item(), global_step)
                    writer.add_scalar("kl/implied_q1_p95", torch.quantile(implied_q1, 0.95).item(), global_step)
                    writer.add_scalar("kl/implied_q1_p99", torch.quantile(implied_q1, 0.99).item(), global_step)
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

    envs.close()
    writer.close()
