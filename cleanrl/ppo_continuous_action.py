# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

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
    """optional W&B group name for organizing runs"""
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
    env_id: str = "HalfCheetah-v4"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
    obs_noise_std: float = 0.0
    """Gaussian observation noise std (0 to disable)"""
    obs_noise_clip: float | None = None
    """clip magnitude for observation noise (None for no clip)"""
    reward_noise_std: float = 0.0
    """Gaussian reward noise std (0 to disable)"""
    action_noise_std: float = 0.0
    """Gaussian action noise std (0 to disable)"""
    action_noise_clip: float | None = None
    """clip magnitude for action noise (None for no clip)"""
    param_override: str = ""
    """env param overrides: name[:mode]=value, comma-separated (mode: set|scale|add)"""
    param_randomize: str = ""
    """env param randomization: name[:mode]=low..high, comma-separated (mode: set|scale|add)"""
    param_strict: bool = True
    """if true, unknown env params in overrides/randomization raise errors"""
    xml_perturb: bool = False
    """if true, create a perturbed MuJoCo XML and load from it"""
    xml_out_dir: str = "perturbed_xml"
    """output directory for perturbed XML files"""
    xml_path_override: str | None = None
    """optional base XML path to perturb (defaults to env's XML)"""
    xml_body_mass_scale: float = 1.0
    """scale body mass attributes in XML"""
    xml_geom_friction_scale: float = 1.0
    """scale geom friction attributes in XML"""
    xml_joint_damping_scale: float = 1.0
    """scale joint damping attributes in XML"""
    xml_actuator_gain_scale: float = 1.0
    """scale actuator gain parameters in XML"""
    xml_actuator_bias_scale: float = 1.0
    """scale actuator bias parameters in XML"""
    tv_clip_value_targets: bool = False
    """if true, apply robust clipping/penalty to value targets"""
    tv_clip_advantages: bool = False
    """if true, apply robust clipping/penalty to advantages (off by default)"""
    tv_mode: str = "fixed_cap"
    """robust target mode: fixed_cap, upper_quantile, central_quantile, penalty"""
    tv_fixed_cap: float | None = None
    """fixed one-sided cap for tv_mode=fixed_cap (targets are min(target, tv_fixed_cap))"""
    tv_keep_prob: float = 0.99
    """quantile used by quantile modes (e.g., 0.99 keeps only the upper 99% cap)"""
    tv_penalty_alpha: float = 0.0
    """alpha for tv_mode=penalty: target <- target - alpha * std(target)"""
    tv90_clip_value_targets: bool = False
    """deprecated alias for --tv-clip-value-targets"""
    tv90_clip_advantages: bool = False
    """deprecated alias for --tv-clip-advantages"""
    tv90_mode: str | None = None
    """deprecated alias for --tv-mode (upper->upper_quantile, central->central_quantile)"""
    tv90_keep_prob: float | None = None
    """deprecated alias for --tv-keep-prob"""
    tv90_penalty_alpha: float | None = None
    """deprecated alias for --tv-penalty-alpha"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name, gamma, args=None, seed=0):
    def thunk():
        if capture_video and idx == 0:
            if args is not None and args.xml_perturb:
                env = make_mujoco_env(
                    env_id,
                    xml_out_dir=args.xml_out_dir,
                    run_name=run_name,
                    body_mass_scale=args.xml_body_mass_scale,
                    geom_friction_scale=args.xml_geom_friction_scale,
                    joint_damping_scale=args.xml_joint_damping_scale,
                    actuator_gain_scale=args.xml_actuator_gain_scale,
                    actuator_bias_scale=args.xml_actuator_bias_scale,
                    xml_path_override=args.xml_path_override,
                    render_mode="rgb_array",
                )
            else:
                env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            if args is not None and args.xml_perturb:
                env = make_mujoco_env(
                    env_id,
                    xml_out_dir=args.xml_out_dir,
                    run_name=run_name,
                    body_mass_scale=args.xml_body_mass_scale,
                    geom_friction_scale=args.xml_geom_friction_scale,
                    joint_damping_scale=args.xml_joint_damping_scale,
                    actuator_gain_scale=args.xml_actuator_gain_scale,
                    actuator_bias_scale=args.xml_actuator_bias_scale,
                    xml_path_override=args.xml_path_override,
                )
            else:
                env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = transform_observation_compat(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = transform_reward_compat(env, lambda reward: np.clip(reward, -10, 10))
        if args is not None:
            env = apply_env_perturbations(
                env,
                obs_noise_std=args.obs_noise_std,
                obs_noise_clip=args.obs_noise_clip,
                reward_noise_std=args.reward_noise_std,
                action_noise_std=args.action_noise_std,
                action_noise_clip=args.action_noise_clip,
                param_override_spec=args.param_override,
                param_randomize_spec=args.param_randomize,
                param_strict=args.param_strict,
                seed=seed,
            )
        env.action_space.seed(seed)
        return env

    return thunk


def transform_observation_compat(env: gym.Env, fn):
    # Gymnasium >=1.0 requires explicitly passing the transformed observation space.
    try:
        return gym.wrappers.TransformObservation(env, fn)
    except TypeError:
        return gym.wrappers.TransformObservation(env, fn, observation_space=env.observation_space)


def transform_reward_compat(env: gym.Env, fn):
    # Gymnasium >=1.0 may require an explicit reward space for transformed rewards.
    try:
        return gym.wrappers.TransformReward(env, fn)
    except TypeError:
        reward_space = gym.spaces.Box(
            low=np.array(-10.0, dtype=np.float32),
            high=np.array(10.0, dtype=np.float32),
            shape=(),
            dtype=np.float32,
        )
        return gym.wrappers.TransformReward(env, fn, reward_space=reward_space)


def central_quantile_clip(x: torch.Tensor, keep_prob: float):
    if keep_prob <= 0.0 or keep_prob > 1.0:
        raise ValueError("tv_keep_prob must be in (0, 1].")
    if keep_prob == 1.0:
        inf = torch.tensor(float("inf"), device=x.device, dtype=x.dtype)
        return x, -inf, inf
    tail = (1.0 - keep_prob) / 2.0
    lower = torch.quantile(x, tail)
    upper = torch.quantile(x, 1.0 - tail)
    return torch.clamp(x, lower, upper), lower, upper


def upper_quantile_cap(x: torch.Tensor, keep_prob: float):
    if keep_prob <= 0.0 or keep_prob > 1.0:
        raise ValueError("tv_keep_prob must be in (0, 1].")
    if keep_prob == 1.0:
        inf = torch.tensor(float("inf"), device=x.device, dtype=x.dtype)
        return x, inf
    upper = torch.quantile(x, keep_prob)
    return torch.clamp(x, max=upper), upper


def fixed_upper_cap(x: torch.Tensor, cap: float):
    return torch.clamp(x, max=cap)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


def save_normalization_stats(envs, model_path: str) -> str:
    # Persist NormalizeObservation/NormalizeReward statistics for reliable checkpoint evaluation.
    env = envs.envs[0]
    obs_rms = env.get_wrapper_attr("obs_rms")
    return_rms = env.get_wrapper_attr("return_rms")
    norm_path = f"{model_path}.norm_stats.npz"
    np.savez_compressed(
        norm_path,
        obs_mean=np.asarray(obs_rms.mean, dtype=np.float64),
        obs_var=np.asarray(obs_rms.var, dtype=np.float64),
        obs_count=np.asarray(obs_rms.count, dtype=np.float64),
        ret_mean=np.asarray(return_rms.mean, dtype=np.float64),
        ret_var=np.asarray(return_rms.var, dtype=np.float64),
        ret_count=np.asarray(return_rms.count, dtype=np.float64),
    )
    return norm_path


if __name__ == "__main__":
    args = tyro.cli(Args)
    # Backward compatibility for older tv90_* flags.
    if args.tv90_clip_value_targets:
        args.tv_clip_value_targets = True
    if args.tv90_clip_advantages:
        args.tv_clip_advantages = True
    if args.tv90_mode is not None:
        legacy_mode_map = {"upper": "upper_quantile", "central": "central_quantile", "penalty": "penalty"}
        args.tv_mode = legacy_mode_map.get(args.tv90_mode, args.tv90_mode)
    if (args.tv90_clip_value_targets or args.tv90_clip_advantages) and args.tv90_mode is None:
        if args.tv_mode == "fixed_cap" and args.tv_fixed_cap is None:
            # Preserve old behavior when legacy flags are used without explicit mode.
            args.tv_mode = "upper_quantile"
    if args.tv90_keep_prob is not None:
        args.tv_keep_prob = args.tv90_keep_prob
    if args.tv90_penalty_alpha is not None:
        args.tv_penalty_alpha = args.tv90_penalty_alpha

    if args.tv_keep_prob <= 0.0 or args.tv_keep_prob > 1.0:
        raise ValueError("--tv-keep-prob must be in (0, 1].")
    if args.tv_mode not in {"fixed_cap", "upper_quantile", "central_quantile", "penalty"}:
        raise ValueError("--tv-mode must be one of: fixed_cap, upper_quantile, central_quantile, penalty.")
    if args.tv_penalty_alpha < 0.0:
        raise ValueError("--tv-penalty-alpha must be >= 0.")
    if args.tv_mode == "fixed_cap" and args.tv_clip_value_targets and args.tv_fixed_cap is None:
        raise ValueError("--tv-fixed-cap must be provided when --tv-mode=fixed_cap and clipping is enabled.")
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
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
    writer = SummaryWriter(os.path.join(args.run_dir, run_name))
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
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma, args, args.seed + i) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        tv_return_lower = None
        tv_return_upper = None
        tv_return_penalty = None
        tv_return_std = None
        tv_return_fixed_cap = None
        tv_adv_lower = None
        tv_adv_upper = None
        tv_adv_penalty = None
        tv_adv_std = None
        tv_adv_fixed_cap = None
        if args.tv_clip_value_targets:
            if args.tv_mode == "fixed_cap":
                tv_return_fixed_cap = torch.tensor(float(args.tv_fixed_cap), device=b_returns.device, dtype=b_returns.dtype)
                b_returns = fixed_upper_cap(b_returns, float(args.tv_fixed_cap))
            elif args.tv_mode == "upper_quantile":
                b_returns, tv_return_upper = upper_quantile_cap(b_returns, args.tv_keep_prob)
            elif args.tv_mode == "central_quantile":
                b_returns, tv_return_lower, tv_return_upper = central_quantile_clip(b_returns, args.tv_keep_prob)
            elif args.tv_mode == "penalty":
                tv_return_std = torch.std(b_returns, unbiased=False)
                tv_return_penalty = args.tv_penalty_alpha * tv_return_std
                b_returns = b_returns - tv_return_penalty
        if args.tv_clip_advantages:
            if args.tv_mode == "fixed_cap":
                tv_adv_fixed_cap = torch.tensor(float(args.tv_fixed_cap), device=b_advantages.device, dtype=b_advantages.dtype)
                b_advantages = fixed_upper_cap(b_advantages, float(args.tv_fixed_cap))
            elif args.tv_mode == "upper_quantile":
                b_advantages, tv_adv_upper = upper_quantile_cap(b_advantages, args.tv_keep_prob)
            elif args.tv_mode == "central_quantile":
                b_advantages, tv_adv_lower, tv_adv_upper = central_quantile_clip(b_advantages, args.tv_keep_prob)
            elif args.tv_mode == "penalty":
                tv_adv_std = torch.std(b_advantages, unbiased=False)
                tv_adv_penalty = args.tv_penalty_alpha * tv_adv_std
                b_advantages = b_advantages - tv_adv_penalty

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        if tv_return_lower is not None:
            writer.add_scalar("robust/tv_return_lower", tv_return_lower.item(), global_step)
        if tv_return_upper is not None:
            writer.add_scalar("robust/tv_return_upper", tv_return_upper.item(), global_step)
        if tv_return_fixed_cap is not None:
            writer.add_scalar("robust/tv_return_fixed_cap", tv_return_fixed_cap.item(), global_step)
        if tv_return_penalty is not None:
            writer.add_scalar("robust/tv_return_penalty", tv_return_penalty.item(), global_step)
            writer.add_scalar("robust/tv_return_std", tv_return_std.item(), global_step)
        if tv_adv_lower is not None:
            writer.add_scalar("robust/tv_adv_lower", tv_adv_lower.item(), global_step)
        if tv_adv_upper is not None:
            writer.add_scalar("robust/tv_adv_upper", tv_adv_upper.item(), global_step)
        if tv_adv_fixed_cap is not None:
            writer.add_scalar("robust/tv_adv_fixed_cap", tv_adv_fixed_cap.item(), global_step)
        if tv_adv_penalty is not None:
            writer.add_scalar("robust/tv_adv_penalty", tv_adv_penalty.item(), global_step)
            writer.add_scalar("robust/tv_adv_std", tv_adv_std.item(), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        model_path = os.path.join(args.run_dir, run_name, f"{args.exp_name}.cleanrl_model")
        Path(os.path.dirname(model_path)).mkdir(parents=True, exist_ok=True)
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")
        norm_stats_path = save_normalization_stats(envs, model_path)
        print(f"normalization stats saved to {norm_stats_path}")
        from cleanrl_utils.evals.ppo_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=Agent,
            device=device,
            capture_video=args.capture_video,
            gamma=args.gamma,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "PPO", os.path.join(args.run_dir, run_name), f"videos/{run_name}-eval")

    envs.close()
    writer.close()
