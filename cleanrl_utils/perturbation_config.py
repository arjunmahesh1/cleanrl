from __future__ import annotations

from typing import Dict

import gymnasium as gym

from cleanrl_utils.perturbation_wrappers import (
    ActionNoiseWrapper,
    ObservationNoiseWrapper,
    ParamOverrideWrapper,
    ParamRandomizationWrapper,
    RewardNoiseWrapper,
)


def _parse_kv_list(spec: str):
    if not spec:
        return []
    return [item.strip() for item in spec.split(",") if item.strip()]


def parse_override_spec(spec: str):
    overrides: Dict[str, object] = {}
    for item in _parse_kv_list(spec):
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}', expected key[[:mode]]=value")
        key, value_str = item.split("=", 1)
        if ":" in key:
            name, mode = key.split(":", 1)
        else:
            name, mode = key, "set"
        overrides[name] = (mode, float(value_str)) if mode != "set" else float(value_str)
    return overrides


def parse_randomization_spec(spec: str):
    randomization: Dict[str, object] = {}
    for item in _parse_kv_list(spec):
        if "=" not in item:
            raise ValueError(f"Invalid randomization '{item}', expected key[[:mode]]=low..high")
        key, range_str = item.split("=", 1)
        if ":" in key:
            name, mode = key.split(":", 1)
        else:
            name, mode = key, "set"
        if ".." not in range_str:
            raise ValueError(f"Invalid range '{range_str}', expected low..high")
        low_str, high_str = range_str.split("..", 1)
        low, high = float(low_str), float(high_str)
        if mode == "set":
            randomization[name] = (low, high)
        else:
            randomization[name] = (mode, low, high)
    return randomization


def apply_env_perturbations(
    env: gym.Env,
    *,
    obs_noise_std: float,
    obs_noise_clip: float | None,
    reward_noise_std: float,
    action_noise_std: float,
    action_noise_clip: float | None,
    param_override_spec: str,
    param_randomize_spec: str,
    param_strict: bool,
    seed: int,
):
    if param_override_spec:
        env = ParamOverrideWrapper(env, parse_override_spec(param_override_spec), strict=param_strict)
    if param_randomize_spec:
        env = ParamRandomizationWrapper(env, parse_randomization_spec(param_randomize_spec), seed=seed, strict=param_strict)
    if obs_noise_std > 0:
        env = ObservationNoiseWrapper(env, noise_std=obs_noise_std, noise_clip=obs_noise_clip, seed=seed)
    if reward_noise_std > 0:
        env = RewardNoiseWrapper(env, noise_std=reward_noise_std, seed=seed)
    if action_noise_std > 0:
        env = ActionNoiseWrapper(env, noise_std=action_noise_std, noise_clip=action_noise_clip, seed=seed)
    return env


__all__ = [
    "apply_env_perturbations",
    "parse_override_spec",
    "parse_randomization_spec",
]
