from __future__ import annotations

from typing import Callable, Dict, Iterable, Optional, Tuple, Union

import gymnasium as gym
import numpy as np

ParamSpec = Union[float, Tuple[str, float]]
RandomSpec = Union[Tuple[float, float], Tuple[str, float, float]]


class ObservationNoiseWrapper(gym.ObservationWrapper):
    """Add Gaussian noise to observations."""

    def __init__(
        self,
        env: gym.Env,
        noise_std: float = 0.0,
        noise_clip: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(env)
        self.noise_std = float(noise_std)
        self.noise_clip = noise_clip
        self.rng = np.random.default_rng(seed)

    def observation(self, observation):  # type: ignore[override]
        if self.noise_std <= 0:
            return observation
        noise = self.rng.normal(0.0, self.noise_std, size=np.asarray(observation).shape)
        if self.noise_clip is not None:
            noise = np.clip(noise, -self.noise_clip, self.noise_clip)
        return observation + noise


class RewardNoiseWrapper(gym.RewardWrapper):
    """Add Gaussian noise to rewards."""

    def __init__(self, env: gym.Env, noise_std: float = 0.0, seed: Optional[int] = None) -> None:
        super().__init__(env)
        self.noise_std = float(noise_std)
        self.rng = np.random.default_rng(seed)

    def reward(self, reward):  # type: ignore[override]
        if self.noise_std <= 0:
            return reward
        return reward + float(self.rng.normal(0.0, self.noise_std))


class ActionNoiseWrapper(gym.ActionWrapper):
    """Add Gaussian noise to continuous actions."""

    def __init__(
        self,
        env: gym.Env,
        noise_std: float = 0.0,
        noise_clip: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(env)
        self.noise_std = float(noise_std)
        self.noise_clip = noise_clip
        self.rng = np.random.default_rng(seed)
        if not isinstance(self.action_space, gym.spaces.Box):
            raise TypeError("ActionNoiseWrapper only supports Box action spaces.")

    def action(self, action):  # type: ignore[override]
        if self.noise_std <= 0:
            return action
        noise = self.rng.normal(0.0, self.noise_std, size=np.asarray(action).shape)
        if self.noise_clip is not None:
            noise = np.clip(noise, -self.noise_clip, self.noise_clip)
        noisy = action + noise
        return np.clip(noisy, self.action_space.low, self.action_space.high)


class ActionReplaceWrapper(gym.ActionWrapper):
    """With probability p, replace the action with a random action sample."""

    def __init__(self, env: gym.Env, replace_prob: float = 0.0, seed: Optional[int] = None) -> None:
        super().__init__(env)
        self.replace_prob = float(replace_prob)
        self.rng = np.random.default_rng(seed)
        if not isinstance(self.action_space, gym.spaces.Box):
            raise TypeError("ActionReplaceWrapper only supports Box action spaces.")
        if self.replace_prob < 0.0 or self.replace_prob > 1.0:
            raise ValueError("replace_prob must be in [0, 1].")
        self._low = np.asarray(self.action_space.low, dtype=np.float64)
        self._high = np.asarray(self.action_space.high, dtype=np.float64)
        self._finite_mask = np.isfinite(self._low) & np.isfinite(self._high)
        self._shape = self.action_space.shape

    def action(self, action):  # type: ignore[override]
        if self.replace_prob <= 0.0:
            return action
        if self.rng.random() < self.replace_prob:
            sample = np.empty(self._shape, dtype=np.float64)
            if np.any(self._finite_mask):
                u = self.rng.random(self._shape)
                sample[self._finite_mask] = (
                    self._low[self._finite_mask]
                    + u[self._finite_mask] * (self._high[self._finite_mask] - self._low[self._finite_mask])
                )
            if np.any(~self._finite_mask):
                sample[~self._finite_mask] = self.rng.normal(0.0, 1.0, size=int(np.sum(~self._finite_mask)))
            return sample.astype(self.action_space.dtype, copy=False)
        return action


class ParamOverrideWrapper(gym.Wrapper):
    """Deterministically override env attributes at each reset.

    Param spec:
      - value (float): sets attribute to value
      - (mode, value): mode in {"set","scale","add"}
    """

    def __init__(
        self,
        env: gym.Env,
        param_overrides: Dict[str, ParamSpec],
        strict: bool = True,
    ) -> None:
        super().__init__(env)
        self.param_overrides = dict(param_overrides)
        self.strict = strict
        self._base_values = self._capture_base_values(self.param_overrides.keys())

    def reset(self, *args, **kwargs):  # type: ignore[override]
        self._apply_overrides()
        return self.env.reset(*args, **kwargs)

    def _capture_base_values(self, keys: Iterable[str]) -> Dict[str, float]:
        base: Dict[str, float] = {}
        for name in keys:
            if hasattr(self.env.unwrapped, name):
                base[name] = float(getattr(self.env.unwrapped, name))
            elif self.strict:
                raise AttributeError(f"Env has no attribute '{name}' for override.")
        return base

    def _apply_overrides(self) -> None:
        for name, spec in self.param_overrides.items():
            if not hasattr(self.env.unwrapped, name):
                if self.strict:
                    raise AttributeError(f"Env has no attribute '{name}' for override.")
                continue
            mode = "set"
            value = spec
            if isinstance(spec, (tuple, list)):
                if len(spec) != 2:
                    raise ValueError(f"Param override for '{name}' must be (mode, value).")
                mode, value = spec[0], spec[1]
            base = self._base_values.get(name, float(getattr(self.env.unwrapped, name)))
            if mode == "set":
                new_val = float(value)
            elif mode == "scale":
                new_val = base * float(value)
            elif mode == "add":
                new_val = base + float(value)
            else:
                raise ValueError(f"Unknown override mode '{mode}' for '{name}'.")
            setattr(self.env.unwrapped, name, new_val)


class ParamRandomizationWrapper(gym.Wrapper):
    """Randomize env attributes at each reset.

    Random spec:
      - (low, high): sets attribute to uniform sample in [low, high]
      - (mode, low, high): mode in {"set","scale","add"}
    """

    def __init__(
        self,
        env: gym.Env,
        param_ranges: Dict[str, RandomSpec],
        seed: Optional[int] = None,
        strict: bool = True,
    ) -> None:
        super().__init__(env)
        self.param_ranges = dict(param_ranges)
        self.strict = strict
        self.rng = np.random.default_rng(seed)
        self._base_values = self._capture_base_values(self.param_ranges.keys())

    def reset(self, *args, **kwargs):  # type: ignore[override]
        self._apply_randomization()
        return self.env.reset(*args, **kwargs)

    def _capture_base_values(self, keys: Iterable[str]) -> Dict[str, float]:
        base: Dict[str, float] = {}
        for name in keys:
            if hasattr(self.env.unwrapped, name):
                base[name] = float(getattr(self.env.unwrapped, name))
            elif self.strict:
                raise AttributeError(f"Env has no attribute '{name}' for randomization.")
        return base

    def _apply_randomization(self) -> None:
        for name, spec in self.param_ranges.items():
            if not hasattr(self.env.unwrapped, name):
                if self.strict:
                    raise AttributeError(f"Env has no attribute '{name}' for randomization.")
                continue
            mode = "set"
            low_high = spec
            if isinstance(spec, (tuple, list)) and len(spec) == 3:
                mode, low, high = spec[0], spec[1], spec[2]
            elif isinstance(spec, (tuple, list)) and len(spec) == 2:
                low, high = spec[0], spec[1]
            else:
                raise ValueError(f"Param randomization for '{name}' must be (low, high) or (mode, low, high).")
            sampled = float(self.rng.uniform(low, high))
            base = self._base_values.get(name, float(getattr(self.env.unwrapped, name)))
            if mode == "set":
                new_val = sampled
            elif mode == "scale":
                new_val = base * sampled
            elif mode == "add":
                new_val = base + sampled
            else:
                raise ValueError(f"Unknown randomization mode '{mode}' for '{name}'.")
            setattr(self.env.unwrapped, name, new_val)


class ParamTransformWrapper(gym.Wrapper):
    """Apply an arbitrary parameter transform on each reset."""

    def __init__(self, env: gym.Env, transform: Callable[[gym.Env, np.random.Generator], None], seed: Optional[int] = None):
        super().__init__(env)
        self.transform = transform
        self.rng = np.random.default_rng(seed)

    def reset(self, *args, **kwargs):  # type: ignore[override]
        self.transform(self.env, self.rng)
        return self.env.reset(*args, **kwargs)


__all__ = [
    "ActionNoiseWrapper",
    "ActionReplaceWrapper",
    "ObservationNoiseWrapper",
    "ParamOverrideWrapper",
    "ParamRandomizationWrapper",
    "ParamTransformWrapper",
    "RewardNoiseWrapper",
]
