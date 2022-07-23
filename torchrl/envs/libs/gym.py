# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings
from types import ModuleType
from typing import List, Optional, Sequence, Dict

import torch
from packaging import version

from torchrl.data import (
    BinaryDiscreteTensorSpec,
    CompositeSpec,
    MultOneHotDiscreteTensorSpec,
    NdBoundedTensorSpec,
    OneHotDiscreteTensorSpec,
    TensorSpec,
    UnboundedContinuousTensorSpec,
)
from ...data.utils import numpy_to_torch_dtype_dict
from ..gym_like import GymLikeEnv, default_info_dict_reader
from ..utils import classproperty

try:
    import gym

    _has_gym = True

except ImportError:
    _has_gym = False


if _has_gym:
    try:
        from gym.wrappers.pixel_observation import PixelObservationWrapper
    except ModuleNotFoundError:
        warnings.warn(
            f"gym {gym.__version__} does not provide the PixelObservationWrapper"
            f"used by torchrl, which will be using a patched version. "
            f"Consider updating gym to a newer version."
        )
        from torchrl.envs.libs.utils import (
            GymPixelObservationWrapper as PixelObservationWrapper,
        )

try:
    import retro

    _has_retro = True
except ImportError:
    _has_retro = False

__all__ = ["GymWrapper", "GymEnv", "RetroEnv"]


def _gym_to_torchrl_spec_transform(spec, dtype=None, device="cpu") -> TensorSpec:
    if isinstance(spec, gym.spaces.tuple.Tuple):
        raise NotImplementedError("gym.spaces.tuple.Tuple mapping not yet implemented")
    if isinstance(spec, gym.spaces.discrete.Discrete):
        return OneHotDiscreteTensorSpec(spec.n, device=device)
    elif isinstance(spec, gym.spaces.multi_binary.MultiBinary):
        return BinaryDiscreteTensorSpec(spec.n, device=device)
    elif isinstance(spec, gym.spaces.multi_discrete.MultiDiscrete):
        return MultOneHotDiscreteTensorSpec(spec.nvec, device=device)
    elif isinstance(spec, gym.spaces.Box):
        if dtype is None:
            dtype = numpy_to_torch_dtype_dict[spec.dtype]
        return NdBoundedTensorSpec(
            torch.tensor(spec.low, device=device, dtype=dtype),
            torch.tensor(spec.high, device=device, dtype=dtype),
            torch.Size(spec.shape),
            dtype=dtype,
            device=device,
        )
    elif isinstance(spec, (dict, gym.spaces.dict.Dict)):
        spec = {
            "next_" + k: _gym_to_torchrl_spec_transform(spec[k], device=device)
            for k in spec
        }
        return CompositeSpec(**spec)
    else:
        raise NotImplementedError(
            f"spec of type {type(spec).__name__} is currently unaccounted for"
        )


def _get_envs(to_dict=False) -> List:
    envs = gym.envs.registration.registry.env_specs.keys()
    envs = list(envs)
    envs = sorted(envs)
    return envs


def _get_gym():
    if _has_gym:
        return gym
    else:
        return None


def _is_from_pixels(env):
    observation_spec = env.observation_space
    if isinstance(observation_spec, (Dict, gym.spaces.dict.Dict)):
        if "pixels" in set(observation_spec.keys()):
            return True
    elif (
        isinstance(observation_spec, gym.spaces.Box)
        and (observation_spec.low == 0).all()
        and (observation_spec.high == 255).all()
        and observation_spec.low.shape[-1] == 3
        and observation_spec.low.ndim == 3
    ):
        return True
    elif isinstance(env, PixelObservationWrapper):
        return True
    return False


class GymWrapper(GymLikeEnv):
    """
    OpenAI Gym environment wrapper.

    Examples:
        >>> env = gym.make("Pendulum-v0")
        >>> env = GymWrapper(env)
        >>> td = env.rand_step()
        >>> print(td)
        >>> print(env.available_envs)
    """

    git_url = "https://github.com/openai/gym"
    libname = "gym"

    def __init__(self, env=None, **kwargs):
        if env is not None:
            kwargs["env"] = env
        self._seed_calls_reset = None
        super().__init__(**kwargs)

    def _check_kwargs(self, kwargs: Dict):
        if "env" not in kwargs:
            raise TypeError("Could not find environment key 'env' in kwargs.")
        env = kwargs["env"]
        if not (hasattr(env, "action_space") and hasattr(env, "observation_space")):
            raise TypeError("env is not of type 'gym.Env'.")

    def _build_env(
        self,
        env,
        from_pixels: bool = False,
        pixels_only: bool = False,
    ) -> "gym.core.Env":
        env_from_pixels = _is_from_pixels(env)
        from_pixels = from_pixels or env_from_pixels
        self.from_pixels = from_pixels
        self.pixels_only = pixels_only
        if from_pixels and not env_from_pixels:
            if isinstance(env, PixelObservationWrapper):
                raise TypeError(
                    "PixelObservationWrapper cannot be used to wrap an environment"
                    "that is already a PixelObservationWrapper instance."
                )
            env = PixelObservationWrapper(env, pixels_only=pixels_only)
        return env

    @classproperty
    def available_envs(cls) -> List[str]:
        return _get_envs()

    @property
    def lib(self) -> ModuleType:
        return gym

    def _set_seed(self, seed: int) -> int:
        skip = False
        if self._seed_calls_reset is None:
            if version.parse(gym.__version__) < version.parse("0.19.0"):
                self._seed_calls_reset = False
                self._env.seed(seed=seed)
            else:
                try:
                    self.reset(seed=seed)
                    skip = True
                    self._seed_calls_reset = True
                except TypeError as err:
                    warnings.warn(
                        f"reset with seed kwarg returned an exception: {err}.\n"
                        f"Calling env.seed from now on."
                    )
                    self._seed_calls_reset = False
        if self._seed_calls_reset and not skip:
            self.reset(seed=seed)
        elif not self._seed_calls_reset:
            self._env.seed(seed=seed)
        return seed

    def _make_specs(self, env: "gym.Env") -> None:
        self.action_spec = _gym_to_torchrl_spec_transform(
            env.action_space, device=self.device
        )
        self.observation_spec = _gym_to_torchrl_spec_transform(
            env.observation_space, device=self.device
        )
        if not isinstance(self.observation_spec, CompositeSpec):
            if self.from_pixels:
                self.observation_spec = CompositeSpec(next_pixels=self.observation_spec)
            else:
                self.observation_spec = CompositeSpec(
                    next_observation=self.observation_spec
                )
        self.reward_spec = UnboundedContinuousTensorSpec(
            device=self.device,
        )

    def _init_env(self):
        self.reset()  # make sure that _is_done is populated

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(env={self._env}, batch_size={self.batch_size})"
        )

    def rebuild_with_kwargs(self, **new_kwargs):
        self._constructor_kwargs.update(new_kwargs)
        self._env = self._build_env(**self._constructor_kwargs)
        self._make_specs(self._env)

    @property
    def info_dict_reader(self):
        if self._info_dict_reader is None:
            self._info_dict_reader = default_info_dict_reader()
        return self._info_dict_reader

    @info_dict_reader.setter
    def info_dict_reader(self, value: callable):
        self._info_dict_reader = value


class GymEnv(GymWrapper):
    """
    OpenAI Gym environment wrapper.

    Examples:
        >>> env = GymEnv(env_name="Pendulum-v0", frame_skip=4)
        >>> td = env.rand_step()
        >>> print(td)
        >>> print(env.available_envs)

    """

    def __init__(self, env_name, disable_env_checker=True, **kwargs):
        kwargs["env_name"] = env_name
        kwargs["disable_env_checker"] = disable_env_checker
        super().__init__(**kwargs)

    def _build_env(
        self,
        env_name: str,
        **kwargs,
    ) -> "gym.core.Env":
        if not _has_gym:
            raise RuntimeError(
                f"gym not found, unable to create {env_name}. "
                f"Consider downloading and installing dm_control from"
                f" {self.git_url}"
            )
        from_pixels = kwargs.get("from_pixels", False)
        if from_pixels:
            kwargs["render_mode"] = "single_rgb_array"
        if "from_pixels" in kwargs:
            del kwargs["from_pixels"]

        pixels_only = kwargs.get("pixels_only", True)
        if "pixels_only" in kwargs:
            del kwargs["pixels_only"]
        try:
            with warnings.catch_warnings(record=True) as w:
                env = self.lib.make(env_name, frameskip=self.frame_skip, **kwargs)
                if any("frameskip" in str(_w.message) for _w in w):
                    raise TypeError("unexpected keyword argument 'frameskip'")
            self.wrapper_frame_skip = 1
        except TypeError as err:
            if "unexpected keyword argument 'frameskip'" not in str(err):
                raise TypeError(err)
            env = self.lib.make(env_name, **kwargs)
            self.wrapper_frame_skip = self.frame_skip
        return super()._build_env(env, pixels_only=pixels_only, from_pixels=from_pixels)

    @property
    def env_name(self):
        return self._constructor_kwargs["env_name"]

    def _check_kwargs(self, kwargs: Dict):
        if "env_name" not in kwargs:
            raise TypeError("Expected 'env_name' to be part of kwargs")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(env={self.env_name}, batch_size={self.batch_size}, device={self.device})"


def _get_retro_envs() -> Sequence:
    if not _has_retro:
        return tuple()
    else:
        return retro.data.list_games()


def _get_retro() -> Optional[ModuleType]:
    if _has_retro:
        return retro
    else:
        return None


class RetroEnv(GymEnv):
    available_envs = _get_retro_envs()
    lib = "retro"
    lib = _get_retro()
