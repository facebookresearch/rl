# Copyright (c) Meta Plobs_dictnc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from copy import deepcopy
from typing import Any, List, Optional, OrderedDict, Sequence, Union
from warnings import warn

import torch
from torch import nn

try:
    _has_tv = True
    from torchvision.transforms.functional_tensor import (
        resize,
    )  # as of now resize is imported from torchvision
except ImportError:
    _has_tv = False

from torchrl.data.tensor_specs import (
    BoundedTensorSpec,
    CompositeSpec,
    ContinuousBox,
    NdUnboundedContinuousTensorSpec,
    TensorSpec,
    UnboundedContinuousTensorSpec,
    BinaryDiscreteTensorSpec,
)
from torchrl.data.tensordict.tensordict import _TensorDict, TensorDict
from torchrl.envs.common import _EnvClass, make_tensordict
from torchrl.envs.transforms import functional as F
from torchrl.envs.transforms.utils import FiniteTensor
from torchrl.envs.utils import step_tensordict

__all__ = [
    "Transform",
    "TransformedEnv",
    "RewardClipping",
    "Resize",
    "GrayScale",
    "Compose",
    "ToTensorImage",
    "ObservationNorm",
    "RewardScaling",
    "ObservationTransform",
    "CatFrames",
    "FiniteTensorDictCheck",
    "DoubleToFloat",
    "CatTensors",
    "NoopResetEnv",
    "BinarizeReward",
    "PinMemoryTransform",
    "MultiAgentTransform",
    "VecNorm",
    "gSDENoise",
]

IMAGE_KEYS = ["next_pixels"]
_MAX_NOOPS_TRIALS = 10


class Transform(nn.Module):
    """Environment transform parent class.

    In principle, a transform receives a tensordict as input and returns (
    the same or another) tensordict as output, where a series of values have
    been modified or created with a new key. When instantiating a new
    transform, the keys that are to be read from are passed to the
    constructor via the `keys` argument.

    Transforms are to be combined with their target environments with the
    TransformedEnv class, which takes as arguments an `_EnvClass` instance
    and a transform. If multiple transforms are to be used, they can be
    concatenated using the `Compose` class.
    A transform can be stateless or stateful (e.g. CatTransform). Because of
    this, Transforms support the `reset` operation, which should reset the
    transform to its initial state (such that successive trajectories are kept
    independent).

    Notably, `Transform` subclasses take care of transforming the affected
    specs from an environment: when querying
    `transformed_env.observation_spec`, the resulting objects will describe
    the specs of the transformed tensors.

    """

    invertible = False

    def __init__(self, keys: Sequence[str]):
        super().__init__()
        self.keys = keys

    def reset(self, tensordict: _TensorDict) -> _TensorDict:
        """Resets a tranform if it is stateful."""
        return tensordict

    def _check_inplace(self) -> None:
        if not hasattr(self, "inplace"):
            raise AttributeError(
                f"Transform of class {self.__class__.__name__} has no "
                f"attribute inplace, consider implementing it."
            )

    def init(self, tensordict) -> None:
        pass

    def _apply(self, obs: torch.Tensor) -> None:
        """Applies the transform to a tensor.
        This operation can be called multiple times (if multiples keys of the
        tensordict match the keys of the transform).

        """
        raise NotImplementedError

    def _call(self, tensordict: _TensorDict) -> _TensorDict:
        """Reads the input tensordict, and for the selected keys, applies the
        transform.

        """
        self._check_inplace()
        for _obs_key in tensordict.keys():
            if _obs_key in self.keys:
                observation = self._apply(tensordict.get(_obs_key))
                tensordict.set(_obs_key, observation, inplace=self.inplace)
        return tensordict

    def forward(self, tensordict: _TensorDict) -> _TensorDict:
        tensordict = self._call(tensordict)
        return tensordict

    def _inv_apply(self, obs: torch.Tensor) -> torch.Tensor:
        if self.invertible:
            raise NotImplementedError
        else:
            return obs

    def _inv_call(self, tensordict: _TensorDict) -> _TensorDict:
        self._check_inplace()
        for _obs_key in tensordict.keys():
            if _obs_key in self.keys:
                observation = self._inv_apply(tensordict.get(_obs_key))
                tensordict.set(_obs_key, observation, inplace=self.inplace)
        return tensordict

    def inv(self, tensordict: _TensorDict) -> _TensorDict:
        self._inv_call(tensordict)
        return tensordict

    def transform_batch_size(self, _batch_size: torch.Size) -> torch.Size:
        return _batch_size

    def transform_action_spec(self, action_spec: TensorSpec) -> TensorSpec:
        """Transforms the action spec such that the resulting spec matches
        transform mapping.

        Args:
            action_spec (TensorSpec): spec before the transform

        Returns:
            expected spec after the transform

        """
        return action_spec

    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        """Transforms the observation spec such that the resulting spec
        matches transform mapping.

        Args:
            observation_spec (TensorSpec): spec before the transform

        Returns:
            expected spec after the transform

        """
        return observation_spec

    def transform_reward_spec(self, reward_spec: TensorSpec) -> TensorSpec:
        """Transforms the reward spec such that the resulting spec matches
        transform mapping.

        Args:
            reward_spec (TensorSpec): spec before the transform

        Returns:
            expected spec after the transform

        """

        return reward_spec

    def dump(self, **kwargs) -> None:
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(keys={self.keys})"

    def set_parent(self, parent: Union[Transform, _EnvClass]) -> None:
        self.__dict__["_parent"] = parent

    @property
    def parent(self) -> _EnvClass:
        if not hasattr(self, "_parent"):
            raise AttributeError("transform parent uninitialized")
        parent = self._parent
        while not isinstance(parent, _EnvClass):
            if not isinstance(parent, Transform):
                raise ValueError(
                    "A transform parent must be either another transform or an environment object."
                )
            parent = parent.parent
        return parent

    def empty_cache(self):
        if self.parent is not None:
            self.parent.empty_cache()


class TransformedEnv(_EnvClass):
    """
    A transformed environment.

    Args:
        env (_EnvClass): original environment to be transformed.
        transform (Transform, optional): transform to apply to the tensordict resulting
            from `env.step(td)`. If none is provided, an empty Compose placeholder is
            used.
        cache_specs (bool, optional): if True, the specs will be cached once
            and for all after the first call (i.e. the specs will be
            transformed only once). If the transform changes during
            training, the original spec transform may not be valid anymore,
            in which case this value should be set  to `False`. Default is
            `True`.

    Examples:
        >>> env = GymEnv("Pendulum-v0")
        >>> transform = RewardScaling(0.0, 1.0)
        >>> transformed_env = TransformedEnv(env, transform)

    """

    def __init__(
        self,
        env: _EnvClass,
        transform: Optional[Transform] = None,
        cache_specs: bool = True,
        **kwargs,
    ):
        self.env = env
        if transform is None:
            transform = Compose()
        self.transform = transform
        transform.set_parent(self)  # allows to find env specs from the transform

        self._last_obs = None
        self.cache_specs = cache_specs

        self._action_spec = None
        self._reward_spec = None
        self._observation_spec = None
        self._batch_size = self.env.batch_size
        self.is_closed = False

        super().__init__(**kwargs)

    @property
    def batch_size(self):
        batch_size = self.env.batch_size
        batch_size = self.transform.transform_batch_size(batch_size)
        return batch_size

    @property
    def observation_spec(self) -> TensorSpec:
        """Observation spec of the transformed environment"""
        if self._observation_spec is None or not self.cache_specs:
            observation_spec = self.transform.transform_observation_spec(
                deepcopy(self.env.observation_spec)
            )
            if self.cache_specs:
                self._observation_spec = observation_spec
        else:
            observation_spec = self._observation_spec
        return observation_spec

    @property
    def action_spec(self) -> TensorSpec:
        """Action spec of the transformed environment"""

        if self._action_spec is None or not self.cache_specs:
            action_spec = self.transform.transform_action_spec(
                deepcopy(self.env.action_spec)
            )
            if self.cache_specs:
                self._action_spec = action_spec
        else:
            action_spec = self._action_spec
        return action_spec

    @property
    def reward_spec(self) -> TensorSpec:
        """Reward spec of the transformed environment"""

        if self._reward_spec is None or not self.cache_specs:
            reward_spec = self.transform.transform_reward_spec(
                deepcopy(self.env.reward_spec)
            )
            if self.cache_specs:
                self._reward_spec = reward_spec
        else:
            reward_spec = self._reward_spec
        return reward_spec

    def _step(self, tensordict: _TensorDict) -> _TensorDict:
        selected_keys = [key for key in tensordict.keys() if "action" in key]
        tensordict_in = tensordict.select(*selected_keys).clone()
        tensordict_in = self.transform.inv(tensordict_in)
        tensordict_out = self.env._step(tensordict_in).to(self.device)
        # tensordict should already have been processed by the transforms
        # for logging purposes
        tensordict_out = self.transform(tensordict_out)
        return tensordict_out

    def set_seed(self, seed: int) -> int:
        """Set the seeds of the environment"""
        return self.env.set_seed(seed)

    def _reset(self, tensordict: Optional[_TensorDict] = None, **kwargs):
        out_tensordict = self.env.reset(execute_step=False, **kwargs).to(self.device)
        out_tensordict = self.transform.reset(out_tensordict)
        out_tensordict = self.transform(out_tensordict)
        return out_tensordict

    def state_dict(self) -> OrderedDict:
        state_dict = self.transform.state_dict()
        return state_dict

    def load_state_dict(self, state_dict: OrderedDict, **kwargs) -> None:
        self.transform.load_state_dict(state_dict, **kwargs)

    def eval(self) -> TransformedEnv:
        self.transform.eval()
        return self

    def train(self, mode: bool = True) -> TransformedEnv:
        self.transform.train(mode)
        return self

    def close(self):
        self.is_closed = True
        self.env.close()

    def empty_cache(self):
        self._observation_spec = None
        self._action_spec = None
        self._reward_spec = None

    def append_transform(self, transform: Transform) -> None:
        if not isinstance(transform, Transform):
            raise ValueError(
                "TransformedEnv.append_transform expected a transform but received an object of "
                f"type {type(transform)} instead."
            )
        if not isinstance(self.transform, Compose):
            self.transform = Compose(self.transform)
            self.transform.set_parent(self)
        self.transform.append(transform)

    def __getattr__(self, attr: str) -> Any:
        if attr in self.__dir__():
            return self.__getattribute__(
                attr
            )  # make sure that appropriate exceptions are raised
        elif attr.startswith("__"):
            raise AttributeError(
                "passing built-in private methods is "
                f"not permitted with type {type(self)}. "
                f"Got attribute {attr}."
            )
        elif "env" in self.__dir__():
            env = self.__getattribute__("env")
            return getattr(env, attr)

        raise AttributeError(
            f"env not set in {self.__class__.__name__}, cannot access {attr}"
        )

    def __repr__(self) -> str:
        return f"TransformedEnv(env={self.env}, transform={self.transform})"


class ObservationTransform(Transform):
    """
    Abstract class for transformations of the observations.

    """

    inplace = False

    def __init__(self, keys: Optional[Sequence[str]] = None):
        if keys is None:
            keys = [
                "next_observation",
                "next_pixels",
                "next_observation_state",
            ]
        super(ObservationTransform, self).__init__(keys=keys)


class Compose(Transform):
    """
    Composes a chain of transforms.

    Examples:
        >>> env = GymEnv("Pendulum-v0")
        >>> transforms = [RewardScaling(1.0, 1.0), RewardClipping(-2.0, 2.0)]
        >>> transforms = Compose(*transforms)
        >>> transformed_env = TransformedEnv(env, transforms)

    """

    inplace = False

    def __init__(self, *transforms: Transform):
        super().__init__(keys=[])
        self.transforms = nn.ModuleList(transforms)
        for t in self.transforms:
            t.set_parent(self)

    def _call(self, tensordict: _TensorDict) -> _TensorDict:
        for t in self.transforms:
            tensordict = t(tensordict)
        return tensordict

    def transform_action_spec(self, action_spec: TensorSpec) -> TensorSpec:
        for t in self.transforms:
            action_spec = t.transform_action_spec(action_spec)
        return action_spec

    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        for t in self.transforms:
            observation_spec = t.transform_observation_spec(observation_spec)
        return observation_spec

    def transform_reward_spec(self, reward_spec: TensorSpec) -> TensorSpec:
        for t in self.transforms:
            reward_spec = t.transform_reward_spec(reward_spec)
        return reward_spec

    def __getitem__(self, item: Union[int, slice, List]) -> Union:
        transform = self.transforms
        transform = transform[item]
        if not isinstance(transform, Transform):
            return Compose(*self.transforms[item])
        return transform

    def dump(self, **kwargs) -> None:
        for t in self:
            t.dump(**kwargs)

    def reset(self, tensordict: _TensorDict) -> _TensorDict:
        for t in self.transforms:
            tensordict = t.reset(tensordict)
        return tensordict

    def init(self, tensordict: _TensorDict) -> None:
        for t in self.transforms:
            t.init(tensordict)

    def append(self, transform):
        self.empty_cache()
        if not isinstance(transform, Transform):
            raise ValueError(
                "Compose.append expected a transform but received an object of "
                f"type {type(transform)} instead."
            )
        self.transforms.append(transform)
        transform.set_parent(self)

    def __len__(self):
        return len(self.transforms)

    def __repr__(self) -> str:
        layers_str = ", \n\t".join([str(trsf) for trsf in self.transforms])
        return f"{self.__class__.__name__}(\n\t{layers_str})"


class ToTensorImage(ObservationTransform):
    """Transforms a numpy-like image (3 x W x H) to a pytorch image
    (3 x W x H).

    Transforms an observation image from a (... x W x H x 3) 0..255 uint8
    tensor to a single/double precision floating point (3 x W x H) tensor
    with values between 0 and 1.

    Args:
        unsqueeze (bool): if True, the observation tensor is unsqueezed
            along the first dimension. default=False.
        dtype (torch.dtype, optional): dtype to use for the resulting
            observations.

    Examples:
        >>> transform = ToTensorImage(keys=["next_pixels"])
        >>> ri = torch.randint(0, 255, (1,1,10,11,3), dtype=torch.uint8)
        >>> td = TensorDict(
        ...     {"next_pixels": ri},
        ...     [1, 1])
        >>> _ = transform(td)
        >>> obs = td.get("next_pixels")
        >>> print(obs.shape, obs.dtype)
        torch.Size([1, 1, 3, 10, 11]) torch.float32
    """

    inplace = False

    def __init__(
        self,
        unsqueeze: bool = False,
        dtype: Optional[torch.device] = None,
        keys: Optional[Sequence[str]] = None,
    ):
        if keys is None:
            keys = IMAGE_KEYS  # default
        super().__init__(keys=keys)
        self.unsqueeze = unsqueeze
        self.dtype = dtype if dtype is not None else torch.get_default_dtype()

    def _apply(self, observation: torch.FloatTensor) -> torch.Tensor:
        observation = observation.div(255).to(self.dtype)
        observation = observation.permute(
            *list(range(observation.ndimension() - 3)), -1, -3, -2
        )
        if observation.ndimension() == 3 and self.unsqueeze:
            observation = observation.unsqueeze(0)
        return observation

    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        if isinstance(observation_spec, CompositeSpec):
            return CompositeSpec(
                **{
                    key: self.transform_observation_spec(_obs_spec)
                    if key in self.keys
                    else _obs_spec
                    for key, _obs_spec in observation_spec._specs.items()
                }
            )
        else:
            _observation_spec = observation_spec
        self._pixel_observation(_observation_spec)
        _observation_spec.shape = torch.Size(
            [
                *_observation_spec.shape[:-3],
                _observation_spec.shape[-1],
                _observation_spec.shape[-3],
                _observation_spec.shape[-2],
            ]
        )
        _observation_spec.dtype = self.dtype
        observation_spec = _observation_spec
        return observation_spec

    def _pixel_observation(self, spec: TensorSpec) -> None:
        if isinstance(spec, BoundedTensorSpec):
            spec.space.maximum = self._apply(spec.space.maximum)
            spec.space.minimum = self._apply(spec.space.minimum)


class RewardClipping(Transform):
    """
    Clips the reward between `clamp_min` and `clamp_max`.

    Args:
        clip_min (scalar): minimum value of the resulting reward.
        clip_max (scalar): maximum value of the resulting reward.

    """

    inplace = True

    def __init__(
        self,
        clamp_min: float = None,
        clamp_max: float = None,
        keys: Optional[Sequence[str]] = None,
    ):
        if keys is None:
            keys = ["reward"]
        super().__init__(keys=keys)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def _apply(self, reward: torch.Tensor) -> torch.Tensor:
        if self.clamp_max is not None and self.clamp_min is not None:
            reward = reward.clamp_(self.clamp_min, self.clamp_max)
        elif self.clamp_min is not None:
            reward = reward.clamp_min_(self.clamp_min)
        elif self.clamp_max is not None:
            reward = reward.clamp_max_(self.clamp_max)
        return reward

    def transform_reward_spec(self, reward_spec: TensorSpec) -> TensorSpec:
        if isinstance(reward_spec, UnboundedContinuousTensorSpec):
            return BoundedTensorSpec(
                self.clamp_min,
                self.clamp_max,
                device=reward_spec.device,
                dtype=reward_spec.dtype,
            )
        else:
            raise NotImplementedError(
                f"{self.__class__.__name__}.transform_reward_spec not "
                f"implemented for tensor spec of type"
                f" {type(reward_spec).__name__}"
            )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"clamp_min={float(self.clamp_min):4.4f}, clamp_max"
            f"={float(self.clamp_max):4.4f}, keys={self.keys})"
        )


class BinarizeReward(Transform):
    """
    Maps the reward to a binary value (0 or 1) if the reward is null or
    non-null, respectively.

    """

    inplace = True

    def __init__(self, keys: Optional[Sequence[str]] = None):
        if keys is None:
            keys = ["reward"]
        super().__init__(keys=keys)

    def _apply(self, reward: torch.Tensor) -> torch.Tensor:
        if not reward.shape or reward.shape[-1] != 1:
            raise RuntimeError(
                f"Reward shape last dimension must be singleton, got reward of shape {reward.shape}"
            )
        return (reward > 0.0).to(torch.long)

    def transform_reward_spec(self, reward_spec: TensorSpec) -> TensorSpec:
        return BinaryDiscreteTensorSpec(n=1, device=reward_spec.device)


class Resize(ObservationTransform):
    """
    Resizes an pixel observation.

    Args:
        w (int): resulting width
        h (int): resulting height
        interpolation (str): interpolation method
    """

    inplace = False

    def __init__(
        self,
        w: int,
        h: int,
        interpolation: str = "bilinear",
        keys: Optional[Sequence[str]] = None,
    ):
        if not _has_tv:
            raise ImportError(
                "Torchvision not found. The Resize transform relies on "
                "torchvision implementation. "
                "Consider installing this dependency."
            )
        if keys is None:
            keys = IMAGE_KEYS  # default
        super().__init__(keys=keys)
        self.w = w
        self.h = h
        self.interpolation = interpolation

    def _apply(self, observation: torch.Tensor) -> torch.Tensor:
        observation = resize(
            observation, [self.w, self.h], interpolation=self.interpolation
        )

        return observation

    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        if isinstance(observation_spec, CompositeSpec):
            return CompositeSpec(
                **{
                    key: self.transform_observation_spec(_obs_spec)
                    if key in self.keys
                    else _obs_spec
                    for key, _obs_spec in observation_spec._specs.items()
                }
            )
        else:
            _observation_spec = observation_spec
        space = _observation_spec.space
        if isinstance(space, ContinuousBox):
            space.minimum = self._apply(space.minimum)
            space.maximum = self._apply(space.maximum)
            _observation_spec.shape = space.minimum.shape
        else:
            _observation_spec.shape = self._apply(
                torch.zeros(_observation_spec.shape)
            ).shape

        observation_spec = _observation_spec
        return observation_spec

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"w={float(self.w):4.4f}, h={float(self.h):4.4f}, "
            f"interpolation={self.interpolation}, keys={self.keys})"
        )


class GrayScale(ObservationTransform):
    """
    Turns a pixel observation to grayscale.

    """

    inplace = False

    def __init__(self, keys: Optional[Sequence[str]] = None):
        if keys is None:
            keys = IMAGE_KEYS
        super(GrayScale, self).__init__(keys=keys)

    def _apply(self, observation: torch.Tensor) -> torch.Tensor:
        observation = F.rgb_to_grayscale(observation)
        return observation

    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        if isinstance(observation_spec, CompositeSpec):
            return CompositeSpec(
                **{
                    key: self.transform_observation_spec(_obs_spec)
                    if key in self.keys
                    else _obs_spec
                    for key, _obs_spec in observation_spec._specs.items()
                }
            )
        else:
            _observation_spec = observation_spec
        space = _observation_spec.space
        if isinstance(space, ContinuousBox):
            space.minimum = self._apply(space.minimum)
            space.maximum = self._apply(space.maximum)
            _observation_spec.shape = space.minimum.shape
        else:
            _observation_spec.shape = self._apply(
                torch.zeros(_observation_spec.shape)
            ).shape
        observation_spec = _observation_spec
        return observation_spec


class ObservationNorm(ObservationTransform):
    """
    Normalizes an observation according to

    .. math::
        obs = obs * scale + loc

    Args:
        loc (number or tensor): location of the affine transform
        scale (number or tensor): scale of the affine transform
        standard_normal (bool, optional): if True, the transform will be

            .. math::
                obs = (obs-loc)/scale

            as it is done for standardization. Default is `False`.

    Examples:
        >>> torch.set_default_tensor_type(torch.DoubleTensor)
        >>> r = torch.randn(100, 3)*torch.randn(3) + torch.randn(3)
        >>> td = TensorDict({'next_obs': r}, [100])
        >>> transform = ObservationNorm(
        ...     loc = td.get('next_obs').mean(0),
        ...     scale = td.get('next_obs').std(0),
        ...     keys=["next_obs"],
        ...     standard_normal=True)
        >>> _ = transform(td)
        >>> print(torch.isclose(td.get('next_obs').mean(0),
        ...     torch.zeros(3)).all())
        Tensor(True)
        >>> print(torch.isclose(td.get('next_obs').std(0),
        ...     torch.ones(3)).all())
        Tensor(True)

    """

    inplace = True

    def __init__(
        self,
        loc: Union[float, torch.Tensor],
        scale: Union[float, torch.Tensor],
        keys: Optional[Sequence[str]] = None,
        # observation_spec_key: =None,
        standard_normal: bool = False,
    ):
        if keys is None:
            keys = [
                "next_observation",
                "next_pixels",
                "next_observation_state",
            ]
        super().__init__(keys=keys)
        if not isinstance(loc, torch.Tensor):
            loc = torch.tensor(loc, dtype=torch.float)
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale, dtype=torch.float)

        # self.observation_spec_key = observation_spec_key
        self.standard_normal = standard_normal
        self.register_buffer("loc", loc)
        eps = 1e-6
        self.register_buffer("scale", scale.clamp_min(eps))

    def _apply(self, obs: torch.Tensor) -> torch.Tensor:
        if self.standard_normal:
            # converts the transform (x-m)/sqrt(v) to x * s + loc
            scale = self.scale.reciprocal()
            loc = -self.loc * scale
        else:
            scale = self.scale
            loc = self.loc
        return obs * scale + loc

    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        if isinstance(observation_spec, CompositeSpec):
            keys = [key for key in observation_spec.keys() if key in self.keys]
            for key in keys:
                observation_spec[key] = self.transform_observation_spec(
                    observation_spec[key]
                )
            return observation_spec
        space = observation_spec.space
        if isinstance(space, ContinuousBox):
            space.minimum = self._apply(space.minimum)
            space.maximum = self._apply(space.maximum)
        return observation_spec

    def __repr__(self) -> str:
        if self.loc.numel() == 1 and self.scale.numel() == 1:
            return (
                f"{self.__class__.__name__}("
                f"loc={float(self.loc):4.4f}, scale"
                f"={float(self.scale):4.4f}, keys={self.keys})"
            )
        else:
            return super().__repr__()


class CatFrames(ObservationTransform):
    """Concatenates successive observation frames into a single tensor.

    This can, for instance, account for movement/velocity of the observed
    feature. Proposed in "Playing Atari with Deep Reinforcement Learning" (
    https://arxiv.org/pdf/1312.5602.pdf).

    CatFrames is a stateful class and it can be reset to its native state by
    calling the `reset()` method.

    Args:
        N (int, optional): number of observation to concatenate.
            Default is `4`.
        cat_dim (int, optional): dimension along which concatenate the
            observations. Default is `cat_dim=-3`.
        keys (list of int, optional): keys pointing to the frames that have
            to be concatenated.

    """

    inplace = False

    def __init__(
        self,
        N: int = 4,
        cat_dim: int = -3,
        keys: Optional[Sequence[str]] = None,
    ):
        if keys is None:
            keys = IMAGE_KEYS
        super().__init__(keys=keys)
        self.N = N
        self.cat_dim = cat_dim
        self.buffer = []

    def reset(self, tensordict: _TensorDict) -> _TensorDict:
        self.buffer = []
        return tensordict

    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        if isinstance(observation_spec, CompositeSpec):
            keys = [key for key in observation_spec.keys() if key in self.keys]
            for key in keys:
                observation_spec[key] = self.transform_observation_spec(
                    observation_spec[key]
                )
            return observation_spec
        else:
            _observation_spec = observation_spec
        space = _observation_spec.space
        if isinstance(space, ContinuousBox):
            space.minimum = torch.cat([space.minimum] * self.N, self.cat_dim)
            space.maximum = torch.cat([space.maximum] * self.N, self.cat_dim)
            _observation_spec.shape = space.minimum.shape
        else:
            shape = list(_observation_spec.shape)
            shape[self.cat_dim] = self.N * shape[self.cat_dim]
            _observation_spec.shape = torch.Size(shape)
        observation_spec = _observation_spec
        return observation_spec

    def _apply(self, obs: torch.Tensor) -> torch.Tensor:
        self.buffer.append(obs)
        self.buffer = self.buffer[-self.N :]
        buffer = list(reversed(self.buffer))
        buffer = [buffer[0]] * (self.N - len(buffer)) + buffer
        if len(buffer) != self.N:
            raise RuntimeError(
                f"actual buffer length ({buffer}) differs from expected (" f"{self.N})"
            )
        return torch.cat(buffer, self.cat_dim)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(N={self.N}, cat_dim"
            f"={self.cat_dim}, keys={self.keys})"
        )


class RewardScaling(Transform):
    """
    Affine transform of the reward according to

    .. math::
        reward = reward * scale + loc

    Args:
        loc (number or torch.Tensor): location of the affine transform
        scale (number or torch.Tensor): scale of the affine transform
    """

    inplace = True

    def __init__(
        self,
        loc: Union[float, torch.Tensor],
        scale: Union[float, torch.Tensor],
        keys: Optional[Sequence[str]] = None,
    ):
        if keys is None:
            keys = ["reward"]
        super().__init__(keys=keys)
        if not isinstance(loc, torch.Tensor):
            loc = torch.tensor(loc)
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale)

        self.register_buffer("loc", loc)
        self.register_buffer("scale", scale.clamp_min(1e-6))

    def _apply(self, reward: torch.Tensor) -> torch.Tensor:
        reward.mul_(self.scale).add_(self.loc)
        return reward

    def transform_reward_spec(self, reward_spec: TensorSpec) -> TensorSpec:
        if isinstance(reward_spec, UnboundedContinuousTensorSpec):
            return reward_spec
        else:
            raise NotImplementedError(
                f"{self.__class__.__name__}.transform_reward_spec not "
                f"implemented for tensor spec of type"
                f" {type(reward_spec).__name__}"
            )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"loc={self.loc.item():4.4f}, scale={self.scale.item():4.4f}, "
            f"keys={self.keys})"
        )


class FiniteTensorDictCheck(Transform):
    """
    This transform will check that all the items of the tensordict are
    finite, and raise an exception if they are not.

    """

    inplace = False

    def __init__(self):
        super().__init__(keys=[])

    def _call(self, tensordict: _TensorDict) -> _TensorDict:
        source = {}
        for key, item in tensordict.items():
            try:
                source[key] = FiniteTensor(item)
            except RuntimeError as err:
                if str(err).rfind("FiniteTensor encountered") > -1:
                    raise ValueError(f"Found non-finite elements in {key}")
                else:
                    raise RuntimeError(str(err))

        finite_tensordict = TensorDict(batch_size=tensordict.batch_size, source=source)
        return finite_tensordict


class DoubleToFloat(Transform):
    """
    Maps actions float to double before they are called on the environment.

    Examples:
        >>> td = TensorDict(
        ...     {'next_obs': torch.ones(1, dtype=torch.double)}, [])
        >>> transform = DoubleToFloat(keys=["next_obs"])
        >>> _ = transform(td)
        >>> print(td.get("next_obs").dtype)
        torch.float32

    """

    invertible = True
    inplace = False

    def __init__(self, keys: Optional[Sequence[str]] = None):
        if keys is None:
            keys = ["action"]
        super().__init__(keys=keys)

    def _apply(self, obs: torch.Tensor) -> torch.Tensor:
        return obs.to(torch.float)

    def _inv_apply(self, obs: torch.Tensor) -> torch.Tensor:
        return obs.to(torch.double)

    def _transform_spec(self, spec: TensorSpec) -> None:
        if isinstance(spec, CompositeSpec):
            for key in spec:
                self._transform_spec(spec[key])
        else:
            spec.dtype = torch.float
            space = spec.space
            if isinstance(space, ContinuousBox):
                space.minimum = space.minimum.to(torch.float)
                space.maximum = space.maximum.to(torch.float)

    def transform_action_spec(self, action_spec: TensorSpec) -> TensorSpec:
        if "action" in self.keys:
            if action_spec.dtype is not torch.double:
                raise TypeError("action_spec.dtype is not double")
            self._transform_spec(action_spec)
        return action_spec

    def transform_reward_spec(self, reward_spec: TensorSpec) -> TensorSpec:
        if "reward" in self.keys:
            if reward_spec.dtype is not torch.double:
                raise TypeError("reward_spec.dtype is not double")

            self._transform_spec(reward_spec)
        return reward_spec

    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        if isinstance(observation_spec, CompositeSpec):
            keys = [key for key in self.keys if key in observation_spec.keys()]
            for key in keys:
                observation_spec[key] = self.transform_observation_spec(
                    observation_spec[key]
                )
            return observation_spec
        self._transform_spec(observation_spec)
        return observation_spec


class CatTensors(Transform):
    """
    Concatenates several keys in a single tensor.
    This is especially useful if multiple keys describe a single state (e.g.
    "observation_position" and
    "observation_velocity")

    Args:
        keys (Sequence of str): keys to be concatenated
        out_key: key of the resulting tensor.
        dim (int, optional): dimension along which the contenation will occur.
            Default is -1.
        del_keys (bool, optional): if True, the input values will be deleted after
            concatenation. Default is True.

    Examples:
        >>> transform = CatTensors(keys=["key1", "key2"])
        >>> td = TensorDict({"key1": torch.zeros(1, 1),
        ...     "key2": torch.ones(1, 1)}, [1])
        >>> _ = transform(td)
        >>> print(td.get("observation_vector"))
        tensor([[0., 1.]])

    """

    invertible = False
    inplace = False

    def __init__(
        self,
        keys: Optional[Sequence[str]] = None,
        out_key: str = "observation_vector",
        dim: int = -1,
        del_keys: bool = True,
    ):
        if keys is None:
            raise Exception("CatTensors requires keys to be non-empty")
        super().__init__(keys=keys)
        if not out_key.startswith("next_") and all(
            key.startswith("next_") for key in keys
        ):
            warn(
                f"It seems that 'next_'-like keys are being concatenated to a non 'next_' key {out_key}. This may result in unwanted behaviours, and the 'next_' flag is missing from the output key."
                f"Consider renaming the out_key to 'next_{out_key}'"
            )
        self.out_key = out_key
        self.keys = sorted(list(self.keys))
        if (
            ("reward" in self.keys)
            or ("action" in self.keys)
            or ("reward" in self.keys)
        ):
            raise RuntimeError(
                "Concatenating observations and reward / action / done state "
                "is not allowed."
            )
        self.dim = dim
        self.del_keys = del_keys

    def _call(self, tensordict: _TensorDict) -> _TensorDict:
        if all([key in tensordict.keys() for key in self.keys]):
            out_tensor = torch.cat(
                [tensordict.get(key) for key in self.keys], dim=self.dim
            )
            tensordict.set(self.out_key, out_tensor)
            if self.del_keys:
                tensordict.exclude(*self.keys, inplace=True)
        else:
            raise Exception(
                f"CatTensor failed, as it expected input keys ="
                f" {sorted(list(self.keys))} but got a TensorDict with keys"
                f" {sorted(list(tensordict.keys()))}"
            )
        return tensordict

    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        if not isinstance(observation_spec, CompositeSpec):
            # then there is a single tensor to be concatenated
            return observation_spec

        keys = [key for key in observation_spec._specs.keys() if key in self.keys]

        sum_shape = sum(
            [
                observation_spec[key].shape[self.dim]
                if observation_spec[key].shape
                else 1
                for key in keys
            ]
        )
        spec0 = observation_spec[keys[0]]
        out_key = self.out_key
        shape = list(spec0.shape)
        shape[self.dim] = sum_shape
        shape = torch.Size(shape)
        observation_spec[out_key] = NdUnboundedContinuousTensorSpec(
            shape=shape,
            dtype=spec0.dtype,
        )
        if self.del_keys:
            for key in self.keys:
                del observation_spec._specs[key]
        return observation_spec

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(in_keys={self.keys}, out_key"
            f"={self.out_key})"
        )


class DiscreteActionProjection(Transform):
    """Projects discrete actions from a high dimensional space to a low
    dimensional space.

    Given a discrete action (from 1 to N) encoded as a one-hot vector and a
    maximum action index M (with M < N), transforms the action such that
    action_out is at most M.

    If the input action is > M, it is being replaced by a random value
    between N and M. Otherwise the same action is kept.
    This is intended to be used with policies applied over multiple discrete
    control environments with different action space.

    Args:
        max_N (int): max number of action considered.
        M (int): resulting number of actions.

    Examples:
        >>> torch.manual_seed(0)
        >>> N = 2
        >>> M = 1
        >>> action = torch.zeros(N, dtype=torch.long)
        >>> action[-1] = 1
        >>> td = TensorDict({"action": action}, [])
        >>> transform = DiscreteActionProjection(N, M)
        >>> _ = transform.inv(td)
        >>> print(td.get("action"))
        tensor([1])
    """

    inplace = False

    def __init__(self, max_N: int, M: int, action_key: str = "action"):
        super().__init__([action_key])
        self.max_N = max_N
        self.M = M

    def _inv_apply(self, action: torch.Tensor) -> torch.Tensor:
        if action.shape[-1] < self.M:
            raise RuntimeError(
                f"action.shape[-1]={action.shape[-1]} is smaller than "
                f"DiscreteActionProjection.M={self.M}"
            )
        action = action.argmax(-1)  # bool to int
        idx = action >= self.M
        if idx.any():
            action[idx] = torch.randint(self.M, (idx.sum(),))
        action = nn.functional.one_hot(action, self.M)
        return action

    def transform_action_spec(self, action_spec: TensorSpec) -> TensorSpec:
        shape = action_spec.shape
        shape = torch.Size([*shape[:-1], self.max_N])
        action_spec.shape = shape
        action_spec.space.n = self.max_N
        return action_spec

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(max_N={self.max_N}, M={self.M}, "
            f"keys={self.keys})"
        )


class NoopResetEnv(Transform):
    """
    Runs a series of random actions when an environment is reset.

    Args:
        env (_EnvClass): env on which the random actions have to be
            performed. Can be the same env as the one provided to the
            TransformedEnv class
        noops (int, optional): number of actions performed after reset.
            Default is `30`.
        random (bool, optional): if False, the number of random ops will
            always be equal to the noops value. If True, the number of
            random actions will be randomly selected between 0 and noops.
            Default is `True`.

    """

    inplace = True

    def __init__(self, env: _EnvClass, noops: int = 30, random: bool = True):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        super().__init__([])
        self.env = env
        self.noops = noops
        self.random = random

    def reset(self, tensordict: _TensorDict) -> _TensorDict:
        """Do no-op action for a number of steps in [1, noop_max]."""
        keys = tensordict.keys()
        noops = (
            self.noops if not self.random else torch.randint(self.noops, (1,)).item()
        )
        i = 0
        trial = 0
        while i < noops:
            i += 1
            tensordict = self.env.rand_step()
            if self.env.is_done:
                self.env.reset()
                i = 0
                trial += 1
                if trial > _MAX_NOOPS_TRIALS:
                    self.env.reset()
                    tensordict = self.env.rand_step()
                    break
        if self.env.is_done:
            raise RuntimeError("NoopResetEnv concluded with done environment")
        td = step_tensordict(tensordict).select(*keys)
        for k in keys:
            if k not in td.keys():
                td.set(k, tensordict.get(k))
        return td

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(noops={self.noops}, random"
            f"={self.random}, keys={self.keys})"
        )


class PinMemoryTransform(Transform):
    """
    Calls pin_memory on the tensordict to facilitate writing on CUDA devices.

    """

    def __init__(self):
        super().__init__([])

    def _call(self, tensordict: _TensorDict) -> _TensorDict:
        return tensordict.pin_memory()


def _sum_left(val, dest):
    while val.ndimension() > dest.ndimension():
        val = val.sum(0)
    return val


class gSDENoise(Transform):
    inplace = False

    def __init__(
        self,
        action_dim: int,
        state_dim: Optional[int] = None,
        observation_key="next_observation_vector",
    ) -> None:
        super().__init__(keys=[])
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.observation_key = observation_key

    def reset(self, tensordict: _TensorDict) -> _TensorDict:
        tensordict = super().reset(tensordict=tensordict)
        if self.state_dim is None:
            raise RuntimeError("state_dim must be provided to gSDENoise transform")
        else:
            state_dim = self.state_dim

        tensordict.set(
            "_eps_gSDE",
            torch.randn(
                *tensordict.batch_size,
                self.action_dim,
                state_dim,
                device=tensordict.device,
            ),
        )
        return tensordict


class MultiAgentTransform(Transform):
    inplace = False

    def __init__(self, max_agents: int, agent_key="agents"):
        super().__init__([])
        self.agent_key = agent_key
        self.max_agents = max_agents

    def _apply(self, obs: torch.Tensor, agents: torch.Tensor) -> None:
        # this will probably break if obs carries gradients
        obs_out = torch.zeros(
            self.max_agents, *obs.shape[1:], dtype=obs.dtype, device=obs.device
        )
        obs_out[agents] = obs
        return obs_out

    def _call(self, tensordict: TensorDict) -> TensorDict:
        self._check_inplace()
        agents = tensordict.get(self.agent_key)
        tensordict = TensorDict(
            {
                _obs_key: self._apply(item, agents)
                for _obs_key, item in tensordict.items()
            },
            [self.max_agents, *tensordict.batch_size[1:]],
        )

        mask = torch.zeros(self.max_agents, dtype=torch.bool)
        mask[agents] = 1
        tensordict.set("mask", mask)
        return tensordict

    def transform_batch_size(self, batch_size: torch.Size) -> torch.Size:
        return torch.Size([self.max_agents, *batch_size[1:]])


class VecNorm(Transform):
    """
    Moving average normalization layer for torchrl environments.
    VecNorm keeps track of the summary statistics of a dataset to standardize
    it on-the-fly. If the transform is in 'eval' mode, the running
    statistics are not updated.

    If multiple processes are running a similar environment, one can pass a
    _TensorDict instance that is placed in shared memory: if so, every time
    the normalization layer is queried it will update the values for all
    processes that share the same reference.

    Args:
        keys (iterable of str, optional): keys to be updated.
            default: ["next_observation", "reward"]
        shared_td (_TensorDict, optional): A shared tensordict containing the
            keys of the transform.
        decay (number, optional): decay rate of the moving average.
            default: 0.99
        eps (number, optional): lower bound of the running standard
            deviation (for numerical underflow). Default is 1e-4.

    Examples:
        >>> from torchrl.envs import GymEnv
        >>> t = VecNorm(decay=0.9)
        >>> env = GymEnv("Pendulum-v0")
        >>> env = TransformedEnv(env, t)
        >>> tds = []
        >>> for _ in range(1000):
        ...     td = env.rand_step()
        ...     if td.get("done"):
        ...         _ = env.reset()
        ...     tds += [td]
        >>> tds = torch.stack(tds, 0)
        >>> print((abs(tds.get("next_observation").mean(0))<0.2).all())
        tensor(True)
        >>> print((abs(tds.get("next_observation").std(0)-1)<0.2).all())
        tensor(True)

    """

    inplace = True

    def __init__(
        self,
        keys: Optional[Sequence[str]] = None,
        shared_td: Optional[_TensorDict] = None,
        decay: float = 0.9999,
        eps: float = 1e-4,
    ) -> None:
        if keys is None:
            keys = ["next_observation", "reward"]
        super().__init__(keys)
        self._td = shared_td
        if shared_td is not None and not (
            shared_td.is_shared() or shared_td.is_memmap()
        ):
            raise RuntimeError(
                "shared_td must be either in shared memory or a memmap " "tensordict."
            )
        if shared_td is not None:
            for key in keys:
                if (
                    (key + "_sum" not in shared_td.keys())
                    or (key + "_ssq" not in shared_td.keys())
                    or (key + "_count" not in shared_td.keys())
                ):
                    raise KeyError(
                        f"key {key} not present in the shared tensordict "
                        f"with keys {shared_td.keys()}"
                    )

        self.decay = decay
        self.eps = eps

    def _call(self, tensordict: _TensorDict) -> _TensorDict:
        for key in self.keys:
            if key not in tensordict.keys():
                continue
            self._init(tensordict, key)
            # update anb standardize
            new_val = self._update(
                key, tensordict.get(key), N=max(1, tensordict.numel())
            )

            tensordict.set_(key, new_val)
        return tensordict

    def _init(self, tensordict: _TensorDict, key: str) -> None:
        if self._td is None or key + "_sum" not in self._td.keys():
            td_view = tensordict.view(-1)
            td_select = td_view[0]
            d = {key + "_sum": torch.zeros_like(td_select.get(key))}
            d.update({key + "_ssq": torch.zeros_like(td_select.get(key))})
            d.update(
                {
                    key
                    + "_count": torch.zeros(
                        1, device=td_select.get(key).device, dtype=torch.float
                    )
                }
            )
            if self._td is None:
                self._td = TensorDict(d, batch_size=[])
            else:
                self._td.update(d)
        else:
            pass

    def _update(self, key, value, N) -> torch.Tensor:
        _sum = self._td.get(key + "_sum")
        _ssq = self._td.get(key + "_ssq")
        _count = self._td.get(key + "_count")

        if self.training:
            value_sum = _sum_left(value, _sum)
            value_ssq = _sum_left(value.pow(2), _ssq)

            _sum = self.decay * _sum + value_sum
            _ssq = self.decay * _ssq + value_ssq
            _count = self.decay * _count + N

            self._td.set_(key + "_sum", _sum)
            self._td.set_(key + "_ssq", _ssq)
            self._td.set_(key + "_count", _count)

        mean = _sum / _count
        std = (_ssq / _count - mean.pow(2)).clamp_min(self.eps).sqrt()
        return (value - mean) / std.clamp_min(self.eps)

    @staticmethod
    def build_td_for_shared_vecnorm(
        env: _EnvClass,
        keys_prefix: Optional[Sequence[str]] = None,
        memmap: bool = False,
    ) -> _TensorDict:
        """Creates a shared tensordict that can be sent to different processes
        for normalization across processes.

        Args:
            env (_EnvClass): example environment to be used to create the
                tensordict
            keys_prefix (iterable of str, optional): prefix of the keys that
                have to be normalized. Default is `["next_", "reward"]`
            memmap (bool): if True, the resulting tensordict will be cast into
                memmory map (using `memmap_()`). Otherwise, the tensordict
                will be placed in shared memory.

        Returns:
            A memory in shared memory to be sent to each process.

        Examples:
            >>> from torch import multiprocessing as mp
            >>> queue = mp.Queue()
            >>> env = make_env()
            >>> td_shared = VecNorm.build_td_for_shared_vecnorm(env,
            ...     ["next_observation", "reward"])
            >>> assert td_shared.is_shared()
            >>> queue.put(td_shared)
            >>> # on workers
            >>> v = VecNorm(shared_td=queue.get())
            >>> env = TransformedEnv(make_env(), v)

        """
        if keys_prefix is None:
            keys_prefix = ["next_", "reward"]
        td = make_tensordict(env)
        keys = set(
            key
            for key in td.keys()
            if any(key.startswith(_prefix) for _prefix in keys_prefix)
        )
        td_select = td.select(*keys)
        if td.batch_dims:
            raise RuntimeError(
                f"VecNorm should be used with non-batched environments. "
                f"Got batch_size={td.batch_size}"
            )
        for key in keys:
            td_select.set(key + "_ssq", td_select.get(key).clone())
            td_select.set(
                key + "_count",
                torch.zeros(
                    *td.batch_size,
                    1,
                    device=td_select.device,
                    dtype=torch.float,
                ),
            )
            td_select.rename_key(key, key + "_sum")
        td_select.zero_()
        if memmap:
            return td_select.memmap_()
        return td_select.share_memory_()

    def get_extra_state(self) -> _TensorDict:
        return self._td

    def set_extra_state(self, td: _TensorDict) -> None:
        if not td.is_shared():
            raise RuntimeError(
                "Only shared tensordicts can be set in VecNorm transforms"
            )
        self._td = td

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(decay={self.decay:4.4f},"
            f"eps={self.eps:4.4f}, keys={self.keys})"
        )
