from copy import deepcopy
from numbers import Number
from typing import Optional, Any, Iterable, Union, List
from warnings import warn

import torch
from torch import nn
from torchvision.transforms.functional_tensor import (
    resize,
)  # as of now resize is imported from torchvision

from torchrl.data.tensor_specs import UnboundedContinuousTensorSpec, CompositeSpec, BoundedTensorSpec, ContinuousBox, \
    NdUnboundedContinuousTensorSpec, TensorSpec
from torchrl.data.tensordict.tensordict import TensorDict, _TensorDict
from torchrl.envs.common import _EnvClass
from . import functional as F
from .utils import FiniteTensor

__all__ = [
    "TransformedEnv",
    "RewardClipping",
    "Resize",
    "GrayScale",
    "Compose",
    "ToTensorImage",
    "ObservationNorm",
    "DataDependentObservationNorm",
    "RewardScaling",
    "ObservationTransform",
    "Transform",
    "CatFrames",
    "FiniteTensorDictCheck",
    "DoubleToFloat",
    "CatTensors",
    "NoopResetEnv",
    "BinerizeReward",
    "PinMemoryTransform",
]

from ...envs.utils import step_tensor_dict

IMAGE_KEYS = ["next_observation", "next_observation_pixels"]


class Transform(nn.Module):
    """
    Environment transform.
    In principle, a transform receives a tensordict as input and returns (the same or another) tensordict as output,
    where a series of keys have been modified or created.
    When instantiating a new transform, it should always be possible to indicate what keys are to be read for the
    transform by passing the `keys` argument to the constructor.
    Transforms can be combined with environments with the TransformedEnv class, which takes as arguments am _EnvClass
    instance and a transform.
    Transforms can be concatenated using the `Compose` class.
    They can be stateless or stateful (e.g. CatTransform). Because of this, Transforms support the `reset` operation,
    which should reset the transform to its initial state (such that successive trajectories are kept independent).

    """
    invertible = False

    def __init__(self, keys: Iterable):
        super().__init__()
        self.keys = keys

    def reset(self, tensor_dict: _TensorDict) -> _TensorDict:
        """
        Resets a tranform if it is stateful.

        """
        return tensor_dict

    def _check_inplace(self) -> None:
        if not hasattr(self, "inplace"):
            raise AttributeError(f"Transform of class {self.__class__.__name__} has no attribute inplace, consider "
                                 f"implementing it.")

    def init(self, tensor_dict) -> None:
        pass

    def _apply(self, obs: torch.Tensor) -> None:
        """
        Applies the transform to a tensor.
        This operation can be called multiple times (if multiples keys of the tensordict match the keys of the
        transform).

        """
        raise NotImplementedError

    def _call(self, tensor_dict: _TensorDict) -> _TensorDict:
        """
        Reads the input tensordict, and for the selected keys, applies the transform.

        """
        self._check_inplace()
        for _obs_key in tensor_dict.keys():
            if _obs_key in self.keys:
                observation = self._apply(tensor_dict.get(_obs_key))
                tensor_dict.set(_obs_key, observation, inplace=self.inplace)
        return tensor_dict

    def forward(self, tensor_dict: _TensorDict) -> _TensorDict:
        self._call(tensor_dict)
        return tensor_dict

    def _inv_apply(self, obs: torch.Tensor) -> torch.Tensor:
        if self.invertible:
            raise NotImplementedError
        else:
            return obs

    def _inv_call(self, tensor_dict: _TensorDict) -> _TensorDict:
        self._check_inplace()
        for _obs_key in tensor_dict.keys():
            if _obs_key in self.keys:
                observation = self._inv_apply(tensor_dict.get(_obs_key))
                tensor_dict.set(_obs_key, observation, inplace=self.inplace)
        return tensor_dict

    def inv(self, tensor_dict: _TensorDict) -> _TensorDict:
        self._inv_call(tensor_dict)
        return tensor_dict

    def transform_action_spec(self, action_spec: TensorSpec) -> TensorSpec:
        """
        Transforms the action spec such that the resulting spec matches transform mapping.
        Args:
            action_spec (TensorSpec): spec before the transform

        Returns: expected spec after the transform

        """
        return action_spec

    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        """
        Transforms the observation spec such that the resulting spec matches transform mapping.
        Args:
            observation_spec (TensorSpec): spec before the transform

        Returns: expected spec after the transform

        """
        return observation_spec

    def transform_reward_spec(self, reward_spec: TensorSpec) -> TensorSpec:
        """
        Transforms the reward spec such that the resulting spec matches transform mapping.
        Args:
            reward_spec (TensorSpec): spec before the transform

        Returns: expected spec after the transform

        """

        return reward_spec

    def dump(self) -> None:
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class TransformedEnv(_EnvClass):
    """
    A transformed environment.

    Args:
        env (_EnvClass): original environment to be transformed.
        transform (Transform): transform to apply to the tensordict resulting from env.step(td)
        cache_specs (bool): if True, the specs will be cached once and for all after the first call (i.e. the
            specs will be transformed only once). If the transform changes during training, the original spec transform
            may not be valid anymore, in which case this value should be set to False.

    Examples:
        >>> env = GymEnv("Pendulum-v0")
        >>> transform = RewardScaling(0.0, 1.0)
        >>> transformed_env = TransformedEnv(env, transform)

    """

    def __init__(self, env: _EnvClass, transform: Transform, cache_specs: bool = True, **kwargs):
        self.env = env
        self.transform = transform
        self._last_obs = None
        self.cache_specs = cache_specs

        self._action_spec = None
        self._reward_spec = None
        self._observation_spec = None
        self.batch_size = self.env.batch_size

        super().__init__(**kwargs)

    @property
    def observation_spec(self) -> TensorSpec:
        """
        Observation spec of the transformed environment

        """
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
        """
        Action spec of the transformed environment

        """

        if self._action_spec is None or not self.cache_specs:
            action_spec = self.transform.transform_action_spec(deepcopy(self.env.action_spec))
            if self.cache_specs:
                self._action_spec = action_spec
        else:
            action_spec = self._action_spec
        return action_spec

    @property
    def reward_spec(self) -> TensorSpec:
        """
        Reward spec of the transformed environment

        """

        if self._reward_spec is None or not self.cache_specs:
            reward_spec = self.transform.transform_reward_spec(deepcopy(self.env.reward_spec))
            if self.cache_specs:
                self._reward_spec = reward_spec
        else:
            reward_spec = self._reward_spec
        return reward_spec

    def _step(self, tensor_dict: _TensorDict) -> _TensorDict:
        selected_keys = [key for key in tensor_dict.keys() if "action" in key]
        tensor_dict_in = tensor_dict.select(*selected_keys)
        tensor_dict_in = self.transform.inv(tensor_dict_in)
        tensor_dict_out = self.env._step(tensor_dict_in).to(self.device)
        # tensor_dict should already have been processed by the transforms
        # for logging purposes
        tensor_dict_out = self.transform(tensor_dict_out.clone())
        return tensor_dict_out

    def set_seed(self, seed: int) -> int:
        """
        Set the seeds of the environment

        """
        return self.env.set_seed(seed)

    def _reset(self, tensor_dict: Optional[_TensorDict] = None):
        out_tensor_dict = self.env.reset().to(self.device)
        out_tensor_dict = self.transform.reset(out_tensor_dict)

        # Transforms are made for "next_observations" and alike. We convert all the observations in next_observations,
        # then map them back to their original key name
        keys = list(out_tensor_dict.keys())
        for key in keys:
            if key.startswith("observation"):
                out_tensor_dict.rename_key(key, "next_" + key, safe=True)

        out_tensor_dict = self.transform(out_tensor_dict)
        keys = list(out_tensor_dict.keys())
        for key in keys:
            if key.startswith("next_observation"):
                out_tensor_dict.rename_key(key, key[5:], safe=True)
        return out_tensor_dict

    def __getattr__(self, attr: str) -> Any:
        if attr in self.__dir__():
            return self.__getattribute__(attr)  # make sure that appropriate exceptions are raised

        try:
            env = self.__getattribute__("env")
        except:
            raise Exception(
                f"env not set in {self.__class__.__name__}, cannot access {attr}"
            )
        return getattr(env, attr)

    def __repr__(self) -> str:
        return f"TransformedEnv(env={self.env}, transform={self.transform})"


class ObservationTransform(Transform):
    """
    Abstract class for transformations of the observations.

    """
    inplace = False

    def __init__(self, keys: Optional[Iterable[str]] = None):
        if keys is None:
            keys = [
                "next_observation",
                "next_observation_pixels",
                "next_observation_state"
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
        self.transforms = transforms

    def _call(self, tensor_dict: _TensorDict) -> _TensorDict:
        for t in self.transforms:
            tensor_dict = t(tensor_dict)
        return tensor_dict

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

    def dump(self) -> None:
        for t in self:
            t.dump()

    def reset(self, tensor_dict: _TensorDict) -> _TensorDict:
        for t in self.transforms:
            tensor_dict = t.reset(tensor_dict)
        return tensor_dict

    def init(self, tensor_dict: _TensorDict) -> None:
        for t in self.transforms:
            t.init(tensor_dict)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({', '.join([str(l) for l in self.transforms])})"


class ToTensorImage(ObservationTransform):
    """
    Transforms an observation image from a (... x W x H x 3) 0..255 uint8 tensor to a single/double precision floating
    point (3 x W x H) tensor with values between 0 and 1.

    Args:
        unsqueeze (bool): if True, the observation tensor is unsqueezed along the first dimension. default=False.
        dtype (optional): dtype to use for the resulting observations.
    """
    inplace = False

    def __init__(self, unsqueeze: bool = False, dtype: Optional[torch.device] = None,
                 keys: Optional[Iterable[str]] = None):
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
        _observation_spec = observation_spec["pixels"]
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
        observation_spec["pixels"] = _observation_spec
        return observation_spec

    def _pixel_observation(self, spec: TensorSpec) -> None:
        if isinstance(spec, BoundedTensorSpec):
            spec.space.maximum = self._apply(spec.space.maximum)
            spec.space.minimum = self._apply(spec.space.minimum)


class RewardClipping(Transform):
    """
    Clips the reward between clamp_min and clamp_max.

    Args:
        clip_min (scalar): minimum value of the resulting reward
        clip_max (scalar): maximum value of the resulting reward
    """
    inplace = True

    def __init__(self, clamp_min: Number = None, clamp_max: Number = None, keys: Optional[Iterable[str]] = None):
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
            return BoundedTensorSpec(self.clamp_min, self.clamp_max, device=reward_spec.device, dtype=reward_spec.dtype)
        else:
            raise NotImplementedError(
                f"{self.__class__.__name__}.transform_reward_spec not implemented for tensor spec of type {type(reward_spec).__name__}"
            )


class BinerizeReward(Transform):
    """
    Maps the reward to a binary value (0 or 1) if the reward is null or non-null, respectively.

    """
    inplace = True

    def __init__(self, keys: Optional[Iterable[str]] = None):
        if keys is None:
            keys = ["reward"]
        super().__init__(keys=keys)

    def _apply(self, reward: torch.Tensor) -> torch.Tensor:
        return (reward != 0.0).to(reward.dtype)

    def transform_reward_spec(self, reward_spec: TensorSpec) -> TensorSpec:
        if isinstance(reward_spec, UnboundedContinuousTensorSpec):
            return BoundedTensorSpec(0.0, 1.0, device=reward_spec.device, dtype=reward_spec.dtype)
        else:
            raise NotImplementedError(
                f"{self.__class__.__name__}.transform_reward_spec not implemented for tensor spec of type {type(reward_spec).__name__}"
            )


class Resize(ObservationTransform):
    """
    Resizes an pixel observation.

    Args:
        w (int): resulting width
        h (int): resulting height
        interpolation (str): interpolation method
    """
    inplace = False

    def __init__(self, w: int, h: int, interpolation: str = "bilinear", keys: Optional[Iterable[str]] = None):
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
        _observation_spec = observation_spec["pixels"]
        space = _observation_spec.space
        if isinstance(space, ContinuousBox):
            space.minimum = self._apply(space.minimum)
            space.maximum = self._apply(space.maximum)
            _observation_spec.shape = space.minimum.shape
        else:
            _observation_spec.shape = self._apply(torch.zeros(_observation_spec.shape)).shape
        observation_spec["pixels"] = _observation_spec
        return observation_spec


class GrayScale(ObservationTransform):
    """
    Turns a pixel observation to grayscale.

    """
    inplace = False

    def __init__(self, keys: Optional[Iterable[str]] = None):
        if keys is None:
            keys = IMAGE_KEYS
        super(GrayScale, self).__init__(keys=keys)

    def _apply(self, observation: torch.Tensor) -> torch.Tensor:
        observation = F.rgb_to_grayscale(observation)
        return observation

    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        _observation_spec = observation_spec["pixels"]
        space = _observation_spec.space
        if isinstance(space, ContinuousBox):
            space.minimum = self._apply(space.minimum)
            space.maximum = self._apply(space.maximum)
            _observation_spec.shape = space.minimum.shape
        else:
            _observation_spec.shape = self._apply(torch.zeros(_observation_spec.shape)).shape
        observation_spec["pixels"] = _observation_spec
        return observation_spec


class ObservationNorm(ObservationTransform):
    """
    Normalizes an observation according to
        obs = obs * scale + loc

    Args:
        loc (number or tensor): location of the affine transform
        scale (number or tensor): scale of the affine transform
        standard_normal (bool): if True, the transform will be
            obs = (obs-loc)/scale,
            as it is done for standardization. default=False
    """

    inplace = True

    def __init__(
            self,
            loc: Union[Number, torch.Tensor],
            scale: Union[Number, torch.Tensor],
            keys: Optional[Iterable[str]] = None,
            # observation_spec_key: =None,
            standard_normal: bool = False,
    ):
        if keys is None:
            keys = ["next_observation", "next_observation_pixels", "next_observation_state"]
        super().__init__(keys=keys)
        if not isinstance(loc, torch.Tensor):
            loc = torch.tensor(loc, dtype=torch.float)
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale, dtype=torch.float)

        # self.observation_spec_key = observation_spec_key
        self.standard_normal = standard_normal
        self.loc = loc
        eps = 1e-6
        self.scale = scale.clamp_min(eps)

        if self.standard_normal:
            # converts the transform (x-m)/sqrt(v) to x * s + loc
            self.scale = self.scale.reciprocal()
            self.loc = -self.loc * self.scale

    def _apply(self, obs: torch.Tensor) -> torch.Tensor:
        return obs * self.scale + self.loc

    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        if isinstance(observation_spec, CompositeSpec):
            key = [key.split("observation_")[-1] for key in self.keys]
            if len(set(key)) != 1:
                raise RuntimeError(f"Too many compatible observation keys: {key}")
            key = key[0]
            _observation_spec = observation_spec[key]
        else:
            _observation_spec = observation_spec
        space = _observation_spec.space
        if isinstance(space, ContinuousBox):
            space.minimum = self._apply(space.minimum)
            space.maximum = self._apply(space.maximum)
        return observation_spec


class DataDependentObservationNorm(ObservationNorm):
    """
    Experimental feature: Data dependent affine standardization.
    """
    inplace = True

    def __init__(self, dims: Iterable[int] = (-1, -2), keys: Optional[Iterable[str]] = None,
                 ):
        warn("DataDependentObservationNorm is an experimental feature under heavy development. It should not be used "
             "in parallel distributed settings.")
        if keys is None:
            keys = ["next_observation", "next_observation_pixels", "next_observation_state"]
        super().__init__(keys=keys)
        self.dims = dims
        self.initialized = False

    def _init(self, obs: torch.Tensor) -> None:
        if not torch.isfinite(obs).all():
            raise ValueError("Non-finite observation found")
        loc = obs.mean(self.dims, True)
        scale = obs.std(self.dims, True).clamp_min(1e-6)
        loc = -loc / scale
        scale = scale.reciprocal()
        self.register_buffer("loc", loc)
        self.register_buffer("scale", scale)
        self.initialized = True

    def init(self, tensor_dict: _TensorDict) -> None:
        for key in self.keys:
            if key in tensor_dict.keys():
                self._init(tensor_dict.get(key))
                break

    def _call(self, tensor_dict: _TensorDict) -> _TensorDict:
        if not self.initialized:
            return tensor_dict
        return super()._call(tensor_dict)

    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        if not self.initialized:
            return observation_spec
        return super().transform_observation_spec(observation_spec)


class CatFrames(ObservationTransform):
    """
    Concatenates successive observation frames into a single tensor.
    This can, for instance, account for movement/velocity of the observed feature.

    Args:
        N (int): number of observation to concatenate
        cat_dim (int): dimension along which concatenate the observations.

    """
    inplace = False

    def __init__(self, N: int = 4, cat_dim: int = -3, keys: Optional[Iterable[str]] = None):
        if keys is None:
            keys = IMAGE_KEYS
        super().__init__(keys=keys)
        self.N = N
        self.cat_dim = cat_dim
        self.buffer = []

    def reset(self, tensor_dict: _TensorDict) -> _TensorDict:
        self.buffer = []
        return tensor_dict

    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        _observation_spec = observation_spec["pixels"]
        space = _observation_spec.space
        if isinstance(space, ContinuousBox):
            space.minimum = torch.cat([space.minimum] * self.N, 0)
            space.maximum = torch.cat([space.maximum] * self.N, 0)
            _observation_spec.shape = space.minimum.shape
        else:
            _observation_spec.shape = torch.Size([self.N, *_observation_spec.shape])
        observation_spec["pixels"] = _observation_spec
        return observation_spec

    def _apply(self, obs: torch.Tensor) -> torch.Tensor:
        self.buffer.append(obs)
        self.buffer = self.buffer[-self.N:]
        buffer = list(reversed(self.buffer))
        buffer = [buffer[0]] * (self.N - len(buffer)) + buffer
        if len(buffer) != self.N:
            raise RuntimeError(f"actual buffer length ({buffer}) differs from expected ({N})")
        return torch.cat(buffer, self.cat_dim)


class RewardScaling(Transform):
    """
    Affine transform of the reward according to
        reward = reward * scale + loc

    Args:
        loc (number or torch.Tensor): location of the affine transform
        scale (number or torch.Tensor): scale of the affine transform
    """
    inplace = True

    def __init__(self, loc: Union[Number, torch.Tensor], scale: Union[Number, torch.Tensor],
                 keys: Optional[Iterable[str]] = None):
        if keys is None:
            keys = ["reward"]
        super().__init__(keys=keys)
        if not isinstance(loc, torch.Tensor):
            loc = torch.tensor(loc)
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale)

        self.loc = loc
        self.scale = scale.clamp_min(1e-6)

    def _apply(self, reward: torch.Tensor) -> torch.Tensor:
        reward.mul_(self.scale).add_(self.loc)
        return reward

    def transform_reward_spec(self, reward_spec: TensorSpec) -> TensorSpec:
        if isinstance(reward_spec, UnboundedContinuousTensorSpec):
            return reward_spec
        else:
            raise NotImplementedError(
                f"{self.__class__.__name__}.transform_reward_spec not implemented for tensor spec of type {type(reward_spec).__name__}"
            )


class FiniteTensorDictCheck(Transform):
    """
    This transform will check that all the items of the tensordict are finite, and raise an exception if they are not.

    """
    inplace = False

    def __init__(self):
        super().__init__(keys=[])

    def _call(self, tensor_dict: _TensorDict) -> _TensorDict:
        source = {}
        for key, item in tensor_dict.items():
            try:
                source[key] = FiniteTensor(item)
            except AssertionError:
                raise Exception(f"Found non-finite elements in {key}")

        finite_tensor_dict = TensorDict(batch_size=tensor_dict.batch_size, source=source)
        return finite_tensor_dict


class DoubleToFloat(Transform):
    """
    Maps actions float to double before they are called on the environment.

    """
    invertible = True
    inplace = False

    def __init__(self, keys: Optional[Iterable[str]] = None):
        if keys is None:
            keys = ["action"]
        super().__init__(keys=keys)

    def _apply(self, obs: torch.Tensor) -> torch.Tensor:
        return obs.to(torch.float)

    def _inv_apply(self, obs: torch.Tensor) -> torch.Tensor:
        return obs.to(torch.double)

    def _transform_spec(self, spec: TensorSpec) -> None:
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
        keys = [key for key in self.keys if "observation" in key]
        if keys:
            keys = [key.split("observation_")[-1] for key in keys]
            for key in keys:
                self._transform_spec(observation_spec[key])
        return observation_spec


class CatTensors(Transform):
    """
    Concatenates several keys toghether in a single tensor.
    This is especially useful if multiple keys describe a single state (e.g. "observation_position" and
    "observation_velocity")

    Args:
        keys (Iterable of str): keys to be concatenated
        out_key: key of the resulting tensor.

    """
    invertible = False
    inplace = False

    def __init__(self, keys: Optional[Iterable[str]] = None, out_key: str = "observation_vector"):
        if keys is None:
            raise Exception("CatTensors requires keys to be non-empty")
        super().__init__(keys=keys)
        if "observation_" not in out_key:
            raise KeyError("CatTensors is currently restricted to observation_* keys")
        self.out_key = out_key
        self.keys = sorted(list(self.keys))

    def _call(self, tensor_dict: _TensorDict) -> _TensorDict:
        if all([key in tensor_dict.keys() for key in self.keys]):
            out_tensor = torch.cat([tensor_dict.get(key) for key in self.keys], -1)
            tensor_dict.set(self.out_key, out_tensor)
            for key in self.keys:
                tensor_dict.del_(key)
        else:
            raise Exception(f"CatTensor failed, as it expected input keys = {sorted(list(self.keys))} but got a "
                            f"TensorDict with keys {sorted(list(tensor_dict.keys()))}")
        return tensor_dict

    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        if not isinstance(observation_spec, CompositeSpec):
            raise TypeError("observation_spec is not of type CompositeSpec")
        keys = [key.split("observation_")[-1] for key in self.keys]

        if all([key in observation_spec for key in keys]):
            sum_shape = sum([
                observation_spec[key].shape[-1] if observation_spec[key].shape else 1
                for key in keys])
            spec0 = observation_spec[keys[0]]
            out_key = self.out_key.split("observation_")[-1]
            observation_spec[out_key] = NdUnboundedContinuousTensorSpec(
                shape=torch.Size([*spec0.shape[:-1], sum_shape]),
                dtype=spec0.dtype)
            for key in keys:
                observation_spec.del_(key)
        return observation_spec


class DiscreteActionProjection(Transform):
    """
    Given a discrete action (from 1 to N) encoded as a one-hot vector and a maximum action index M (with M < N),
    transforms the action such that action_out is at most M.
    If the input action is > M, it is being replaced by a random value between N and M.
    Otherwise the same action is kept.
    This is intended to be used with policies applied over multiple discrete control environments with different action
    space.

    Args:
        max_N (int): max number of action considered
        M (int): resulting number of actions
    """
    inplace = False

    def __init__(self, max_N: int, M: int, action_key: str = "action"):
        super().__init__([action_key])
        self.max_N = max_N
        self.M = M

    def _inv_apply(self, action: torch.Tensor) -> torch.Tensor:
        if action.shape[-1] < self.M:
            raise RuntimeError(f"action.shape[-1]={action.shape[-1]} is smaller than "
                               f"DiscreteActionProjection.M={self.M}")
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


class NoopResetEnv(Transform):
    """
    Runs a series of random actions when an environment is reset.

    Args:
        env (_EnvClass): env on which the random actions have to be performed. Can be the same env as the one provided
            to the TransformedEnv class
        noops (int): number of actions performed after reset
        random (bool): if False, the number of random ops will always be equal to the noops value. If True, the number
            of random actions will be randomly selected between 0 and noops. Default=True.

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

    def reset(self, tensor_dict: _TensorDict) -> _TensorDict:
        """ Do no-op action for a number of steps in [1, noop_max]."""
        noops = self.noops if not self.random else torch.randint(self.noops, (1,)).item()
        for _ in range(noops):
            tensor_dict = self.env.rand_step()

        return step_tensor_dict(tensor_dict)

class PinMemoryTransform(Transform):
    """
    Calls pin_memory on the tensordict to facilitate writing on CUDA devices.

    """
    def __init__(self):
        super().__init__([])

    def _call(self, tensor_dict: _TensorDict) -> _TensorDict:
        return tensor_dict.pin_memory()