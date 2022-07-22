# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import abc
from collections import OrderedDict
from numbers import Number
from typing import Any, Callable, Iterator, Optional, Union, Dict

import numpy as np
import torch

from torchrl import seed_generator, prod
from torchrl.data import CompositeSpec, TensorDict, TensorSpec
from ..data.tensordict.tensordict import _TensorDict
from ..data.utils import DEVICE_TYPING
from .utils import get_available_libraries, step_tensordict

LIBRARIES = get_available_libraries()


def _tensor_to_np(t):
    return t.detach().cpu().numpy()


dtype_map = {
    torch.float: np.float32,
    torch.double: np.float64,
    torch.bool: bool,
}

__all__ = ["Specs", "make_tensordict"]


class Specs:
    """Container for action, observation and reward specs.

    This class allows one to create an environment, retrieve all of the specs
    in a single data container (and access them in one place) before erasing
    the environment from the workspace.

    Args:
        env (_EnvClass): environment from which the specs have to be read.

    """

    _keys = {
        "action_spec",
        "observation_spec",
        "reward_spec",
        "input_spec",
        "from_pixels",
    }

    def __init__(self, env: _EnvClass):
        self.env = env

    def __getitem__(self, item: str) -> Any:
        if item not in self._keys:
            raise KeyError(f"item must be one of {self._keys}")
        return getattr(self.env, item)

    def keys(self) -> dict:
        return self._keys

    def build_tensordict(
        self, next_observation: bool = True, log_prob: bool = False
    ) -> _TensorDict:
        """returns a TensorDict with empty tensors of the desired shape"""
        # build a tensordict from specs
        td = TensorDict({}, batch_size=torch.Size([]))
        action_placeholder = torch.zeros(
            self["action_spec"].shape, dtype=self["action_spec"].dtype
        )
        if not isinstance(self["observation_spec"], CompositeSpec):
            raise RuntimeError("observation_spec is expected to be of Composite type.")
        else:
            for i, (key, item) in enumerate(self["observation_spec"].items()):
                if not key.startswith("next_"):
                    raise RuntimeError(
                        f"All observation keys must start with the `'next_'` prefix. Found {key}"
                    )
                observation_placeholder = torch.zeros(item.shape, dtype=item.dtype)
                if next_observation:
                    td.set(key, observation_placeholder)
                td.set(
                    key[5:],
                    observation_placeholder.clone(),
                )

        reward_placeholder = torch.zeros(
            self["reward_spec"].shape, dtype=self["reward_spec"].dtype
        )
        done_placeholder = torch.zeros_like(reward_placeholder, dtype=torch.bool)

        td.set("action", action_placeholder)
        td.set("reward", reward_placeholder)

        if log_prob:
            td.set(
                "log_prob",
                torch.zeros_like(reward_placeholder, dtype=torch.float32),
            )  # we assume log_prob to be of type float32
        td.set("done", done_placeholder)
        return td


class _EnvBase(torch.nn.Module):
    """
    Abstract environment parent class for TorchRL.

    Properties:
        - observation_spec (CompositeSpec): sampling spec of the observations;
        - action_spec (TensorSpec): sampling spec of the actions;
        - input_spec (CompositeSpec): sampling spec of the actions and/or other inputs;
        - reward_spec (TensorSpec): sampling spec of the rewards;
        - batch_size (torch.Size): number of environments contained in the instance;
        - device (torch.device): device where the env input and output are expected to live
        - is_done (torch.Tensor): boolean value(s) indicating if the environment has reached a done state since the
            last reset

    Methods:
        step (_TensorDict -> _TensorDict): step in the environment
        reset (_TensorDict, optional -> _TensorDict): reset the environment
        set_seed (int -> int): sets the seed of the environment
        rand_step (_TensorDict, optional -> _TensorDict): random step given the action spec
        rollout (Callable, ... -> _TensorDict): executes a rollout in the environment with the given policy (or random
            steps if no policy is provided)

    """
    def __init__(
        self,
        device: DEVICE_TYPING = "cpu",
        dtype: Optional[Union[torch.dtype, np.dtype]] = None,
        batch_size: Optional[torch.Size] = None,
    ):
        super().__init__()
        if device is not None:
            self.device = torch.device(device)
        self._is_done = None
        self.dtype = dtype_map.get(dtype, dtype)
        if not hasattr(self, "is_closed"):
            self.is_closed = True
        if "_action_spec" not in self.__dir__():
            self._action_spec = None
        if "_input_spec" not in self.__dir__():
            self._input_spec = CompositeSpec(action=self._action_spec)
        if "_reward_spec" not in self.__dir__():
            self._reward_spec = None
        if "_observation_spec" not in self.__dir__():
            self._observation_spec = None
        if batch_size is not None:
            # we want an error to be raised if we pass batch_size but
            # it's already been set
            self.batch_size = batch_size
        elif ("batch_size" not in self.__dir__()) and (
            "batch_size" not in self.__class__.__dict__
        ):
            self.batch_size = torch.Size([])

    @classmethod
    def __new__(cls, *args, **kwargs):
        cls._inplace_update = True
        return super().__new__(cls)

    @property
    def action_spec(self) -> TensorSpec:
        return self._action_spec

    @action_spec.setter
    def action_spec(self, value: TensorSpec) -> None:
        self._action_spec = value

    @property
    def input_spec(self) -> TensorSpec:
        if self._input_spec is None:
            self._input_spec = CompositeSpec(action=self.action_spec)
        return self._input_spec

    @input_spec.setter
    def input_spec(self, value: TensorSpec) -> None:
        self._input_spec = value

    @property
    def reward_spec(self) -> TensorSpec:
        return self._reward_spec

    @reward_spec.setter
    def reward_spec(self, value: TensorSpec) -> None:
        self._reward_spec = value

    @property
    def observation_spec(self) -> TensorSpec:
        return self._observation_spec

    @observation_spec.setter
    def observation_spec(self, value: TensorSpec) -> None:
        self._observation_spec = value

    def step(self, tensordict: _TensorDict) -> _TensorDict:
        """Makes a step in the environment.
        Step accepts a single argument, tensordict, which usually carries an 'action' key which indicates the action
        to be taken.
        Step will call an out-place private method, _step, which is the method to be re-written by _EnvClass subclasses.

        Args:
            tensordict (_TensorDict): Tensordict containing the action to be taken.

        Returns:
            the input tensordict, modified in place with the resulting observations, done state and reward
            (+ others if needed).

        """

        # sanity check
        if tensordict.get("action").dtype is not self.action_spec.dtype:
            raise TypeError(
                f"expected action.dtype to be {self.action_spec.dtype} "
                f"but got {tensordict.get('action').dtype}"
            )

        tensordict.is_locked = True  # make sure _step does not modify the tensordict
        tensordict_out = self._step(tensordict)
        tensordict.is_locked = False

        if tensordict_out is tensordict:
            raise RuntimeError(
                "_EnvClass._step should return outplace changes to the input "
                "tensordict. Consider emptying the TensorDict first (e.g. tensordict.empty() or "
                "tensordict.select()) inside _step before writing new tensors onto this new instance."
            )
        self.is_done = tensordict_out.get("done")

        for key in self._select_observation_keys(tensordict_out):
            obs = tensordict_out.get(key)
            self.observation_spec.type_check(obs, key)

        if tensordict_out._get_meta("reward").dtype is not self.reward_spec.dtype:
            raise TypeError(
                f"expected reward.dtype to be {self.reward_spec.dtype} "
                f"but got {tensordict_out.get('reward').dtype}"
            )

        if tensordict_out._get_meta("done").dtype is not torch.bool:
            raise TypeError(
                f"expected done.dtype to be torch.bool but got {tensordict_out.get('done').dtype}"
            )
        tensordict.update(tensordict_out, inplace=self._inplace_update)

        del tensordict_out
        return tensordict
    def forward(self, tensordict: _TensorDict) -> _TensorDict:
        return self.step(tensordict)

    def _step(
        self,
        tensordict: _TensorDict,
    ) -> _TensorDict:
        raise NotImplementedError

    def _reset(self, tensordict: _TensorDict, **kwargs) -> _TensorDict:
        raise NotImplementedError

    def reset(
        self,
        tensordict: Optional[_TensorDict] = None,
        execute_step: bool = True,
        **kwargs,
    ) -> _TensorDict:
        raise NotImplementedError

    def numel(self) -> int:
        return prod(self.batch_size)

    def set_seed(self, seed: int) -> int:
        raise NotImplementedError

    def _set_seed(self, seed: Optional[int]):
        raise NotImplementedError

    def set_state(self):
        raise NotImplementedError

    def _assert_tensordict_shape(self, tensordict: _TensorDict) -> None:
        if tensordict.batch_size != self.batch_size:
            raise RuntimeError(
                f"Expected a tensordict with shape==env.shape, "
                f"got {tensordict.batch_size} and {self.batch_size}"
            )

    def is_done_get_fn(self) -> bool:
        if self._is_done is None:
            self._is_done = torch.zeros(self.batch_size, device=self.device)
        return self._is_done.all()

    def is_done_set_fn(self, val: torch.Tensor) -> None:
        self._is_done = val

    is_done = property(is_done_get_fn, is_done_set_fn)

    def rand_step(self, tensordict: Optional[_TensorDict] = None) -> _TensorDict:
        """Performs a random step in the environment given the action_spec attribute.

        Args:
            tensordict (_TensorDict, optional): tensordict where the resulting info should be written.

        Returns:
            a tensordict object with the new observation after a random step in the environment. The action will
            be stored with the "action" key.

        """
        if tensordict is None:
            tensordict = TensorDict({}, device=self.device, batch_size=self.batch_size)
        action = self.action_spec.rand(self.batch_size)
        tensordict.set("action", action)
        return self.step(tensordict)

    @property
    def specs(self) -> Specs:
        """

        Returns a Specs container where all the environment specs are contained.
        This feature allows one to create an environment, retrieve all of the specs in a single data container and then
        erase the environment from the workspace.

        """
        return Specs(self)

    def rollout(
        self,
        max_steps: int,
        policy: Optional[Callable[[_TensorDict], _TensorDict]] = None,
        callback: Optional[Callable[[_TensorDict, ...], _TensorDict]] = None,
        auto_reset: bool = True,
        auto_cast_to_device: bool = False,
        break_when_any_done: bool = True,
        tensordict: Optional[_TensorDict] = None,
    ) -> _TensorDict:
        """Executes a rollout in the environment.

        The function will stop as soon as one of the contained environments
        returns done=True.

        Args:
            max_steps (int): maximum number of steps to be executed. The actual number of steps can be smaller if
                the environment reaches a done state before max_steps have been executed.
            policy (callable, optional): callable to be called to compute the desired action. If no policy is provided,
                actions will be called using `env.rand_step()`
                default = None
            callback (callable, optional): function to be called at each iteration with the given TensorDict.
            auto_reset (bool, optional): if True, resets automatically the environment
                if it is in a done state when the rollout is initiated.
                Default is `True`.
            auto_cast_to_device (bool, optional): if True, the device of the tensordict is automatically cast to the
                policy device before the policy is used. Default is `False`.
            break_when_any_done (bool): breaks if any of the done state is True. Default is True.
            tensordict (TensorDict, optional): if auto_reset is False, an initial
                tensordict must be provided.

        Returns:
            TensorDict object containing the resulting trajectory.

        """
        try:
            policy_device = next(policy.parameters()).device
        except AttributeError:
            policy_device = "cpu"

        env_device = self.device

        if auto_reset:
            if tensordict is not None:
                raise RuntimeError(
                    "tensordict cannot be provided when auto_reset is True"
                )
            tensordict = self.reset()
        elif tensordict is None:
            raise RuntimeError("tensordict must be provided when auto_reset is False")

        if policy is None:

            def policy(td):
                return td.set("action", self.action_spec.rand(self.batch_size))

        tensordicts = []
        if not self.is_done:
            for i in range(max_steps):
                if auto_cast_to_device:
                    tensordict = tensordict.to(policy_device)
                tensordict = policy(tensordict)
                if auto_cast_to_device:
                    tensordict = tensordict.to(env_device)
                tensordict = self.step(tensordict)
                tensordicts.append(tensordict.clone())
                if (
                    break_when_any_done and tensordict.get("done").any()
                ) or i == max_steps - 1:
                    break
                tensordict = step_tensordict(tensordict, keep_other=True)

                if callback is not None:
                    callback(self, tensordict)
        else:
            raise Exception("reset env before calling rollout!")

        out_td = torch.stack(tensordicts, len(self.batch_size))
        return out_td

    def _select_observation_keys(self, tensordict: _TensorDict) -> Iterator[str]:
        for key in tensordict.keys():
            if key.rfind("observation") >= 0:
                yield key

    def _to_tensor(
        self,
        value: Union[dict, bool, float, torch.Tensor, np.ndarray],
        device: Optional[DEVICE_TYPING] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Union[torch.Tensor, dict]:
        if device is None:
            device = self.device

        if isinstance(value, dict):
            return {
                _key: self._to_tensor(_value, dtype=dtype, device=device)
                for _key, _value in value.items()
            }
        elif isinstance(value, (bool, Number)):
            value = np.array(value)

        if dtype is None and self.dtype is not None:
            dtype = self.dtype
        elif dtype is not None:
            dtype = dtype_map.get(dtype, dtype)
        else:
            dtype = value.dtype

        if not isinstance(value, torch.Tensor):
            if dtype is not None:
                try:
                    value = value.astype(dtype)
                except TypeError:
                    raise Exception(
                        "dtype must be a numpy-compatible dtype. Got {dtype}"
                    )
            value = torch.as_tensor(value, device=device)
        else:
            value = value.to(device)
        # if dtype is not None:
        #     value = value.to(dtype)
        return value

    def close(self):
        self.is_closed = True

    def __del__(self):
        if not self.is_closed:
            self.close()

    def to(self, device: DEVICE_TYPING) -> _EnvClass:
        device = torch.device(device)
        if device == self.device:
            return self
        self.action_spec = self.action_spec.to(device)
        self.reward_spec = self.reward_spec.to(device)
        self.observation_spec = self.observation_spec.to(device)
        self.input_spec = self.input_spec.to(device)

        self.is_done = self.is_done.to(device)
        self.device = device
        return self

class _EnvClass(_EnvBase):
    """
    Abstract environment parent class for TorchRL.

    Properties:
        - observation_spec (CompositeSpec): sampling spec of the observations;
        - action_spec (TensorSpec): sampling spec of the actions;
        - input_spec (CompositeSpec): sampling spec of the actions and/or other inputs;
        - reward_spec (TensorSpec): sampling spec of the rewards;
        - batch_size (torch.Size): number of environments contained in the instance;
        - device (torch.device): device where the env input and output are expected to live
        - is_done (torch.Tensor): boolean value(s) indicating if the environment has reached a done state since the
            last reset

    Methods:
        step (_TensorDict -> _TensorDict): step in the environment
        reset (_TensorDict, optional -> _TensorDict): reset the environment
        set_seed (int -> int): sets the seed of the environment
        rand_step (_TensorDict, optional -> _TensorDict): random step given the action spec
        rollout (Callable, ... -> _TensorDict): executes a rollout in the environment with the given policy (or random
            steps if no policy is provided)

    """
    def __init__(
        self,
        device: DEVICE_TYPING = "cpu",
        dtype: Optional[Union[torch.dtype, np.dtype]] = None,
        batch_size: Optional[torch.Size] = None,
    ):
        super().__init__(device, dtype, batch_size)
        self.eval()
    
    def set_seed(self, seed: int) -> int:
        """Sets the seed of the environment and returns the next seed to be used (
        which is the input seed if a single environment is present)

        Args:
            seed: integer

        Returns:
            integer representing the "next seed": i.e. the seed that should be
            used for another environment if created concomittently to this environment.

        """
        if seed is not None:
            torch.manual_seed(seed)
        self._set_seed(seed)
        if seed is not None:
            new_seed = seed_generator(seed)
            seed = new_seed
        return seed
    def state_dict(self) -> OrderedDict:
        return OrderedDict()

    def load_state_dict(self, state_dict: OrderedDict, **kwargs) -> None:
        pass

    def eval(self) -> _EnvClass:
        return self

    def train(self, mode: bool = True) -> _EnvClass:
        return self
    
    def reset(
        self,
        tensordict: Optional[_TensorDict] = None,
        execute_step: bool = True,
        **kwargs,
    ) -> _TensorDict:
        """Resets the environment.
        As for step and _step, only the private method `_reset` should be overwritten by _EnvClass subclasses.

        Args:
            tensordict (_TensorDict, optional): tensordict to be used to contain the resulting new observation.
                In some cases, this input can also be used to pass argument to the reset function.
            execute_step (bool, optional): if True, a `step_tensordict` is executed on the output TensorDict,
                hereby removing the `"next_"` prefixes from the keys.
            kwargs (optional): other arguments to be passed to the native
                reset function.
        Returns:
            a tensordict (or the input tensordict, if any), modified in place with the resulting observations.

        """
        if tensordict is None:
            tensordict = TensorDict({}, device=self.device, batch_size=self.batch_size)
        tensordict_reset = self._reset(tensordict, **kwargs)
        if tensordict_reset is tensordict:
            raise RuntimeError(
                "_EnvClass._reset should return outplace changes to the input "
                "tensordict. Consider emptying the TensorDict first (e.g. tensordict.empty() or "
                "tensordict.select()) inside _reset before writing new tensors onto this new instance."
            )
        if not isinstance(tensordict_reset, _TensorDict):
            raise RuntimeError(
                f"env._reset returned an object of type {type(tensordict_reset)} but a TensorDict was expected."
            )

        self.is_done = tensordict_reset.get(
            "done",
            torch.zeros(self.batch_size, dtype=torch.bool, device=self.device),
        )
        if self.is_done:
            raise RuntimeError(
                f"Env {self} was done after reset. This is (currently) not allowed."
            )
        if execute_step:
            tensordict_reset = step_tensordict(
                tensordict_reset,
                exclude_done=False,
                exclude_reward=True,
                exclude_action=True,
            )
        if tensordict is not None:
            tensordict.update(tensordict_reset)
        else:
            tensordict = tensordict_reset
        return tensordict
    def to(self, device: DEVICE_TYPING) -> _EnvClass:
        device = torch.device(device)
        if device == self.device:
            return self
        self.action_spec = self.action_spec.to(device)
        self.reward_spec = self.reward_spec.to(device)
        self.observation_spec = self.observation_spec.to(device)
        self.input_spec = self.input_spec.to(device)

        self.is_done = self.is_done.to(device)
        self.device = device
        return self


class _EnvWrapper(_EnvClass, metaclass=abc.ABCMeta):
    """Abstract environment wrapper class.

    Unlike _EnvClass, _EnvWrapper comes with a `_build_env` private method that will be called upon instantiation.
    Interfaces with other libraries should be coded using _EnvWrapper.

    It is possible to directly query attributed from the nested environment it its name does not conflict with
    an attribute of the wrapper:
        >>> env = SomeWrapper(...)
        >>> custom_attribute0 = env._env.custom_attribute
        >>> custom_attribute1 = env.custom_attribute
        >>> assert custom_attribute0 is custom_attribute1  # should return True

    """

    git_url: str = ""
    available_envs: dict = {}
    libname: str = ""

    def __init__(
        self,
        *args,
        dtype: Optional[np.dtype] = None,
        device: DEVICE_TYPING = "cpu",
        batch_size: Optional[torch.Size] = None,
        **kwargs,
    ):
        super().__init__(
            device=device,
            dtype=dtype,
            batch_size=batch_size,
        )
        if len(args):
            raise ValueError(
                "`_EnvWrapper.__init__` received a non-empty args list of arguments."
                "Make sure only keywords arguments are used when calling `super().__init__`."
            )

        frame_skip = kwargs.get("frame_skip", 1)
        if "frame_skip" in kwargs:
            del kwargs["frame_skip"]
        self.frame_skip = frame_skip
        # this value can be changed if frame_skip is passed during env construction
        self.wrapper_frame_skip = frame_skip

        self._constructor_kwargs = kwargs
        self._check_kwargs(kwargs)
        self._env = self._build_env(**kwargs)  # writes the self._env attribute
        self._make_specs(self._env)  # writes the self._env attribute
        self.is_closed = False
        self._init_env()  # runs all the steps to have a ready-to-use env

    @abc.abstractmethod
    def _check_kwargs(self, kwargs: Dict):
        raise NotImplementedError

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

        elif "_env" in self.__dir__():
            env = self.__getattribute__("_env")
            return getattr(env, attr)

        raise AttributeError(
            f"env not set in {self.__class__.__name__}, cannot access {attr}"
        )

    def _init_env(self) -> Optional[int]:
        """Runs all the necessary steps such that the environment is ready to use.

        This step is intended to ensure that a seed is provided to the environment (if needed) and that the environment
        is reset (if needed). For instance, DMControl envs require the env to be reset before being used, but Gym envs
        don't.

        Returns:
            the resulting seed

        """
        raise NotImplementedError

    @abc.abstractmethod
    def _build_env(self, **kwargs) -> "gym.Env":
        """Creates an environment from the target library and stores it with the `_env` attribute.

        When overwritten, this function should pass all the required kwargs to the env instantiation method.

        """
        raise NotImplementedError

    @abc.abstractmethod
    def _make_specs(self, env: "gym.Env") -> None:
        raise NotImplementedError

    def close(self) -> None:
        """Closes the contained environment if possible."""
        self.is_closed = True
        try:
            self._env.close()
        except AttributeError:
            pass

    def set_seed(self, seed: Optional[int] = None) -> Optional[int]:
        if seed is not None:
            torch.manual_seed(seed)
        self._set_seed(seed)
        if seed is not None:
            new_seed = seed_generator(seed)
            seed = new_seed
        return seed

    @abc.abstractmethod
    def _set_seed(self, seed: Optional[int]):
        raise NotImplementedError


def make_tensordict(
    env: _EnvClass,
    policy: Optional[Callable[[_TensorDict, ...], _TensorDict]] = None,
) -> _TensorDict:
    """
    Returns a zeroed-tensordict with fields matching those required for a full step
    (action selection and environment step) in the environment

    Args:
        env (_EnvWrapper): environment defining the observation, action and reward space;
        policy (Callable, optional): policy corresponding to the environment.

    """
    with torch.no_grad():
        tensordict = env.reset()
        if policy is not None:
            tensordict = tensordict.unsqueeze(0)
            tensordict = policy(tensordict)
            tensordict = tensordict.squeeze(0)
        else:
            tensordict.set("action", env.action_spec.rand(), inplace=False)
        tensordict = env.step(tensordict)
        return tensordict.zero_()
