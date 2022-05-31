# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import functools
from collections import OrderedDict
from typing import Iterable, Optional, Union

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from torchrl.data.tensordict.tensordict import _TensorDict
from torchrl.envs.utils import step_tensordict
from torchrl.modules import TensorDictModule

__all__ = ["SoftUpdate", "HardUpdate", "distance_loss", "hold_out_params"]


class _context_manager:
    def __init__(self, value=True):
        self.value = value
        self.prev = []

    def __call__(self, func):
        @functools.wraps(func)
        def decorate_context(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return decorate_context


def distance_loss(
    v1: torch.Tensor,
    v2: torch.Tensor,
    loss_function: str,
    strict_shape: bool = True,
) -> torch.Tensor:
    """
    Computes a distance loss between two tensors.

    Args:
        v1 (Tensor): a tensor with a shape compatible with v2
        v2 (Tensor): a tensor with a shape compatible with v1
        loss_function (str): One of "l2", "l1" or "smooth_l1" representing which loss function is to be used.
        strict_shape (bool): if False, v1 and v2 are allowed to have a different shape.
            Default is `True`.

    Returns:
         A tensor of the shape v1.view_as(v2) or v2.view_as(v1) with values equal to the distance loss between the
        two.

    """
    if v1.shape != v2.shape and strict_shape:
        raise RuntimeError(
            f"The input tensors have shapes {v1.shape} and {v2.shape} which are incompatible."
        )

    if loss_function == "l2":
        value_loss = F.mse_loss(
            v1,
            v2,
            reduction="none",
        )

    elif loss_function == "l1":
        value_loss = F.l1_loss(
            v1,
            v2,
            reduction="none",
        )

    elif loss_function == "smooth_l1":
        value_loss = F.smooth_l1_loss(
            v1,
            v2,
            reduction="none",
        )
    else:
        raise NotImplementedError(f"Unknown loss {loss_function}")
    return value_loss


class ValueLoss:
    value_network: nn.Module
    target_value_network: nn.Module


class _TargetNetUpdate:
    """
    An abstract class for target network update in Double DQN/DDPG.

    Args:
        loss_module (DQNLoss or DDPGLoss): loss module where the target network should be updated.

    """

    def __init__(
        self,
        loss_module: Union["DQNLoss", "DDPGLoss", "SACLoss"],
    ):

        _target_names = []
        # for properties
        for name in loss_module.__class__.__dict__:
            if (
                name.startswith("_target_")
                and (name.endswith("params") or name.endswith("buffers"))
                and (getattr(loss_module, name) is not None)
            ):
                _target_names.append(name)

        # for regular lists: raise an exception
        for name in loss_module.__dict__:
            if (
                name.startswith("_target_")
                and (name.endswith("params") or name.endswith("buffers"))
                and (getattr(loss_module, name) is not None)
            ):
                raise RuntimeError(
                    "Your module seems to have a _target tensor list contained "
                    "in a non-dynamic structure (such as a list). If the "
                    "module is cast onto a device, the reference to these "
                    "tensors will be lost."
                )

        if len(_target_names) == 0:
            raise RuntimeError(
                "Did not found any target parameters or buffers in the loss module."
            )

        _source_names = ["".join(name.split("_target_")) for name in _target_names]

        for _source in _source_names:
            try:
                getattr(loss_module, _source)
            except AttributeError:
                raise RuntimeError(
                    f"Incongruent target and source parameter lists: "
                    f"{_source} is not an attribute of the loss_module"
                )

        self._target_names = _target_names
        self._source_names = _source_names
        self.loss_module = loss_module
        self.initialized = False

    @property
    def _targets(self):
        return OrderedDict(
            {name: getattr(self.loss_module, name) for name in self._target_names}
        )

    @property
    def _sources(self):
        return OrderedDict(
            {name: getattr(self.loss_module, name) for name in self._source_names}
        )

    def init_(self) -> None:
        for source, target in zip(self._sources.values(), self._targets.values()):
            for p_source, p_target in zip(source, target):
                if p_target.requires_grad:
                    raise RuntimeError("the target parameter is part of a graph.")
                p_target.data.copy_(p_source.data)
        self.initialized = True

    def step(self) -> None:
        if not self.initialized:
            raise Exception(
                f"{self.__class__.__name__} must be "
                f"initialized (`{self.__class__.__name__}.init_()`) before calling step()"
            )

        for source, target in zip(self._sources.values(), self._targets.values()):
            for p_source, p_target in zip(source, target):
                if p_target.requires_grad:
                    raise RuntimeError("the target parameter is part of a graph.")
                self._step(p_source, p_target)

    def _step(self, p_source: Tensor, p_target: Tensor) -> None:
        raise NotImplementedError

    def __repr__(self) -> str:
        string = (
            f"{self.__class__.__name__}(sources={[name for name in self._sources]}, targets="
            f"{[name for name in self._targets]})"
        )
        return string


class SoftUpdate(_TargetNetUpdate):
    """
    A soft-update class for target network update in Double DQN/DDPG.
    This was proposed in "CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING", https://arxiv.org/pdf/1509.02971.pdf

    Args:
        loss_module (DQNLoss or DDPGLoss): loss module where the target network should be updated.
        eps (scalar): epsilon in the update equation:
            param = prev_param * eps + new_param * (1-eps)
            default: 0.999
    """

    def __init__(
        self,
        loss_module: Union["DQNLoss", "DDPGLoss", "SACLoss", "REDQLoss"],
        eps: float = 0.999,
    ):
        if not (eps < 1.0 and eps > 0.0):
            raise ValueError(
                f"Got eps = {eps} when it was supposed to be between 0 and 1."
            )
        super(SoftUpdate, self).__init__(loss_module)
        self.eps = eps

    def _step(self, p_source: Tensor, p_target: Tensor) -> None:
        p_target.data.copy_(p_target.data * self.eps + p_source.data * (1 - self.eps))


class HardUpdate(_TargetNetUpdate):
    """
    A hard-update class for target network update in Double DQN/DDPG (by contrast with soft updates).
    This was proposed in the original Double DQN paper: "Deep Reinforcement Learning with Double Q-learning",
    https://arxiv.org/abs/1509.06461.

    Args:
        loss_module (DQNLoss or DDPGLoss): loss module where the target network should be updated.
        value_network_update_interval (scalar): how often the target network should be updated.
            default: 1000
    """

    def __init__(
        self,
        loss_module: Union["DQNLoss", "DDPGLoss", "SACLoss"],
        value_network_update_interval: float = 1000,
    ):
        super(HardUpdate, self).__init__(loss_module)
        self.value_network_update_interval = value_network_update_interval
        self.counter = 0

    def _step(self, p_source: Tensor, p_target: Tensor) -> None:
        if self.counter == self.value_network_update_interval:
            p_target.data.copy_(p_source.data)

    def step(self) -> None:
        super().step()
        if self.counter == self.value_network_update_interval:
            self.counter = 0
        else:
            self.counter += 1


class hold_out_net(_context_manager):
    """Context manager to hold a network out of a computational graph."""

    def __init__(self, network: nn.Module) -> None:
        self.network = network
        try:
            self.p_example = next(network.parameters())
        except StopIteration:
            raise RuntimeError(
                "hold_out_net requires the network parameter set to be " "non-empty."
            )
        self._prev_state = []

    def __enter__(self) -> None:
        self._prev_state.append(self.p_example.requires_grad)
        self.network.requires_grad_(False)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.network.requires_grad_(self._prev_state.pop())


class hold_out_params(_context_manager):
    """Context manager to hold a list of parameters out of a computational graph."""

    def __init__(self, params: Iterable[Tensor]) -> None:
        self.params = tuple(p.detach() for p in params)

    def __enter__(self) -> None:
        return self.params

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass


@torch.no_grad()
def next_state_value(
    tensordict: _TensorDict,
    operator: Optional[TensorDictModule] = None,
    next_val_key: str = "state_action_value",
    gamma: float = 0.99,
    pred_next_val: Optional[Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    """
    Computes the next state value (without gradient) to compute a target for the MSE loss
        L = Sum[ (q_value - target_value)^2 ]
    The target value is computed as
        r + gamma ** n_steps_to_next * value_next_state
    If the reward is the immediate reward, n_steps_to_next=1. If N-steps rewards are used, n_steps_to_next is gathered
    from the input tensordict.

    Args:
        tensordict (_TensorDict): Tensordict containing a reward and done key (and a n_steps_to_next key for n-steps
            rewards).
        operator (ProbabilisticTDModule, optional): the value function operator. Should write a 'next_val_key'
            key-value in the input tensordict when called. It does not need to be provided if pred_next_val is given.
        next_val_key (str, optional): key where the next value will be written.
            Default: 'state_action_value'
        gamma (float, optional): return discount rate.
            default: 0.99
        pred_next_val (Tensor, optional): the next state value can be provided if it is not computed with the operator.

    Returns:
        a Tensor of the size of the input tensordict containing the predicted value state.
    """
    try:
        steps_to_next_obs = tensordict.get("steps_to_next_obs").squeeze(-1)
    except KeyError:
        steps_to_next_obs = 1

    rewards = tensordict.get("reward").squeeze(-1)
    done = tensordict.get("done").squeeze(-1)

    if pred_next_val is None:
        next_td = step_tensordict(tensordict)  # next_observation -> observation
        next_td = next_td.select(*operator.in_keys)
        operator(next_td, **kwargs)
        pred_next_val_detach = next_td.get(next_val_key).squeeze(-1)
    else:
        pred_next_val_detach = pred_next_val.squeeze(-1)
    done = done.to(torch.float)
    target_value = (1 - done) * pred_next_val_detach
    rewards = rewards.to(torch.float)
    target_value = rewards + (gamma ** steps_to_next_obs) * target_value
    return target_value
