# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union, Optional, List

import torch

# for value, log_policy, reward, entropy in list(zip(values, log_policies, rewards, entropies))[::-1]:
#     gae = gae * opt.gamma * opt.tau
#     gae = gae + reward + opt.gamma * next_value.detach() - value.detach()
#     next_value = value
#     actor_loss = actor_loss + log_policy * gae
#     R = R * opt.gamma + reward
#     critic_loss = critic_loss + (R - value) ** 2 / 2
#     entropy_loss = entropy_loss + entropy
from torch import Tensor

from torchrl.data.tensordict.tensordict import _TensorDict
from torchrl.envs.utils import step_tensordict
from torchrl.modules import TensorDictModule
from torchrl.objectives.value.functional import (
    generalized_advantage_estimate,
    td_lambda_advantage_estimate,
    vec_td_lambda_advantage_estimate,
)
from .functional import td_advantage_estimate

__all__ = ["GAE", "TDLambdaEstimate", "TDEstimate"]


class TDEstimate:
    """Temporal Difference estimate of advantage function.

    Args:
        gamma (scalar): exponential mean discount.
        value_network (TensorDictModule): value operator used to retrieve the value estimates.
        average_rewards (bool, optional): if True, rewards will be standardized
            before the TD is computed.
        gradient_mode (bool, optional): if True, gradients are propagated throught
            the computation of the value function. Default is `False`.
        value_key (str, optional): key pointing to the state value. Default is
            `"state_value"`.
    """

    def __init__(
        self,
        gamma: Union[float, torch.Tensor],
        value_network: TensorDictModule,
        average_rewards: bool = False,
        gradient_mode: bool = False,
        value_key: str = "state_value",
    ):
        self.gamma = gamma
        self.value_network = value_network
        self.is_functional = value_network.is_functional

        self.average_rewards = average_rewards
        self.gradient_mode = gradient_mode
        self.value_key = value_key

    def __call__(
        self,
        tensordict: _TensorDict,
        *unused_args,
        params: Optional[List[Tensor]] = None,
        buffers: Optional[List[Tensor]] = None,
        target_params: Optional[List[Tensor]] = None,
        target_buffers: Optional[List[Tensor]] = None,
    ) -> _TensorDict:
        """Computes the GAE given the data in tensordict.

        Args:
            tensordict (_TensorDict): A TensorDict containing the data (observation, action, reward, done state)
                necessary to compute the value estimates and the GAE.

        Returns:
            An updated TensorDict with an "advantage" and a "value_error" keys

        """
        with torch.set_grad_enabled(self.gradient_mode):
            if tensordict.batch_dims < 1:
                raise RuntimeError(
                    "Expected input tensordict to have at least one dimensions, got"
                    f"tensordict.batch_size = {tensordict.batch_size}"
                )
            reward = tensordict.get("reward")
            if self.average_rewards:
                reward = reward - reward.mean()
                reward = reward / reward.std().clamp_min(1e-4)
                tensordict.set_(
                    "reward", reward
                )  # we must update the rewards if they are used later in the code

            gamma = self.gamma
            kwargs = {}
            if self.is_functional and params is None:
                raise RuntimeError(
                    "Expected params to be passed to advantage module but got none."
                )
            if params is not None:
                kwargs["params"] = params
            if buffers is not None:
                kwargs["buffers"] = buffers
            self.value_network(tensordict, **kwargs)
            value = tensordict.get(self.value_key)

        with torch.set_grad_enabled(False):
            step_td = step_tensordict(tensordict)
            if target_params is not None:
                kwargs["params"] = target_params
            if target_buffers is not None:
                kwargs["buffers"] = target_buffers
            self.value_network(step_td, **kwargs)
            next_value = step_td.get(self.value_key)

        done = tensordict.get("done")
        with torch.set_grad_enabled(self.gradient_mode):
            adv = td_advantage_estimate(gamma, value, next_value, reward, done)
            tensordict.set("advantage", adv.detach())
            if self.gradient_mode:
                tensordict.set("value_error", adv)
        return tensordict


class TDLambdaEstimate:
    """TD-Lambda estimate of advantage function.

    Args:
        gamma (scalar): exponential mean discount.
        lmbda (scalar): trajectory discount.
        value_network (TensorDictModule): value operator used to retrieve the value estimates.
        average_rewards (bool, optional): if True, rewards will be standardized
            before the TD is computed.
        gradient_mode (bool, optional): if True, gradients are propagated throught
            the computation of the value function. Default is `False`.
        value_key (str, optional): key pointing to the state value. Default is
            `"state_value"`.
        vectorized (bool, optional): whether to use the vectorized version of the
            lambda return. Default is `True`.
    """

    def __init__(
        self,
        gamma: Union[float, torch.Tensor],
        lmbda: Union[float, torch.Tensor],
        value_network: TensorDictModule,
        average_rewards: bool = False,
        gradient_mode: bool = False,
        value_key: str = "state_value",
        vectorized: bool = True,
    ):
        self.gamma = gamma
        self.lmbda = lmbda
        self.value_network = value_network
        self.is_functional = value_network.is_functional
        self.vectorized = vectorized

        self.average_rewards = average_rewards
        self.gradient_mode = gradient_mode
        self.value_key = value_key

    def __call__(
        self,
        tensordict: _TensorDict,
        *unused_args,
        params: Optional[List[Tensor]] = None,
        buffers: Optional[List[Tensor]] = None,
        target_params: Optional[List[Tensor]] = None,
        target_buffers: Optional[List[Tensor]] = None,
    ) -> _TensorDict:
        """Computes the GAE given the data in tensordict.

        Args:
            tensordict (_TensorDict): A TensorDict containing the data (observation, action, reward, done state)
                necessary to compute the value estimates and the GAE.

        Returns:
            An updated TensorDict with an "advantage" and a "value_error" keys

        """
        with torch.set_grad_enabled(self.gradient_mode):
            if tensordict.batch_dims < 1:
                raise RuntimeError(
                    "Expected input tensordict to have at least one dimensions, got"
                    f"tensordict.batch_size = {tensordict.batch_size}"
                )
            reward = tensordict.get("reward")
            if self.average_rewards:
                reward = reward - reward.mean()
                reward = reward / reward.std().clamp_min(1e-4)
                tensordict.set_(
                    "reward", reward
                )  # we must update the rewards if they are used later in the code

            gamma = self.gamma
            lmbda = self.lmbda

            kwargs = {}
            if self.is_functional and params is None:
                raise RuntimeError(
                    "Expected params to be passed to advantage module but got none."
                )
            if params is not None:
                kwargs["params"] = params
            if buffers is not None:
                kwargs["buffers"] = buffers
            self.value_network(tensordict, **kwargs)
            value = tensordict.get(self.value_key)

        with torch.set_grad_enabled(False):
            step_td = step_tensordict(tensordict)
            if target_params is not None:
                kwargs["params"] = target_params
            if target_buffers is not None:
                kwargs["buffers"] = target_buffers
            self.value_network(step_td, **kwargs)
            next_value = step_td.get(self.value_key)

        done = tensordict.get("done")
        with torch.set_grad_enabled(self.gradient_mode):
            if self.vectorized:
                adv = vec_td_lambda_advantage_estimate(
                    gamma, lmbda, value, next_value, reward, done
                )
            else:
                adv = td_lambda_advantage_estimate(
                    gamma, lmbda, value, next_value, reward, done
                )

            tensordict.set("advantage", adv.detach())
            if self.gradient_mode:
                tensordict.set("value_error", adv)
        return tensordict


class GAE:
    """
    A class wrapper around the generalized advantage estimate functional.
    Refer to "HIGH-DIMENSIONAL CONTINUOUS CONTROL USING GENERALIZED ADVANTAGE ESTIMATION"
    https://arxiv.org/pdf/1506.02438.pdf for more context.

    Args:
        gamma (scalar): exponential mean discount.
        lmbda (scalar): trajectory discount.
        value_network (TensorDictModule): value operator used to retrieve the value estimates.
        average_rewards (bool): if True, rewards will be standardized before the GAE is computed.
        gradient_mode (bool): if True, gradients are propagated throught the computation of the value function.
            Default is `False`.
    """

    def __init__(
        self,
        gamma: Union[float, torch.Tensor],
        lmbda: float,
        value_network: TensorDictModule,
        average_rewards: bool = False,
        gradient_mode: bool = False,
    ):
        self.gamma = gamma
        self.lmbda = lmbda
        self.value_network = value_network
        self.is_functional = value_network.is_functional

        self.average_rewards = average_rewards
        self.gradient_mode = gradient_mode

    def __call__(
        self,
        tensordict: _TensorDict,
        *unused_args,
        params: Optional[List[Tensor]] = None,
        buffers: Optional[List[Tensor]] = None,
        target_params: Optional[List[Tensor]] = None,
        target_buffers: Optional[List[Tensor]] = None,
    ) -> _TensorDict:
        """Computes the GAE given the data in tensordict.

        Args:
            tensordict (_TensorDict): A TensorDict containing the data (observation, action, reward, done state)
                necessary to compute the value estimates and the GAE.

        Returns:
            An updated TensorDict with an "advantage" and a "value_error" keys

        """
        with torch.set_grad_enabled(self.gradient_mode):
            if tensordict.batch_dims < 1:
                raise RuntimeError(
                    "Expected input tensordict to have at least one dimensions, got"
                    f"tensordict.batch_size = {tensordict.batch_size}"
                )
            reward = tensordict.get("reward")
            if self.average_rewards:
                reward = reward - reward.mean()
                reward = reward / reward.std().clamp_min(1e-4)
                tensordict.set_(
                    "reward", reward
                )  # we must update the rewards if they are used later in the code

            gamma, lmbda = self.gamma, self.lmbda
            kwargs = {}
            if self.is_functional and params is None:
                raise RuntimeError(
                    "Expected params to be passed to advantage module but got none."
                )
            if params is not None:
                kwargs["params"] = params
            if buffers is not None:
                kwargs["buffers"] = buffers
            self.value_network(tensordict, **kwargs)
            value = tensordict.get("state_value")

        with torch.set_grad_enabled(False):
            step_td = step_tensordict(tensordict)
            if target_params is not None:
                kwargs["params"] = target_params
            if target_buffers is not None:
                kwargs["buffers"] = target_buffers
            self.value_network(step_td, **kwargs)
            next_value = step_td.get("state_value")
            done = tensordict.get("done")
            adv, value_target = generalized_advantage_estimate(
                gamma, lmbda, value, next_value, reward, done
            )

        tensordict.set("advantage", adv.detach())
        if self.gradient_mode:
            tensordict.set("value_error", value_target - value)

        return tensordict
