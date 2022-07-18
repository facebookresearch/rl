# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Tuple

import torch

from torchrl.data.tensordict.tensordict import _TensorDict, TensorDict
from torchrl.envs.utils import set_exploration_mode, step_tensordict
from torchrl.modules import TensorDictModule
from torchrl.modules.tensordict_module.actors import ActorCriticWrapper
from torchrl.objectives.costs.utils import (
    distance_loss,
    hold_out_params,
    next_state_value,
    zip_stack,
)
from .common import LossModule

__all__ = ["DDPGLoss", "vecDDPGLoss", "vecDDPGLossGrad"]

import functorch


class DDPGLoss(LossModule):
    """
    The DDPG Loss class.
    Args:
        actor_network (TensorDictModule): a policy operator.
        value_network (TensorDictModule): a Q value operator.
        gamma (scalar): a discount factor for return computation.
        device (str, int or torch.device, optional): a device where the losses will be computed, if it can't be found
            via the value operator.
        loss_function (str): loss function for the value discrepancy. Can be one of "l1", "l2" or "smooth_l1".
        delay_actor (bool, optional): whether to separate the target actor networks from the actor networks used for
            data collection. Default is `False`.
        delay_value (bool, optional): whether to separate the target value networks from the value networks used for
            data collection. Default is `False`.
    """

    def __init__(
        self,
        actor_network: TensorDictModule,
        value_network: TensorDictModule,
        gamma: float,
        loss_function: str = "l2",
        delay_actor: bool = False,
        delay_value: bool = False,
        update_actor_frequency: int = 1,
    ) -> None:
        super().__init__()
        self.delay_actor = delay_actor
        self.delay_value = delay_value
        self.convert_to_functional(
            actor_network,
            "actor_network",
            create_target_params=self.delay_actor,
        )
        self.convert_to_functional(
            value_network,
            "value_network",
            create_target_params=self.delay_value,
            compare_against=list(actor_network.parameters()),
        )
        self.actor_critic = ActorCriticWrapper(self.actor_network, self.value_network)

        self.actor_in_keys = actor_network.in_keys

        self.gamma = gamma
        self.loss_funtion = loss_function
        self.update_actor_frequency = update_actor_frequency
        self._counter = 0

    def forward(self, input_tensordict: _TensorDict) -> TensorDict:
        """Computes the DDPG losses given a tensordict sampled from the replay buffer.
        This function will also write a "td_error" key that can be used by prioritized replay buffers to assign
            a priority to items in the tensordict.

        Args:
            input_tensordict (_TensorDict): a tensordict with keys ["done", "reward"] and the in_keys of the actor
                and value networks.

        Returns:
            a tuple of 2 tensors containing the DDPG loss.

        """
        if not input_tensordict.device == self.device:
            raise RuntimeError(
                f"Got device={input_tensordict.device} but actor_network.device={self.device} "
                f"(self.device={self.device})"
            )
        self._counter += 1

        loss_value, td_error, pred_val, target_value = self._loss_value(
            input_tensordict,
        )
        td_error = td_error.detach()
        td_error = td_error.unsqueeze(input_tensordict.ndimension())
        td_error = td_error.to(input_tensordict.device)
        input_tensordict.set(
            "td_error",
            td_error,
            inplace=True,
        )
        if self._counter % self.update_actor_frequency == 0:
            loss_actor = self._loss_actor(input_tensordict)
        else:
            loss_actor = torch.zeros_like(loss_value)

        return TensorDict(
            source={
                "loss_actor": loss_actor.mean(),
                "loss_value": loss_value.mean(),
                "pred_value": pred_val.mean().detach(),
                "target_value": target_value.mean().detach(),
                "pred_value_max": pred_val.max().detach(),
                "target_value_max": target_value.max().detach(),
            },
            batch_size=[],
        )

    def _loss_actor(
        self,
        tensordict: _TensorDict,
    ) -> torch.Tensor:
        td_copy = tensordict.select(*self.actor_in_keys).detach()
        td_copy = self.actor_network(
            td_copy,
            params=self.actor_network_params,
            buffers=self.actor_network_buffers,
        )
        with hold_out_params(self.value_network_params) as params:
            td_copy = self.value_network(
                td_copy, params=params, buffers=self.value_network_buffers
            )
        return -td_copy.get("state_action_value")

    def _loss_value(
        self,
        tensordict: _TensorDict,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # value loss
        td_copy = tensordict.select(*self.value_network.in_keys).detach()
        self.value_network(
            td_copy,
            params=self.value_network_params,
            buffers=self.value_network_buffers,
        )
        pred_val = td_copy.get("state_action_value").squeeze(-1)

        target_params = list(self.target_actor_network_params) + list(
            self.target_value_network_params
        )
        target_buffers = list(self.target_actor_network_buffers) + list(
            self.target_value_network_buffers
        )
        with set_exploration_mode("mode"):
            target_value = next_state_value(
                tensordict,
                self.actor_critic,
                gamma=self.gamma,
                params=target_params,
                buffers=target_buffers,
            )

        # td_error = pred_val - target_value
        loss_value = distance_loss(
            pred_val, target_value, loss_function=self.loss_funtion
        )

        return loss_value, abs(pred_val - target_value), pred_val, target_value


class vecDDPGLoss(LossModule):
    """
    The vectorized DDPG Loss class.
    """

    def __init__(
        self,
        actor_network: TensorDictModule,
        value_network: TensorDictModule,
        gamma: float,
        loss_function: str = "l2",
        delay_actor: bool = False,
        delay_value: bool = False,
    ) -> None:
        super().__init__()
        self.delay_actor = delay_actor
        self.delay_value = delay_value
        self.convert_to_functional(
            actor_network,
            "actor_network",
            create_target_params=self.delay_actor,
        )
        self.convert_to_functional(
            value_network,
            "value_network",
            create_target_params=self.delay_value,
            compare_against=list(actor_network.parameters()),
        )
        self.actor_critic = ActorCriticWrapper(self.actor_network, self.value_network)

        self.actor_in_keys = actor_network.in_keys

        self.gamma = gamma
        self.loss_funtion = loss_function

    def forward(self, input_tensordict: _TensorDict) -> TensorDict:
        """Computes the DDPG losses given a tensordict sampled from the replay buffer.
        This function will also write a "td_error" key that can be used by prioritized replay buffers to assign
            a priority to items in the tensordict.

        Args:
            input_tensordict (_TensorDict): a tensordict with keys ["done", "reward"] and the in_keys of the actor
                and value networks.

        Returns:
            a tuple of 2 tensors containing the DDPG loss.

        """
        if not input_tensordict.device == self.device:
            raise RuntimeError(
                f"Got device={input_tensordict.device} but actor_network.device={self.device} "
                f"(self.device={self.device})"
            )

        with set_exploration_mode("mode"), hold_out_params(
            self.value_network_params
        ) as value_net_actor_loss_params:
            actor_loss_params_an = self.actor_network_params
            actor_loss_params_vn = value_net_actor_loss_params
            actor_loss_buffers_an = self.actor_network_buffers
            actor_loss_buffers_vn = self.value_network_buffers

            value_loss_target_params_an = self.target_actor_network_params
            value_loss_target_params_vn = self.target_value_network_params
            value_loss_target_buffers_an = self.target_actor_network_buffers
            value_loss_target_buffers_vn = self.target_value_network_buffers

            value_loss_params_vn = self.value_network_params
            value_loss_buffers_vn = self.value_network_buffers

            tensordict = torch.stack(
                [
                    input_tensordict,
                    step_tensordict(input_tensordict),
                ],
                0,
            ).to_tensordict()

            actor_params = zip_stack(
                (actor_loss_params_an, value_loss_target_params_an)
            )
            actor_buffers = zip_stack(
                (actor_loss_buffers_an, value_loss_target_buffers_an)
            )

            value_params = zip_stack(
                (
                    actor_loss_params_vn,
                    value_loss_target_params_vn,
                    value_loss_params_vn,
                )
            )
            value_buffers = zip_stack(
                (
                    actor_loss_buffers_vn,
                    value_loss_target_buffers_vn,
                    value_loss_buffers_vn,
                )
            )

            tensordict = self.actor_network(
                tensordict, params=actor_params, buffers=actor_buffers, vmap=(0, 0, 0)
            )
            tensordict = torch.cat(
                [
                    tensordict.select(*self.value_network.in_keys),
                    input_tensordict.select(*self.value_network.in_keys).unsqueeze(0),
                ],
                0,
            )
            tensordict = self.value_network(
                tensordict, params=value_params, buffers=value_buffers, vmap=(0, 0, 0)
            )
            loss_actor, next_state_action_value, pred_val = tensordict.get(
                "state_action_value"
            ).squeeze(-1)
            loss_actor = -loss_actor

        target_value = next_state_value(
            input_tensordict,
            gamma=self.gamma,
            pred_next_val=next_state_action_value,
        )

        loss_value = distance_loss(
            pred_val, target_value, loss_function=self.loss_funtion
        )

        td_error = abs(pred_val - target_value)

        td_error = td_error.detach()
        td_error = td_error.unsqueeze(input_tensordict.ndimension())
        td_error = td_error.to(input_tensordict.device)
        input_tensordict.set(
            "td_error",
            td_error,
            inplace=True,
        )
        return TensorDict(
            source={
                "loss_actor": loss_actor.mean(),
                "loss_value": loss_value.mean(),
                "pred_value": pred_val.mean().detach(),
                "target_value": target_value.mean().detach(),
                "pred_value_max": pred_val.max().detach(),
                "target_value_max": target_value.max().detach(),
            },
            batch_size=[],
            device=self.device,
            _run_checks=False,
        )


class vecDDPGLossGrad(LossModule):
    """
    The DDPG Loss class, with internal gradient computation
    """

    def __init__(
        self,
        actor_network: TensorDictModule,
        value_network: TensorDictModule,
        gamma: float,
        loss_function: str = "l2",
        delay_actor: bool = False,
        delay_value: bool = False,
    ) -> None:
        super().__init__()
        self.delay_actor = delay_actor
        self.delay_value = delay_value
        self.convert_to_functional(
            actor_network,
            "actor_network",
            create_target_params=self.delay_actor,
        )
        self.convert_to_functional(
            value_network,
            "value_network",
            create_target_params=self.delay_value,
            compare_against=list(actor_network.parameters()),
        )
        self.actor_critic = ActorCriticWrapper(self.actor_network, self.value_network)

        self.actor_in_keys = actor_network.in_keys

        self.gamma = gamma
        self.loss_funtion = loss_function

    def forward(self, input_tensordict: _TensorDict) -> TensorDict:
        """Computes the DDPG losses given a tensordict sampled from the replay buffer.
        This function will also write a "td_error" key that can be used by prioritized replay buffers to assign
            a priority to items in the tensordict.

        Args:
            input_tensordict (_TensorDict): a tensordict with keys ["done", "reward"] and the in_keys of the actor
                and value networks.

        Returns:
            a tuple of 2 tensors containing the DDPG loss.

        """
        if not input_tensordict.device == self.device:
            raise RuntimeError(
                f"Got device={input_tensordict.device} but actor_network.device={self.device} "
                f"(self.device={self.device})"
            )
        n = len(self.actor_network_params)
        if not hasattr(self, "td"):
            self.td = input_tensordict
            self.grad = functorch.grad(
                lambda *params: self._get_loss(self.td, params[:n], params[n:])
            )
        else:
            self.td.update_(input_tensordict)
        g = self.grad(*self.actor_network_params, *self.value_network_params)

    def _get_loss(self, input_tensordict, actor_network_params, value_network_params):
        with set_exploration_mode("mode"), hold_out_params(
            self.value_network_params
        ) as value_net_actor_loss_params:
            actor_loss_params_an = actor_network_params
            actor_loss_params_vn = value_net_actor_loss_params
            actor_loss_buffers_an = self.actor_network_buffers
            actor_loss_buffers_vn = self.value_network_buffers

            value_loss_target_params_an = self.target_actor_network_params
            value_loss_target_params_vn = self.target_value_network_params
            value_loss_target_buffers_an = self.target_actor_network_buffers
            value_loss_target_buffers_vn = self.target_value_network_buffers

            value_loss_params_vn = value_network_params
            value_loss_buffers_vn = self.value_network_buffers

            tensordict = torch.stack(
                [
                    input_tensordict,
                    step_tensordict(input_tensordict),
                ],
                0,
            ).to_tensordict()

            actor_params = zip_stack(
                (actor_loss_params_an, value_loss_target_params_an)
            )
            actor_buffers = zip_stack(
                (actor_loss_buffers_an, value_loss_target_buffers_an)
            )

            value_params = zip_stack(
                (
                    actor_loss_params_vn,
                    value_loss_target_params_vn,
                    value_loss_params_vn,
                )
            )
            value_buffers = zip_stack(
                (
                    actor_loss_buffers_vn,
                    value_loss_target_buffers_vn,
                    value_loss_buffers_vn,
                )
            )

            tensordict = self.actor_network(
                tensordict, params=actor_params, buffers=actor_buffers, vmap=(0, 0, 0)
            )
            tensordict = torch.cat(
                [
                    tensordict.select(*self.value_network.in_keys),
                    input_tensordict.select(*self.value_network.in_keys).unsqueeze(0),
                ],
                0,
            )
            tensordict = self.value_network(
                tensordict, params=value_params, buffers=value_buffers, vmap=(0, 0, 0)
            )
            loss_actor, next_state_action_value, pred_val = tensordict.get(
                "state_action_value"
            ).squeeze(-1)
            loss_actor = -loss_actor

        target_value = next_state_value(
            input_tensordict,
            gamma=self.gamma,
            pred_next_val=next_state_action_value,
        )

        loss_value = distance_loss(
            pred_val, target_value, loss_function=self.loss_funtion
        )

        td_error = abs(pred_val - target_value)

        td_error = td_error.detach()
        td_error = td_error.unsqueeze(input_tensordict.ndimension())
        td_error = td_error.to(input_tensordict.device)
        input_tensordict.set(
            "td_error",
            td_error,
            inplace=False,
        )
        return loss_actor.mean() + loss_value.mean()
