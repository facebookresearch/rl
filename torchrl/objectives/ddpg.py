# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Tuple

import torch

from torchrl.data.tensordict.tensordict import _TensorDict, TensorDict
from torchrl.envs.utils import set_exploration_mode
from torchrl.modules import TensorDictModule
from torchrl.modules.tensordict_module.actors import ActorCriticWrapper
from torchrl.objectives.utils import (
    distance_loss,
    hold_out_params,
    next_state_value,
)
from .common import _LossModule


class DDPGLoss(_LossModule):
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
        )

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
        loss_actor = self._loss_actor(input_tensordict)
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
        # TODO: validate this experimentally
        # td_copy = self.value_network(
        #     td_copy,
        #     params=self.target_value_network_params,
        #     buffers=self.value_network_buffers,
        # )
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

        actor_critic = ActorCriticWrapper(self.actor_network, self.value_network)
        target_params = list(self.target_actor_network_params) + list(
            self.target_value_network_params
        )
        target_buffers = list(self.target_actor_network_buffers) + list(
            self.target_value_network_buffers
        )
        with set_exploration_mode("mode"):
            target_value = next_state_value(
                tensordict,
                actor_critic,
                gamma=self.gamma,
                params=target_params,
                buffers=target_buffers,
            )

        # td_error = pred_val - target_value
        loss_value = distance_loss(
            pred_val, target_value, loss_function=self.loss_funtion
        )

        return loss_value, abs(pred_val - target_value), pred_val, target_value
