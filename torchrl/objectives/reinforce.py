# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Callable

import torch

from torchrl.data.tensordict.tensordict import _TensorDict, TensorDict
from torchrl.envs.utils import step_tensordict
from torchrl.modules import TensorDictModule, ProbabilisticTensorDictModule
from torchrl.objectives import distance_loss
from torchrl.objectives.common import _LossModule


class ReinforceLoss(_LossModule):
    """Reinforce loss module, as presented in
    "Simple statistical gradient-following algorithms for connectionist reinforcement learning", Williams, 1992
    https://doi.org/10.1007/BF00992696

    """

    def __init__(
        self,
        actor_network: ProbabilisticTensorDictModule,
        advantage_module: Callable[[_TensorDict], _TensorDict],
        critic: Optional[TensorDictModule] = None,
        delay_value: bool = False,
        gamma: float = 0.99,
        advantage_key: str = "advantage",
        advantage_diff_key: str = "value_error",
        loss_critic_type: str = "smooth_l1",
    ) -> None:
        super().__init__()

        self.delay_value = delay_value
        self.advantage_key = advantage_key
        self.advantage_diff_key = advantage_diff_key
        self.loss_critic_type = loss_critic_type
        self.gamma = gamma

        if (
            hasattr(advantage_module, "is_functional")
            and not advantage_module.is_functional
        ):
            raise RuntimeError(
                "The advantage module must be functional, as it must support params and target params arguments"
            )

        # Actor
        self.convert_to_functional(
            actor_network,
            "actor_network",
            create_target_params=False,
        )

        # Value
        if critic is not None:
            self.convert_to_functional(
                critic,
                "critic",
                create_target_params=self.delay_value,
            )

        self.advantage_module = advantage_module

    def forward(self, tensordict: _TensorDict) -> _TensorDict:
        # get advantage
        tensordict = self.advantage_module(
            tensordict,
            params=self.critic_params,
            buffers=self.critic_buffers,
            target_params=self.target_critic_params,
            target_buffers=self.target_critic_buffers,
        )
        advantage = tensordict.get(self.advantage_key)

        # compute log-prob
        tensordict = self.actor_network(
            tensordict,
            params=self.actor_network_params,
            buffers=self.actor_network_buffers,
        )

        log_prob = tensordict.get("sample_log_prob")
        loss_actor = -log_prob * advantage.detach()
        loss_actor = loss_actor.mean()
        td_out = TensorDict({"loss_actor": loss_actor}, [])

        td_out.set("loss_value", self.loss_critic(tensordict).mean())

        return td_out

    def loss_critic(self, tensordict: _TensorDict) -> torch.Tensor:
        if self.advantage_diff_key in tensordict.keys():
            advantage_diff = tensordict.get(self.advantage_diff_key)
            if not advantage_diff.requires_grad:
                raise RuntimeError(
                    "value_target retrieved from tensordict does not requires grad."
                )
            loss_value = distance_loss(
                advantage_diff,
                torch.zeros_like(advantage_diff),
                loss_function=self.loss_critic_type,
            )
        else:
            with torch.no_grad():
                reward = tensordict.get("reward")
                next_td = step_tensordict(tensordict)
                next_value = self.critic(
                    next_td,
                    params=self.critic_params,
                    buffers=self.critic_buffers,
                ).get("state_value")
                value_target = reward + next_value * self.gamma
            tensordict_select = tensordict.select(*self.critic.in_keys).clone()
            value = self.critic(
                tensordict_select,
                params=self.critic_params,
                buffers=self.critic_buffers,
            ).get("state_value")

            loss_value = distance_loss(
                value, value_target, loss_function=self.loss_critic_type
            )
        return loss_value
