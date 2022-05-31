# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from numbers import Number
from typing import Tuple, Union

import numpy as np
import torch
from torch import Tensor

from torchrl.data.tensordict.tensordict import _TensorDict, TensorDict
from torchrl.modules import ProbabilisticActor
from torchrl.modules import TensorDictModule
from torchrl.modules.tensordict_module.actors import (
    ActorCriticWrapper,
)
from torchrl.objectives.utils import distance_loss, next_state_value
from .common import _LossModule

__all__ = ["SACLoss"]

from torchrl.envs.utils import set_exploration_mode


class SACLoss(_LossModule):
    """
    TorchRL implementation of the SAC loss, as presented in "Soft Actor-Critic: Off-Policy Maximum Entropy Deep
    Reinforcement Learning with a Stochastic Actor" https://arxiv.org/pdf/1801.01290.pdf

    Args:
        actor_network (ProbabilisticActor): stochastic actor
        qvalue_network (TensorDictModule): Q(s, a) parametric model
        value_network (TensorDictModule): V(s) parametric model\
        qvalue_network_bis (ProbabilisticTDModule, optional): if required, the
            Q-value can be computed twice independently using two separate
            networks. The minimum predicted value will then be used for
            inference.
        gamma (number, optional): discount for return computation
            Default is 0.99
        priority_key (str, optional): tensordict key where to write the
            priority (for prioritized replay buffer usage). Default is
            `"td_error"`.
        loss_function (str, optional): loss function to be used with
            the value function loss. Default is `"smooth_l1"`.
        alpha_init (float, optional): initial entropy multiplier.
            Default is 1.0.
        min_alpha (float, optional): min value of alpha.
            Default is 0.1.
        max_alpha (float, optional): max value of alpha.
            Default is 10.0.
        fixed_alpha (bool, optional): if True, alpha will be fixed to its
            initial value. Otherwise, alpha will be optimized to
            match the 'target_entropy' value.
            Default is `False`.
        target_entropy (float or str, optional): Target entropy for the
            stochastic policy. Default is "auto", where target entropy is
            computed as `-prod(n_actions)`.
        delay_actor (bool, optional): Whether to separate the target actor
            networks from the actor networks used for data collection.
            Default is `False`.
        delay_qvalue (bool, optional): Whether to separate the target Q value
            networks from the Q value networks used for data collection.
            Default is `False`.
        delay_value (bool, optional): Whether to separate the target value
            networks from the value networks used for data collection.
            Default is `False`.
    """

    def __init__(
        self,
        actor_network: ProbabilisticActor,
        qvalue_network: TensorDictModule,
        value_network: TensorDictModule,
        num_qvalue_nets: int = 2,
        gamma: Number = 0.99,
        priotity_key: str = "td_error",
        loss_function: str = "smooth_l1",
        alpha_init: float = 1.0,
        min_alpha: float = 0.1,
        max_alpha: float = 10.0,
        fixed_alpha: bool = False,
        target_entropy: Union[str, float] = "auto",
        delay_actor: bool = False,
        delay_qvalue: bool = False,
        delay_value: bool = False,
    ) -> None:
        super().__init__()

        # Actor
        self.delay_actor = delay_actor
        self.convert_to_functional(
            actor_network,
            "actor_network",
            create_target_params=self.delay_actor,
        )

        # Value
        self.delay_value = delay_value
        self.convert_to_functional(
            value_network,
            "value_network",
            create_target_params=self.delay_value,
        )

        # Q value
        self.delay_qvalue = delay_qvalue
        self.num_qvalue_nets = num_qvalue_nets
        self.convert_to_functional(
            qvalue_network,
            "qvalue_network",
            num_qvalue_nets,
            create_target_params=self.delay_qvalue,
        )

        self.gamma = gamma
        self.priority_key = priotity_key
        self.loss_function = loss_function
        try:
            device = next(self.parameters()).device
        except AttributeError:
            device = torch.device("cpu")
        self.register_buffer("alpha_init", torch.tensor(alpha_init, device=device))
        self.register_buffer(
            "min_log_alpha", torch.tensor(min_alpha, device=device).log()
        )
        self.register_buffer(
            "max_log_alpha", torch.tensor(max_alpha, device=device).log()
        )
        self.fixed_alpha = fixed_alpha
        if fixed_alpha:
            self.register_buffer(
                "log_alpha", torch.tensor(math.log(alpha_init), device=device)
            )
        else:
            self.register_parameter(
                "log_alpha",
                torch.nn.Parameter(torch.tensor(math.log(alpha_init), device=device)),
            )

        if target_entropy == "auto":
            if actor_network.spec is None:
                raise RuntimeError(
                    "Cannot infer the dimensionality of the action. Consider providing "
                    "the target entropy explicitely or provide the spec of the "
                    "action tensor in the actor network."
                )
            target_entropy = -float(np.prod(actor_network.spec["action"].shape))
        self.register_buffer(
            "target_entropy", torch.tensor(target_entropy, device=device)
        )

    @property
    def device(self) -> torch.device:
        for p in self.actor_network_params:
            return p.device
        for p in self.qvalue_network_params:
            return p.device
        for p in self.value_network_params:
            return p.device
        raise RuntimeError(
            "At least one of the networks of SACLoss must have trainable " "parameters."
        )

    def forward(self, tensordict: _TensorDict) -> _TensorDict:
        if tensordict.ndimension() > 1:
            tensordict = tensordict.view(-1)

        device = self.device
        td_device = tensordict.to(device)

        loss_actor = self._loss_actor(td_device)
        loss_qvalue, priority = self._loss_qvalue(td_device)
        loss_value = self._loss_value(td_device)
        loss_alpha = self._loss_alpha(td_device)
        tensordict.set(self.priority_key, priority)
        if (loss_actor.shape != loss_qvalue.shape) or (
            loss_actor.shape != loss_value.shape
        ):
            raise RuntimeError(
                f"Losses shape mismatch: {loss_actor.shape}, {loss_qvalue.shape} and {loss_value.shape}"
            )
        return TensorDict(
            {
                "loss_actor": loss_actor.mean(),
                "loss_qvalue": loss_qvalue.mean(),
                "loss_value": loss_value.mean(),
                "loss_alpha": loss_alpha.mean(),
                "alpha": self._alpha,
                "entropy": -td_device.get("_log_prob").mean().detach(),
            },
            [],
        )

    def _loss_actor(self, tensordict: _TensorDict) -> Tensor:
        # KL lossa
        with set_exploration_mode("random"):
            dist = self.actor_network.get_dist(
                tensordict,
                params=list(self.actor_network_params),
                buffers=list(self.actor_network_buffers),
            )[0]
            a_reparm = dist.rsample()
        # if not self.actor_network.spec.is_in(a_reparm):
        #     a_reparm.data.copy_(self.actor_network.spec.project(a_reparm.data))
        log_prob = dist.log_prob(a_reparm)

        td_q = tensordict.select(*self.qvalue_network.in_keys)
        td_q.set("action", a_reparm)
        td_q = self.qvalue_network(
            td_q,
            params=list(self.target_qvalue_network_params),
            buffers=list(self.qvalue_network_buffers),
            vmap=True,
        )
        min_q_logprob = td_q.get("state_action_value").min(0)[0].squeeze(-1)

        if log_prob.shape != min_q_logprob.shape:
            raise RuntimeError(
                f"Losses shape mismatch: {log_prob.shape} and {min_q_logprob.shape}"
            )

        # write log_prob in tensordict for alpha loss
        tensordict.set("_log_prob", log_prob.detach())
        return self._alpha * log_prob - min_q_logprob

    def _loss_qvalue(self, tensordict: _TensorDict) -> Tuple[Tensor, Tensor]:
        actor_critic = ActorCriticWrapper(self.actor_network, self.value_network)
        params = list(self.target_actor_network_params) + list(
            self.target_value_network_params
        )
        buffers = list(self.target_actor_network_buffers) + list(
            self.target_value_network_buffers
        )
        with set_exploration_mode("mode"):
            target_value = next_state_value(
                tensordict,
                actor_critic,
                gamma=self.gamma,
                next_val_key="state_value",
                params=params,
                buffers=buffers,
            )

        # value loss
        qvalue_network = self.qvalue_network

        # Q-nets must be trained independently: as such, we split the data in 2 if required and train each q-net on
        # one half of the data.
        shape = tensordict.shape
        if shape[0] % self.num_qvalue_nets != 0:
            raise RuntimeError(
                f"Batch size={tensordict.shape} is incompatible "
                f"with num_qvqlue_nets={self.num_qvalue_nets}."
            )
        tensordict_chunks = torch.stack(
            tensordict.chunk(self.num_qvalue_nets, dim=0), 0
        )
        target_chunks = torch.stack(target_value.chunk(self.num_qvalue_nets, dim=0), 0)

        # if vmap=True, it is assumed that the input tensordict must be cast to the param shape
        tensordict_chunks = qvalue_network(
            tensordict_chunks,
            params=list(self.qvalue_network_params),
            buffers=list(self.qvalue_network_buffers),
            vmap=(
                0,
                0,
                0,
                0,
            ),
        )
        pred_val = tensordict_chunks.get("state_action_value").squeeze(-1)
        loss_value = distance_loss(
            pred_val, target_chunks, loss_function=self.loss_function
        ).view(*shape)
        priority_value = torch.cat(abs(pred_val - target_chunks).unbind(0), 0)

        return loss_value, priority_value

    def _loss_value(self, tensordict: _TensorDict) -> Tensor:
        # value loss
        td_copy = tensordict.select(*self.value_network.in_keys).detach()
        self.value_network(
            td_copy,
            params=list(self.value_network_params),
            buffers=list(self.value_network_buffers),
        )
        pred_val = td_copy.get("state_value").squeeze(-1)

        action_dist = self.actor_network.get_dist(
            td_copy,
            params=list(self.target_actor_network_params),
            buffers=list(self.target_actor_network_buffers),
        )[
            0
        ]  # resample an action
        action = action_dist.rsample()
        # if not self.actor_network.spec.is_in(action):
        #     action.data.copy_(self.actor_network.spec.project(action.data))

        td_copy.set("action", action, inplace=False)

        qval_net = self.qvalue_network
        td_copy = qval_net(
            td_copy,
            params=list(self.target_qvalue_network_params),
            buffers=list(self.target_qvalue_network_buffers),
            vmap=True,
        )

        min_qval = td_copy.get("state_action_value").squeeze(-1).min(0)[0]

        log_p = action_dist.log_prob(action)
        if log_p.shape != min_qval.shape:
            raise RuntimeError(
                f"Losses shape mismatch: {min_qval.shape} and {log_p.shape}"
            )
        target_val = min_qval - self._alpha * log_p

        loss_value = distance_loss(
            pred_val, target_val, loss_function=self.loss_function
        )
        return loss_value

    def _loss_alpha(self, tensordict: _TensorDict) -> Tensor:
        log_pi = tensordict.get("_log_prob")
        if self.target_entropy is not None:
            # we can compute this loss even if log_alpha is not a parameter
            alpha_loss = -self.log_alpha.exp() * (log_pi.detach() + self.target_entropy)
        else:
            # placeholder
            alpha_loss = torch.zeros_like(log_pi)
        return alpha_loss

    @property
    def _alpha(self):
        self.log_alpha.data.clamp_(self.min_log_alpha, self.max_log_alpha)
        with torch.no_grad():
            alpha = self.log_alpha.exp()
        return alpha
