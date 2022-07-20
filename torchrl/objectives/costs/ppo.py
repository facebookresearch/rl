# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Callable, Optional, Tuple

import torch
from torch import distributions as d

from torchrl.data.tensordict.tensordict import _TensorDict, TensorDict
from torchrl.envs.utils import step_tensordict
from torchrl.modules import TensorDictModule
from ...modules.tensordict_module import ProbabilisticTensorDictModule

__all__ = ["PPOLoss", "ClipPPOLoss", "KLPENPPOLoss"]

from torchrl.objectives.costs.utils import distance_loss
from .common import LossModule


class PPOLoss(LossModule):
    """
    A parent PPO loss class.

    PPO (Proximal Policy Optimisation) is a model-free, online RL algorithm that makes use of a recorded (batch of)
    trajectories to perform several optimization steps, while actively preventing the updated policy to deviate too
    much from its original parameter configuration.

    PPO loss can be found in different flavours, depending on the way the constrained optimisation is implemented:
        ClipPPOLoss and KLPENPPOLoss.
    Unlike its subclasses, this class does not implement any regularisation and should therefore be used cautiously.

    For more details regarding PPO, refer to: "Proximal Policy Optimization Algorithms",
    https://arxiv.org/abs/1707.06347

    Args:
        actor (ProbabilisticTensorDictModule): policy operator.
        critic (ValueOperator): value operator.
        advantage_key (str): the input tensordict key where the advantage is expected to be written.
            default: "advantage"
        entropy_bonus (bool): if True, an entropy bonus will be added to the loss to favour exploratory policies.
        samples_mc_entropy (int): if the distribution retrieved from the policy operator does not have a closed form
            formula for the entropy, a Monte-Carlo estimate will be used. samples_mc_entropy will control how many
            samples will be used to compute this estimate.
            default: 1
        entropy_coef (scalar): entropy multiplier when computing the total loss.
            default: 0.01
        critic_coef (scalar): critic loss multiplier when computing the total loss.
            default: 1.0
        gamma (scalar): a discount factor for return computation.
        loss_function (str): loss function for the value discrepancy. Can be one of "l1", "l2" or "smooth_l1".

    """

    def __init__(
        self,
        actor: ProbabilisticTensorDictModule,
        critic: TensorDictModule,
        advantage_key: str = "advantage",
        advantage_diff_key: str = "value_error",
        entropy_bonus: bool = True,
        samples_mc_entropy: int = 1,
        entropy_coef: float = 0.01,
        critic_coef: float = 1.0,
        gamma: float = 0.99,
        loss_critic_type: str = "smooth_l1",
        advantage_module: Optional[Callable[[_TensorDict], _TensorDict]] = None,
        make_stateless: bool = True,
    ):
        super().__init__()
        self.make_stateless = make_stateless
        if self.make_stateless:
            self.convert_to_functional(actor, "actor")
            # we want to make sure there are no duplicates in the params: the
            # params of critic must be refs to actor if they're shared
            self.convert_to_functional(
                critic, "critic", compare_against=self.actor_params
            )
        else:
            self.actor = actor
            self.critic = critic
            self.actor_params = None
            self.actor_buffers = None
            self.critic_params = None
            self.critic_buffers = None

        self.advantage_key = advantage_key
        self.advantage_diff_key = advantage_diff_key
        self.samples_mc_entropy = samples_mc_entropy
        self.entropy_bonus = entropy_bonus and entropy_coef
        self.entropy_coef = entropy_coef
        self.critic_coef = critic_coef
        self.gamma = gamma
        self.loss_critic_type = loss_critic_type
        self.advantage_module = advantage_module

    def reset(self) -> None:
        pass

    def get_entropy_bonus(self, dist: Optional[d.Distribution] = None) -> torch.Tensor:
        try:
            entropy = dist.entropy()
        except NotImplementedError:
            x = dist.rsample((self.samples_mc_entropy,))
            entropy = -dist.log_prob(x)
        return entropy.unsqueeze(-1)

    def _log_weight(
        self, tensordict: _TensorDict
    ) -> Tuple[torch.Tensor, d.Distribution]:
        # current log_prob of actions
        action = tensordict.get("action")
        if action.requires_grad:
            raise RuntimeError("tensordict stored action requires grad.")
        tensordict_clone = tensordict.select(*self.actor.in_keys).clone()

        dist, *_ = self.actor.get_dist(
            tensordict_clone, params=self.actor_params, buffers=self.actor_buffers
        )
        log_prob = dist.log_prob(action)
        log_prob = log_prob.unsqueeze(-1)

        prev_log_prob = tensordict.get("sample_log_prob")
        if prev_log_prob.requires_grad:
            raise RuntimeError("tensordict prev_log_prob requires grad.")

        log_weight = log_prob - prev_log_prob
        return log_weight, dist

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
                    next_td, params=self.critic_params, buffers=self.critic_buffers
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
        return self.critic_coef * loss_value

    def forward(self, tensordict: _TensorDict) -> _TensorDict:
        if self.advantage_module is not None:
            tensordict = self.advantage_module(
                tensordict,
            )
        tensordict = tensordict.clone()
        advantage = tensordict.get(self.advantage_key)
        log_weight, dist = self._log_weight(tensordict)
        neg_loss = (log_weight.exp() * advantage).mean()
        td_out = TensorDict({"loss_objective": -neg_loss.mean()}, [])
        if self.entropy_bonus:
            entropy = self.get_entropy_bonus(dist)
            td_out.set("entropy", entropy.mean().detach())  # for logging
            td_out.set("loss_entropy", -self.entropy_coef * entropy.mean())
        if self.critic_coef:
            loss_critic = self.loss_critic(tensordict).mean()
            td_out.set("loss_critic", loss_critic.mean())
        return td_out


class ClipPPOLoss(PPOLoss):
    """
    Clipped PPO loss.

    The clipped importance weighted loss is computed as follows:
        loss = -min( weight * advantage, min(max(weight, 1-eps), 1+eps) * advantage)

    Args:
        actor (ProbabilisticTensorDictModule): policy operator.
        critic (ValueOperator): value operator.
        advantage_key (str): the input tensordict key where the advantage is expected to be written.
            default: "advantage"
        clip_epsilon (scalar): weight clipping threshold in the clipped PPO loss equation.
            default: 0.2
        entropy_bonus (bool): if True, an entropy bonus will be added to the loss to favour exploratory policies.
        samples_mc_entropy (int): if the distribution retrieved from the policy operator does not have a closed form
            formula for the entropy, a Monte-Carlo estimate will be used. samples_mc_entropy will control how many
            samples will be used to compute this estimate.
            default: 1
        entropy_coef (scalar): entropy multiplier when computing the total loss.
            default: 0.01
        critic_coef (scalar): critic loss multiplier when computing the total loss.
            default: 1.0
        gamma (scalar): a discount factor for return computation.
        loss_function (str): loss function for the value discrepancy. Can be one of "l1", "l2" or "smooth_l1".

    """

    def __init__(
        self,
        actor: ProbabilisticTensorDictModule,
        critic: TensorDictModule,
        advantage_key: str = "advantage",
        clip_epsilon: float = 0.2,
        entropy_bonus: bool = True,
        samples_mc_entropy: int = 1,
        entropy_coef: float = 0.01,
        critic_coef: float = 1.0,
        gamma: float = 0.99,
        loss_critic_type: str = "l2",
        **kwargs,
    ):
        super(ClipPPOLoss, self).__init__(
            actor,
            critic,
            advantage_key,
            entropy_bonus=entropy_bonus,
            samples_mc_entropy=samples_mc_entropy,
            entropy_coef=entropy_coef,
            critic_coef=critic_coef,
            gamma=gamma,
            loss_critic_type=loss_critic_type,
            **kwargs,
        )
        self.clip_epsilon = clip_epsilon
        self._clip_bounds = (
            math.log1p(-self.clip_epsilon),
            math.log1p(self.clip_epsilon),
        )

    def forward(self, tensordict: _TensorDict) -> _TensorDict:
        if self.advantage_module is not None:
            tensordict = self.advantage_module(tensordict)
        tensordict = tensordict.clone()
        advantage = tensordict.get(self.advantage_key)
        log_weight, dist = self._log_weight(tensordict)
        # ESS for logging
        with torch.no_grad():
            # In theory, ESS should be computed on particles sampled from the same source. Here we sample according
            # to different, unrelated trajectories, which is not standard. Still it can give a idea of the dispersion
            # of the weights.
            lw = log_weight.squeeze()
            ess = (2 * lw.logsumexp(0) - (2 * lw).logsumexp(0)).exp()
            batch = log_weight.shape[0]

        if not advantage.shape == log_weight.shape:
            raise RuntimeError(
                f"advantage.shape and log_weight.shape do not match (got {advantage.shape} "
                f"and {log_weight.shape})"
            )
        gain1 = log_weight.exp() * advantage
        log_weight_clip = torch.empty_like(log_weight)
        # log_weight_clip.data.clamp_(*self._clip_bounds)
        idx_pos = advantage >= 0
        log_weight_clip[idx_pos] = log_weight[idx_pos].clamp_max(self._clip_bounds[1])
        log_weight_clip[~idx_pos] = log_weight[~idx_pos].clamp_min(self._clip_bounds[0])

        gain2 = log_weight_clip.exp() * advantage
        gain = torch.stack([gain1, gain2], -1).min(dim=-1)[0]
        td_out = TensorDict({"loss_objective": -gain.mean()}, [])

        if self.entropy_bonus:
            entropy = self.get_entropy_bonus(dist)
            td_out.set("entropy", entropy.mean().detach())  # for logging
            td_out.set("loss_entropy", -self.entropy_coef * entropy.mean())
        if self.critic_coef:
            loss_critic = self.loss_critic(tensordict)
            td_out.set("loss_critic", loss_critic.mean())
        td_out.set("ESS", ess.mean() / batch)
        return td_out


class KLPENPPOLoss(PPOLoss):
    """
    KL Penalty PPO loss.

    The KL penalty loss has the following formula:
        loss = loss - beta * KL(old_policy, new_policy)
    The "beta" parameter is adapted on-the-fly to match a target KL divergence between the new and old policy, thus
    favouring a certain level of distancing between the two while still preventing them to be too much apart.

    Args:
        actor (ProbabilisticTensorDictModule): policy operator.
        critic (ValueOperator): value operator.
        advantage_key (str): the input tensordict key where the advantage is expected to be written.
            default: "advantage"
        dtarg (scalar): target KL divergence.
        beta (scalar): initial KL divergence multiplier.
            default: 1.0
        increment (scalar): how much beta should be incremented if KL > dtarg. Valid range: increment >= 1.0
            default: 2.0
        decrement (scalar): how much beta should be decremented if KL < dtarg. Valid range: decrement <= 1.0
            default: 0.5
        entropy_bonus (bool): if True, an entropy bonus will be added to the loss to favour exploratory policies.
        samples_mc_entropy (int): if the distribution retrieved from the policy operator does not have a closed form
            formula for the entropy, a Monte-Carlo estimate will be used. samples_mc_entropy will control how many
            samples will be used to compute this estimate.
            default: 1
        entropy_coef (scalar): entropy multiplier when computing the total loss.
            default: 0.01
        critic_coef (scalar): critic loss multiplier when computing the total loss.
            default: 1.0
        gamma (scalar): a discount factor for return computation.
        loss_critic_type (str): loss function for the value discrepancy. Can be one of "l1", "l2" or "smooth_l1".

    """

    def __init__(
        self,
        actor: ProbabilisticTensorDictModule,
        critic: TensorDictModule,
        advantage_key="advantage",
        dtarg: float = 0.01,
        beta: float = 1.0,
        increment: float = 2,
        decrement: float = 0.5,
        samples_mc_kl: int = 1,
        entropy_bonus: bool = True,
        samples_mc_entropy: int = 1,
        entropy_coef: float = 0.01,
        critic_coef: float = 1.0,
        gamma: float = 0.99,
        loss_critic_type: str = "l2",
        **kwargs,
    ):
        super(KLPENPPOLoss, self).__init__(
            actor,
            critic,
            advantage_key,
            entropy_bonus=entropy_bonus,
            samples_mc_entropy=samples_mc_entropy,
            entropy_coef=entropy_coef,
            critic_coef=critic_coef,
            gamma=gamma,
            loss_critic_type=loss_critic_type,
            **kwargs,
        )

        self.dtarg = dtarg
        self._beta_init = beta
        self.beta = beta

        if increment < 1.0:
            raise ValueError(
                f"increment should be >= 1.0 in KLPENPPOLoss, got {increment:4.4f}"
            )
        self.increment = increment
        if decrement > 1.0:
            raise ValueError(
                f"decrement should be <= 1.0 in KLPENPPOLoss, got {decrement:4.4f}"
            )
        self.decrement = decrement
        self.samples_mc_kl = samples_mc_kl

    def forward(self, tensordict: _TensorDict) -> TensorDict:
        if self.advantage_module is not None:
            tensordict = self.advantage_module(tensordict)
        tensordict = tensordict.clone()
        advantage = tensordict.get(self.advantage_key)
        log_weight, dist = self._log_weight(tensordict)
        neg_loss = log_weight.exp() * advantage

        tensordict_clone = tensordict.select(
            *self.actor.in_keys, *self.actor.out_keys
        ).clone()

        previous_dist = self.actor.build_dist_from_params(tensordict_clone)
        current_dist, *_ = self.actor.get_dist(
            tensordict_clone, params=self.actor_params, buffers=self.actor_buffers
        )
        try:
            kl = torch.distributions.kl.kl_divergence(previous_dist, current_dist)
        except NotImplementedError:
            x = previous_dist.sample((self.samples_mc_kl,))
            kl = (previous_dist.log_prob(x) - current_dist.log_prob(x)).mean(0)
        kl = kl.unsqueeze(-1)
        neg_loss = neg_loss - self.beta * kl
        if kl.mean() > self.dtarg * 1.5:
            self.beta *= self.increment
        elif kl.mean() < self.dtarg / 1.5:
            self.beta *= self.decrement
        td_out = TensorDict(
            {
                "loss_objective": -neg_loss.mean(),
                "kl": kl.detach().mean(),
            },
            [],
        )

        if self.entropy_bonus:
            entropy = self.get_entropy_bonus(dist)
            td_out.set("entropy", entropy.mean().detach())  # for logging
            td_out.set("loss_entropy", -self.entropy_coef * entropy.mean())

        if self.critic_coef:
            loss_critic = self.loss_critic(tensordict)
            td_out.set("loss_critic", loss_critic.mean())

        return td_out

    def reset(self) -> None:
        self.beta = self._beta_init
