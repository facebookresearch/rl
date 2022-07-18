# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

__all__ = [
    "make_sac_loss",
    "make_dqn_loss",
    "make_ddpg_loss",
    "make_target_updater",
    "make_ppo_loss",
    "make_redq_loss",
]

from typing import Optional, Tuple, Any

from torchrl.modules import ActorValueOperator, ActorCriticOperator
from torchrl.objectives import (
    DistributionalDQNLoss,
    ClipPPOLoss,
    vecDDPGLoss as DDPGLoss,
    # DDPGLoss,
    DQNLoss,
    HardUpdate,
    KLPENPPOLoss,
    PPOLoss,
    SACLoss,
    SoftUpdate,
)
from torchrl.objectives.costs.common import LossModule
from torchrl.objectives.costs.redq import REDQLoss

# from torchrl.objectives.costs.redq import REDQLoss
from torchrl.objectives.costs.utils import _TargetNetUpdate
from torchrl.objectives.returns.advantages import GAE


def make_target_updater(
    cfg: "DictConfig", loss_module: LossModule
) -> Optional[_TargetNetUpdate]:
    """Builds a target network weight update object."""
    if cfg.loss == "double":
        if not cfg.hard_update:
            target_net_updater = SoftUpdate(
                loss_module, 1 - 1 / cfg.value_network_update_interval
            )
        else:
            target_net_updater = HardUpdate(
                loss_module, cfg.value_network_update_interval
            )
        # assert len(target_net_updater.net_pairs) == 3, "length of target_net_updater nets should be 3"
        target_net_updater.init_()
    else:
        if cfg.hard_update:
            raise RuntimeError(
                "hard/soft-update are supposed to be used with double SAC loss. "
                "Consider using --loss=double or discarding the hard_update flag."
            )
        target_net_updater = None
    return target_net_updater


def make_sac_loss(model, cfg) -> Tuple[SACLoss, Optional[_TargetNetUpdate]]:
    """Builds the SAC loss module."""
    loss_kwargs = {}
    if hasattr(cfg, "distributional") and cfg.distributional:
        raise NotImplementedError
    else:
        loss_kwargs.update({"loss_function": cfg.loss_function})
        loss_kwargs.update(
            {
                "target_entropy": cfg.target_entropy
                if cfg.target_entropy is not None
                else "auto"
            }
        )
        loss_class = SACLoss
        if cfg.loss == "double":
            loss_kwargs.update(
                {
                    "delay_actor": False,
                    "delay_qvalue": True,
                    "delay_value": True,
                }
            )
        elif cfg.loss == "single":
            loss_kwargs.update(
                {
                    "delay_actor": False,
                    "delay_qvalue": False,
                    "delay_value": False,
                }
            )
        else:
            raise NotImplementedError(
                f"cfg.loss {cfg.loss} unsupported. Consider chosing from 'double' or 'single'"
            )

    actor_model, qvalue_model, value_model = model

    loss_module = loss_class(
        actor_network=actor_model,
        qvalue_network=qvalue_model,
        value_network=value_model,
        num_qvalue_nets=cfg.num_q_values,
        gamma=cfg.gamma,
        **loss_kwargs,
    )
    target_net_updater = make_target_updater(cfg, loss_module)
    return loss_module, target_net_updater


def make_redq_loss(model, cfg) -> Tuple[REDQLoss, Optional[_TargetNetUpdate]]:
    """Builds the REDQ loss module."""
    loss_kwargs = {}
    if hasattr(cfg, "distributional") and cfg.distributional:
        raise NotImplementedError
    else:
        loss_kwargs.update({"loss_function": cfg.loss_function})
        loss_kwargs.update({"delay_qvalue": cfg.loss == "double"})
        loss_class = REDQLoss
    if isinstance(model, ActorValueOperator):
        actor_model = model.get_policy_operator()
        qvalue_model = model.get_value_operator()
    elif isinstance(model, ActorCriticOperator):
        raise RuntimeError(
            "Although REDQ Q-value depends upon selected actions, using the"
            "ActorCriticOperator will lead to resampling of the actions when"
            "computing the Q-value loss, which we don't want. Please use the"
            "ActorValueOperator instead."
        )
    else:
        actor_model, qvalue_model = model

    loss_module = loss_class(
        actor_network=actor_model,
        qvalue_network=qvalue_model,
        num_qvalue_nets=cfg.num_q_values,
        gamma=cfg.gamma,
        gSDE=cfg.gSDE,
        **loss_kwargs,
    )
    target_net_updater = make_target_updater(cfg, loss_module)
    return loss_module, target_net_updater


def make_ddpg_loss(model, cfg) -> Tuple[DDPGLoss, Optional[_TargetNetUpdate]]:
    """Builds the DDPG loss module."""
    actor, value_net = model
    loss_kwargs = {}
    if cfg.distributional:
        raise NotImplementedError
    else:
        loss_kwargs.update({"loss_function": cfg.loss_function})
        loss_class = DDPGLoss
    if cfg.loss not in ("single", "double"):
        raise NotImplementedError
    double_loss = cfg.loss == "double"
    loss_kwargs.update({"delay_actor": double_loss, "delay_value": double_loss})
    loss_module = loss_class(actor, value_net, gamma=cfg.gamma, **loss_kwargs)
    target_net_updater = make_target_updater(cfg, loss_module)
    return loss_module, target_net_updater


def make_dqn_loss(model, cfg) -> Tuple[DQNLoss, Optional[_TargetNetUpdate]]:
    """Builds the DQN loss module."""
    loss_kwargs = {}
    if cfg.distributional:
        loss_class = DistributionalDQNLoss
    else:
        loss_kwargs.update({"loss_function": cfg.loss_function})
        loss_class = DQNLoss
    if cfg.loss not in ("single", "double"):
        raise NotImplementedError
    loss_kwargs.update({"delay_value": cfg.loss == "double"})
    loss_module = loss_class(model, gamma=cfg.gamma, **loss_kwargs)
    target_net_updater = make_target_updater(cfg, loss_module)
    return loss_module, target_net_updater


def make_ppo_loss(model, cfg) -> PPOLoss:
    """Builds the PPO loss module."""
    loss_dict = {
        "clip": ClipPPOLoss,
        "kl": KLPENPPOLoss,
        "base": PPOLoss,
        "": PPOLoss,
    }
    actor_model = model.get_policy_operator()
    critic_model = model.get_value_operator()

    advantage = GAE(
        cfg.gamma,
        cfg.lmbda,
        value_network=critic_model,
        average_rewards=True,
        gradient_mode=False,
    )
    loss_module = loss_dict[cfg.loss](
        actor=actor_model,
        critic=critic_model,
        advantage_module=advantage,
        loss_critic_type=cfg.loss_function,
        entropy_coef=cfg.entropy_coef,
    )
    return loss_module


@dataclass
class LossConfig:
    loss: str = "double"
    # whether double or single SAC loss should be used. Default=double
    hard_update: bool = False
    # whether soft-update should be used with double SAC loss (default) or hard updates.
    loss_function: str = "smooth_l1"
    # loss function for the value network. Either one of l1, l2 or smooth_l1 (default).
    value_network_update_interval: int = 1000
    # how often the target value network weights are updated (in number of updates).
    # If soft-updates are used, the value is translated into a moving average decay by using
    # the formula decay=1-1/cfg.value_network_update_interval. Default=1000
    gamma: float = 0.99
    # Decay factor for return computation. Default=0.99.
    num_q_values: int = 2
    # As suggested in the original SAC paper and in https://arxiv.org/abs/1802.09477, we can
    # use two (or more!) different qvalue networks trained independently and choose the lowest value
    # predicted to predict the state action value. This can be disabled by using this flag.
    # REDQ uses an arbitrary number of Q-value functions to speed up learning in MF contexts.
    target_entropy: Any = None
    # Target entropy for the policy distribution. Default is None (auto calculated as the `target_entropy = -action_dim`)


@dataclass
class PPOLossConfig:
    loss: str = "clip"
    # PPO loss class, either clip or kl or base/<empty>. Default=clip
    gamma: float = 0.99
    # Decay factor for return computation. Default=0.99.
    lmbda: float = 0.95
    # lambda factor in GAE (using 'lambda' as attribute is prohibited in python, hence the misspelling)
    entropy_coef: float = 1e-3
    # Entropy factor for the PPO loss
    loss_function: str = "smooth_l1"
    # loss function for the value network. Either one of l1, l2 or smooth_l1 (default).
