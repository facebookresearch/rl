# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from argparse import ArgumentParser, Namespace

__all__ = [
    "make_sac_loss",
    "make_dqn_loss",
    "make_ddpg_loss",
    "make_target_updater",
    "make_ppo_loss",
    "make_redq_loss",
    "parser_loss_args",
    "parser_loss_args_ppo",
]

from typing import Optional, Tuple

from torchrl.objectives import (
    ClipPPOLoss,
    DDPGLoss,
    DistributionalDQNLoss,
    DQNLoss,
    HardUpdate,
    KLPENPPOLoss,
    PPOLoss,
    SACLoss,
    SoftUpdate,
)
from torchrl.objectives.common import _LossModule
from torchrl.objectives.redq import REDQLoss

# from torchrl.objectives.costs.redq import REDQLoss
from torchrl.objectives.utils import _TargetNetUpdate
from torchrl.objectives.value.advantages import GAE


def make_target_updater(
    args: Namespace, loss_module: _LossModule
) -> Optional[_TargetNetUpdate]:
    """Builds a target network weight update object."""
    if args.loss == "double":
        if not args.hard_update:
            target_net_updater = SoftUpdate(
                loss_module, 1 - 1 / args.value_network_update_interval
            )
        else:
            target_net_updater = HardUpdate(
                loss_module, args.value_network_update_interval
            )
        # assert len(target_net_updater.net_pairs) == 3, "length of target_net_updater nets should be 3"
        target_net_updater.init_()
    else:
        if args.hard_update:
            raise RuntimeError(
                "hard/soft-update are supposed to be used with double SAC loss. "
                "Consider using --loss=double or discarding the hard_update flag."
            )
        target_net_updater = None
    return target_net_updater


def make_sac_loss(model, args) -> Tuple[SACLoss, Optional[_TargetNetUpdate]]:
    """Builds the SAC loss module."""
    loss_kwargs = {}
    if hasattr(args, "distributional") and args.distributional:
        raise NotImplementedError
    else:
        loss_kwargs.update({"loss_function": args.loss_function})
        loss_kwargs.update(
            {
                "target_entropy": args.target_entropy
                if args.target_entropy is not None
                else "auto"
            }
        )
        loss_class = SACLoss
        if args.loss == "double":
            loss_kwargs.update(
                {
                    "delay_actor": False,
                    "delay_qvalue": True,
                    "delay_value": True,
                }
            )
        elif args.loss == "single":
            loss_kwargs.update(
                {
                    "delay_actor": False,
                    "delay_qvalue": False,
                    "delay_value": False,
                }
            )
        else:
            raise NotImplementedError(
                f"args.loss {args.loss} unsupported. Consider chosing from 'double' or 'single'"
            )

    actor_model, qvalue_model, value_model = model

    loss_module = loss_class(
        actor_network=actor_model,
        qvalue_network=qvalue_model,
        value_network=value_model,
        num_qvalue_nets=args.num_q_values,
        gamma=args.gamma,
        **loss_kwargs,
    )
    target_net_updater = make_target_updater(args, loss_module)
    return loss_module, target_net_updater


def make_redq_loss(model, args) -> Tuple[REDQLoss, Optional[_TargetNetUpdate]]:
    """Builds the REDQ loss module."""
    loss_kwargs = {}
    if hasattr(args, "distributional") and args.distributional:
        raise NotImplementedError
    else:
        loss_kwargs.update({"loss_function": args.loss_function})
        loss_kwargs.update({"delay_qvalue": args.loss == "double"})
        loss_class = REDQLoss
    actor_model, qvalue_model = model

    loss_module = loss_class(
        actor_network=actor_model,
        qvalue_network=qvalue_model,
        num_qvalue_nets=args.num_q_values,
        gamma=args.gamma,
        gSDE=args.gSDE,
        **loss_kwargs,
    )
    target_net_updater = make_target_updater(args, loss_module)
    return loss_module, target_net_updater


def make_ddpg_loss(model, args) -> Tuple[DDPGLoss, Optional[_TargetNetUpdate]]:
    """Builds the DDPG loss module."""
    actor, value_net = model
    loss_kwargs = {}
    if args.distributional:
        raise NotImplementedError
    else:
        loss_kwargs.update({"loss_function": args.loss_function})
        loss_class = DDPGLoss
    if args.loss not in ("single", "double"):
        raise NotImplementedError
    double_loss = args.loss == "double"
    loss_kwargs.update({"delay_actor": double_loss, "delay_value": double_loss})
    loss_module = loss_class(actor, value_net, gamma=args.gamma, **loss_kwargs)
    target_net_updater = make_target_updater(args, loss_module)
    return loss_module, target_net_updater


def make_dqn_loss(model, args) -> Tuple[DQNLoss, Optional[_TargetNetUpdate]]:
    """Builds the DQN loss module."""
    loss_kwargs = {}
    if args.distributional:
        loss_class = DistributionalDQNLoss
    else:
        loss_kwargs.update({"loss_function": args.loss_function})
        loss_class = DQNLoss
    if args.loss not in ("single", "double"):
        raise NotImplementedError
    loss_kwargs.update({"delay_value": args.loss == "double"})
    loss_module = loss_class(model, gamma=args.gamma, **loss_kwargs)
    target_net_updater = make_target_updater(args, loss_module)
    return loss_module, target_net_updater


def make_ppo_loss(model, args) -> PPOLoss:
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
        args.gamma,
        args.lmbda,
        value_network=critic_model,
        average_rewards=True,
        gradient_mode=False,
    )
    loss_module = loss_dict[args.loss](
        actor=actor_model,
        critic=critic_model,
        advantage_module=advantage,
        loss_critic_type=args.loss_function,
        entropy_coef=args.entropy_coef,
    )
    return loss_module


def parser_loss_args(parser: ArgumentParser, algorithm: str) -> ArgumentParser:
    """
    Populates the argument parser to build the off-policy loss function (REDQ, SAC, DDPG, DQN).

    Args:
        parser (ArgumentParser): parser to be populated.
        algorithm (str): one of `"DDPG"`, `"SAC"`, `"REDQ"`, `"DQN"`

    """
    parser.add_argument(
        "--loss",
        type=str,
        default="double",
        choices=["double", "single"],
        help="whether double or single SAC loss should be used. Default=double",
    )
    parser.add_argument(
        "--hard_update",
        "--hard-update",
        action="store_true",
        help="whether soft-update should be used with double SAC loss (default) or hard updates.",
    )
    parser.add_argument(
        "--loss_function",
        "--loss-function",
        type=str,
        default="smooth_l1",
        choices=["l1", "l2", "smooth_l1"],
        help="loss function for the value network. Either one of l1, l2 or smooth_l1 (default).",
    )
    parser.add_argument(
        "--value_network_update_interval",
        "--value-network-update-interval",
        type=int,
        default=1000,
        help="how often the target value network weights are updated (in number of updates)."
        "If soft-updates are used, the value is translated into a moving average decay by using "
        "the formula decay=1-1/args.value_network_update_interval. Default=1000",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Decay factor for return computation. Default=0.99.",
    )
    if algorithm in ("SAC", "REDQ"):
        parser.add_argument(
            "--num_q_values",
            "--num-q-values",
            default=2,
            type=int,
            help="As suggested in the original SAC paper and in https://arxiv.org/abs/1802.09477, we can "
            "use two (or more!) different qvalue networks trained independently and choose the lowest value "
            "predicted to predict the state action value. This can be disabled by using this flag."
            "REDQ uses an arbitrary number of Q-value functions to speed up learning in MF contexts.",
        )

        parser.add_argument(
            "--target_entropy",
            "--target-entropy",
            default=None,
            type=float,
            help="Target entropy for the policy distribution. Default is None (auto calculated as"
            "the `target_entropy = -action_dim`)",
        )

    return parser


def parser_loss_args_ppo(parser: ArgumentParser) -> ArgumentParser:
    """
    Populates the argument parser to build the PPO loss function.

    Args:
        parser (ArgumentParser): parser to be populated.

    """
    parser.add_argument(
        "--loss",
        type=str,
        default="clip",
        choices=["clip", "kl", "base", ""],
        help="PPO loss class, either clip or kl or base/<empty>. Default=clip",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Decay factor for return computation. Default=0.99.",
    )
    parser.add_argument(
        "--lmbda",
        default=0.95,
        type=float,
        help="lambda factor in GAE (using 'lambda' as attribute is prohibited in python, "
        "hence the misspelling)",
    )
    parser.add_argument(
        "--entropy_coef",
        "--entropy-coef",
        type=float,
        default=1e-3,
        help="Entropy factor for the PPO loss",
    )
    parser.add_argument(
        "--loss_function",
        type=str,
        default="smooth_l1",
        choices=["l1", "l2", "smooth_l1"],
        help="loss function for the value network. Either one of l1, l2 or smooth_l1 (default).",
    )

    return parser
