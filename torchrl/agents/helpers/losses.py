from argparse import ArgumentParser, Namespace

__all__ = [
    "make_sac_loss",
    "parser_loss_args_offline",
    "make_dqn_loss",
    "make_ddpg_loss",
    "make_target_updater",
    "make_ppo_loss",
    "parser_loss_args_ppo",
]

from typing import Optional, Tuple

from torchrl.objectives import (
    DoubleSACLoss,
    SACLoss,
    SoftUpdate,
    HardUpdate,
    DDPGLoss,
    DoubleDDPGLoss,
    DistributionalDQNLoss,
    DistributionalDoubleDQNLoss,
    DQNLoss,
    DoubleDQNLoss,
    PPOLoss,
    ClipPPOLoss,
    KLPENPPOLoss,
    GAE,
)
from torchrl.objectives.costs.common import _LossModule
from torchrl.objectives.costs.utils import _TargetNetUpdate


def make_target_updater(
    args: Namespace, loss_module: _LossModule
) -> Optional[_TargetNetUpdate]:
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
        assert not args.hard_update, (
            "hard/soft-update are supposed to be used with double SAC loss. "
            "Consider using --loss=double or discarding the hard_update flag."
        )
        target_net_updater = None
    return target_net_updater


def make_sac_loss(model, args) -> Tuple[SACLoss, Optional[_TargetNetUpdate]]:
    loss_kwargs = {}
    if args.distributional:
        raise NotImplementedError
    else:
        loss_kwargs.update({"loss_function": args.loss_function})
        if args.loss == "double":
            loss_class = DoubleSACLoss
            loss_kwargs.update(
                {
                    "delay_actor": False,
                    "delay_qvalue": False,
                }
            )
        else:
            loss_class = SACLoss
    qvalue_model_bis = None
    if len(model) == 3:
        actor_model, qvalue_model, value_model = model
    else:
        actor_model, qvalue_model, qvalue_model_bis, value_model = model

    loss_module = loss_class(
        actor_network=actor_model,
        qvalue_network=qvalue_model,
        value_network=value_model,
        qvalue_network_bis=qvalue_model_bis,
        gamma=args.gamma,
        **loss_kwargs
    )
    target_net_updater = make_target_updater(args, loss_module)
    return loss_module, target_net_updater


def make_ddpg_loss(model, args) -> Tuple[DDPGLoss, Optional[_TargetNetUpdate]]:
    actor, value_net = model
    loss_kwargs = {}
    if args.distributional:
        raise NotImplementedError
    else:
        loss_kwargs.update({"loss_function": args.loss_function})
        if args.loss == "single":
            loss_class = DDPGLoss
        elif args.loss == "double":
            loss_class = DoubleDDPGLoss
        else:
            raise NotImplementedError
    loss_module = loss_class(actor, value_net, gamma=args.gamma, **loss_kwargs)
    target_net_updater = make_target_updater(args, loss_module)
    return loss_module, target_net_updater


def make_dqn_loss(model, args) -> Tuple[DQNLoss, Optional[_TargetNetUpdate]]:
    loss_kwargs = {}
    if args.distributional:
        if args.loss == "single":
            loss_class = DistributionalDQNLoss
        elif args.loss == "double":
            loss_class = DistributionalDoubleDQNLoss
        else:
            raise NotImplementedError
    else:
        loss_kwargs.update({"loss_function": args.loss_function})
        if args.loss == "single":
            loss_class = DQNLoss
        elif args.loss == "double":
            loss_class = DoubleDQNLoss
        else:
            raise NotImplementedError
    loss_module = loss_class(model, gamma=args.gamma, **loss_kwargs)
    target_net_updater = make_target_updater(args, loss_module)
    return loss_module, target_net_updater


def make_ppo_loss(model, args) -> PPOLoss:
    loss_dict = {
        "clip": ClipPPOLoss,
        "kl": KLPENPPOLoss,
        "base": PPOLoss,
        "": PPOLoss,
    }
    actor_model = model.get_policy_operator()
    critic_model = model.get_value_operator()

    advantage = GAE(args.gamma, args.lamda, critic=critic_model, average_rewards=True)
    loss_module = loss_dict[args.loss](
        actor=actor_model,
        critic=critic_model,
        advantage_module=advantage,
        loss_critic_type=args.loss_function,
        entropy_factor=args.entropy_factor,
    )
    return loss_module


def parser_loss_args_offline(parser: ArgumentParser) -> ArgumentParser:
    """
    To be used for DQN, DDPG, SAC
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
        action="store_true",
        help="whether soft-update should be used with double SAC loss (default) or hard updates.",
    )
    parser.add_argument(
        "--loss_function",
        type=str,
        default="smooth_l1",
        choices=["l1", "l2", "smooth_l1"],
        help="loss function for the value network. Either one of l1, l2 or smooth_l1 (default).",
    )
    parser.add_argument(
        "--value_network_update_interval",
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
    parser.add_argument(
        "--distributional",
        action="store_true",
        help="whether a distributional loss should be used (TODO: not implemented yet).",
    )
    parser.add_argument(
        "--atoms",
        type=int,
        default=51,
        help="number of atoms used for the distributional loss (TODO)",
    )

    return parser


def parser_loss_args_ppo(parser: ArgumentParser) -> ArgumentParser:
    """
    To be used for PPO
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
        "--lamda",
        default=0.95,
        type=float,
        help="lambda factor in GAE (using 'lambda' as attribute is prohibited in python, "
        "hence the misspelling)",
    )
    parser.add_argument(
        "--entropy_factor",
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