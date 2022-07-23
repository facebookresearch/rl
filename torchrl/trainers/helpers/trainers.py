# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional, Union, List
from warnings import warn

from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from torchrl.collectors.collectors import _DataCollector
from torchrl.data import ReplayBuffer
from torchrl.envs.common import _EnvClass
from torchrl.modules import TensorDictModule, TensorDictModuleWrapper, reset_noise
from torchrl.objectives.costs.common import LossModule
from torchrl.objectives.costs.utils import _TargetNetUpdate
from torchrl.trainers.loggers import Logger
from torchrl.trainers.trainers import (
    Trainer,
    SelectKeys,
    ReplayBufferTrainer,
    LogReward,
    RewardNormalizer,
    mask_batch,
    BatchSubSampler,
    UpdateWeights,
    Recorder,
    CountFramesLog,
)

OPTIMIZERS = {
    "adam": optim.Adam,
    "sgd": optim.SGD,
    "adamax": optim.Adamax,
}

__all__ = [
    "make_trainer",
]


@dataclass
class TrainerConfig:
    optim_steps_per_batch: int = 500
    # Number of optimization steps in between two collection of data. See frames_per_batch below.
    optimizer: str = "adam"
    # Optimizer to be used.
    lr_scheduler: str = "cosine"
    # LR scheduler.
    selected_keys: Optional[List] = None
    # a list of strings that indicate the data that should be kept from the data collector. Since storing and
    # retrieving information from the replay buffer does not come for free, limiting the amount of data
    # passed to it can improve the algorithm performance.
    batch_size: int = 256
    # batch size of the TensorDict retrieved from the replay buffer. Default=256.
    log_interval: int = 10000
    # logging interval, in terms of optimization steps. Default=10000.
    lr: float = 3e-4
    # Learning rate used for the optimizer. Default=3e-4.
    weight_decay: float = 0.0
    # Weight-decay to be used with the optimizer. Default=0.0.
    clip_norm: float = 1000.0
    # value at which the total gradient norm / single derivative should be clipped. Default=1000.0
    clip_grad_norm: bool = False
    # if called, the gradient will be clipped based on its L2 norm. Otherwise, single gradient values will be clipped to the desired threshold.
    normalize_rewards_online: bool = False
    # Computes the running statistics of the rewards and normalizes them before they are passed to the loss module.
    normalize_rewards_online_scale: float = 1.0
    # Final value of the normalized rewards.
    sub_traj_len: int = -1
    # length of the trajectories that sub-samples must have in online settings.


def make_trainer(
    collector: _DataCollector,
    loss_module: LossModule,
    recorder: Optional[_EnvClass] = None,
    target_net_updater: Optional[_TargetNetUpdate] = None,
    policy_exploration: Optional[
        Union[TensorDictModuleWrapper, TensorDictModule]
    ] = None,
    replay_buffer: Optional[ReplayBuffer] = None,
    logger: Optional[Logger] = None,
    cfg: "DictConfig" = None,
) -> Trainer:
    """Creates a Trainer instance given its constituents.

    Args:
        collector (_DataCollector): A data collector to be used to collect data.
        loss_module (LossModule): A TorchRL loss module
        recorder (_EnvClass, optional): a recorder environment. If None, the trainer will train the policy without
            testing it.
        target_net_updater (_TargetNetUpdate, optional): A target network update object.
        policy_exploration (TDModule or TensorDictModuleWrapper, optional): a policy to be used for recording and exploration
            updates (should be synced with the learnt policy).
        replay_buffer (ReplayBuffer, optional): a replay buffer to be used to collect data.
        logger (Logger, optional): a Logger to be used for logging.
        cfg (DictConfig, optional): a DictConfig containing the arguments of the script. If None, the default
            arguments are used.

    Returns:
        A trainer built with the input objects. The optimizer is built by this helper function using the cfg provided.

    Examples:
        >>> import torch
        >>> import tempfile
        >>> from torchrl.trainers.loggers import TensorboardLogger
        >>> from torchrl.trainers import Trainer
        >>> from torchrl.envs import EnvCreator
        >>> from torchrl.collectors.collectors import SyncDataCollector
        >>> from torchrl.data import TensorDictReplayBuffer
        >>> from torchrl.envs import GymEnv
        >>> from torchrl.modules import TensorDictModuleWrapper, TensorDictModule, ValueOperator, EGreedyWrapper
        >>> from torchrl.objectives.costs.common import LossModule
        >>> from torchrl.objectives.costs.utils import _TargetNetUpdate
        >>> from torchrl.objectives import DDPGLoss
        >>> env_maker = EnvCreator(lambda: GymEnv("Pendulum-v0"))
        >>> env_proof = env_maker()
        >>> obs_spec = env_proof.observation_spec
        >>> action_spec = env_proof.action_spec
        >>> net = torch.nn.Linear(env_proof.observation_spec.shape[-1], action_spec.shape[-1])
        >>> net_value = torch.nn.Linear(env_proof.observation_spec.shape[-1], 1)  # for the purpose of testing
        >>> policy = TensorDictModule(action_spec, net, in_keys=["observation"], out_keys=["action"])
        >>> value = ValueOperator(net_value, in_keys=["observation"], out_keys=["state_action_value"])
        >>> collector = SyncDataCollector(env_maker, policy, total_frames=100)
        >>> loss_module = DDPGLoss(policy, value, gamma=0.99)
        >>> recorder = env_proof
        >>> target_net_updater = None
        >>> policy_exploration = EGreedyWrapper(policy)
        >>> replay_buffer = TensorDictReplayBuffer(1000)
        >>> dir = tempfile.gettempdir()
        >>> logger = TensorboardLogger(exp_name=dir)
        >>> trainer = make_trainer(collector, loss_module, recorder, target_net_updater, policy_exploration,
        ...    replay_buffer, logger)
        >>> print(trainer)

    """
    if cfg is None:
        warn(
            "Getting default cfg for the trainer. "
            "This should be only used for debugging."
        )
        cfg = TrainerConfig()
        cfg.frame_skip = 1
        cfg.total_frames = 1000
        cfg.record_frames = 10
        cfg.record_interval = 10

    optimizer_kwargs = {} if cfg.optimizer != "adam" else {"betas": (0.0, 0.9)}
    optimizer = OPTIMIZERS[cfg.optimizer](
        loss_module.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        **optimizer_kwargs,
    )
    device = next(loss_module.parameters()).device
    if cfg.lr_scheduler == "cosine":
        optim_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=int(
                cfg.total_frames / cfg.frames_per_batch * cfg.optim_steps_per_batch
            ),
        )
    elif cfg.lr_scheduler == "":
        optim_scheduler = None
    else:
        raise NotImplementedError(f"lr scheduler {cfg.lr_scheduler}")

    print(
        f"collector = {collector}; \n"
        f"loss_module = {loss_module}; \n"
        f"recorder = {recorder}; \n"
        f"target_net_updater = {target_net_updater}; \n"
        f"policy_exploration = {policy_exploration}; \n"
        f"replay_buffer = {replay_buffer}; \n"
        f"logger = {logger}; \n"
        f"cfg = {cfg}; \n"
    )

    if logger is not None:
        # log hyperparams
        logger.log_hparams(cfg)

    trainer = Trainer(
        collector=collector,
        frame_skip=cfg.frame_skip,
        total_frames=cfg.total_frames * cfg.frame_skip,
        loss_module=loss_module,
        optimizer=optimizer,
        logger=logger,
        optim_steps_per_batch=cfg.optim_steps_per_batch,
        clip_grad_norm=cfg.clip_grad_norm,
        clip_norm=cfg.clip_norm,
    )

    if hasattr(cfg, "noisy") and cfg.noisy:
        trainer.register_op("pre_optim_steps", lambda: loss_module.apply(reset_noise))

    if cfg.selected_keys:
        trainer.register_op("batch_process", SelectKeys(cfg.selected_keys))
    trainer.register_op("batch_process", lambda batch: batch.cpu())

    if replay_buffer is not None:
        # replay buffer is used 2 or 3 times: to register data, to sample
        # data and to update priorities
        rb_trainer = ReplayBufferTrainer(
            replay_buffer, cfg.batch_size, memmap=False, device=device
        )

        trainer.register_op("batch_process", rb_trainer.extend)
        trainer.register_op("process_optim_batch", rb_trainer.sample)
        trainer.register_op("post_loss", rb_trainer.update_priority)
    else:
        # trainer.register_op("batch_process", mask_batch)
        trainer.register_op(
            "process_optim_batch",
            BatchSubSampler(batch_size=cfg.batch_size, sub_traj_len=cfg.sub_traj_len),
        )

    if optim_scheduler is not None:
        trainer.register_op("post_optim", optim_scheduler.step)

    if target_net_updater is not None:
        trainer.register_op("post_optim", target_net_updater.step)

    if cfg.normalize_rewards_online:
        # if used the running statistics of the rewards are computed and the
        # rewards used for training will be normalized based on these.
        reward_normalizer = RewardNormalizer(scale=cfg.normalize_rewards_online_scale)
        trainer.register_op("batch_process", reward_normalizer.update_reward_stats)
        trainer.register_op("process_optim_batch", reward_normalizer.normalize_reward)

    if policy_exploration is not None and hasattr(policy_exploration, "step"):
        trainer.register_op(
            "post_steps", policy_exploration.step, frames=cfg.frames_per_batch
        )

    trainer.register_op(
        "post_steps_log", lambda *cfg: {"lr": optimizer.param_groups[0]["lr"]}
    )

    if recorder is not None:
        recorder_obj = Recorder(
            record_frames=cfg.record_frames,
            frame_skip=cfg.frame_skip,
            policy_exploration=policy_exploration,
            recorder=recorder,
            record_interval=cfg.record_interval,
            log_keys=cfg.recorder_log_keys,
        )
        trainer.register_op(
            "post_steps_log",
            recorder_obj,
        )
        recorder_obj(None)
        recorder_obj_explore = Recorder(
            record_frames=cfg.record_frames,
            frame_skip=cfg.frame_skip,
            policy_exploration=policy_exploration,
            recorder=recorder,
            record_interval=cfg.record_interval,
            exploration_mode="random",
            suffix="exploration",
            out_keys={"reward": "r_evaluation_exploration"},
        )
        trainer.register_op(
            "post_steps_log",
            recorder_obj_explore,
        )
        recorder_obj_explore(None)

    trainer.register_op(
        "post_steps", UpdateWeights(collector, update_weights_interval=1)
    )

    trainer.register_op("pre_steps_log", LogReward())
    trainer.register_op("pre_steps_log", CountFramesLog(frame_skip=cfg.frame_skip))

    return trainer
