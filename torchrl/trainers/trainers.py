# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import pathlib
import warnings
from collections import OrderedDict, defaultdict
from textwrap import indent
from typing import Callable, Dict, Optional, Union, Sequence, Tuple, Type, List

import numpy as np
import torch.nn
from torch import nn, optim

from torchrl import KeyDependentDefaultDict

try:
    from tqdm import tqdm

    _has_tqdm = True
except ImportError:
    _has_tqdm = False

from torchrl.collectors.collectors import _DataCollector
from torchrl.data import (
    ReplayBuffer,
    TensorDictPrioritizedReplayBuffer,
    TensorDictReplayBuffer,
)
from torchrl.data.tensordict.tensordict import _TensorDict
from torchrl.data.utils import expand_right, DEVICE_TYPING
from torchrl.envs.common import _EnvClass
from torchrl.envs.transforms import TransformedEnv
from torchrl.envs.utils import set_exploration_mode
from torchrl.modules import TensorDictModule
from torchrl.objectives.costs.common import _LossModule

REPLAY_BUFFER_CLASS = {
    "prioritized": TensorDictPrioritizedReplayBuffer,
    "circular": TensorDictReplayBuffer,
}

WRITER_METHODS = {
    "grad_norm": "add_scalar",
    "loss": "add_scalar",
}

__all__ = [
    "Trainer",
    "BatchSubSampler",
    "CountFramesLog",
    "LogReward",
    "Recorder",
    "ReplayBuffer",
    "RewardNormalizer",
    "SelectKeys",
    "UpdateWeights",
]

TYPE_DESCR = {float: "4.4f", int: ""}


class Trainer:
    """A generic Trainer class.

    A trainer is responsible for collecting data and training the model.
    To keep the class as versatile as possible, Trainer does not construct any
    of its specific operations: they all must be hooked at specific points in
    the training loop.

    To build a Trainer, one needs an iterable data source (a `collector`), a
    loss module and an optimizer.

    Args:
        collector (Sequence[_TensorDict]): An iterable returning batches of
            data in a TensorDict form of shape [batch x time steps].
        total_frames (int): Total number of frames to be collected during
            training.
        loss_module (_LossModule): A module that reads TensorDict batches
            (possibly sampled from a replay buffer) and return a loss
            TensorDict where every key points to a different loss component.
        optimizer (optim.Optimizer): An optimizer that trains the parameters
            of the model.
        writer (SummaryWriter, optional): a Tensorboard summary writer for
            logging purposes.
        optim_steps_per_batch (int, optional): number of optimization steps
            per collection of data. An trainer works as follows: a main loop
            collects batches of data (epoch loop), and a sub-loop (training
            loop) performs model updates in between two collections of data.
            Default is 500
        clip_grad_norm (bool, optional): If True, the gradients will be clipped
            based on the total norm of the model parameters. If False,
            all the partial derivatives will be clamped to
            (-clip_norm, clip_norm). Default is `True`.
        clip_norm (Number, optional): value to be used for clipping gradients.
            Default is 100.0.
        progress_bar (bool, optional): If True, a progress bar will be
            displayed using tqdm. If tqdm is not installed, this option
            won't have any effect. Default is `True`
        seed (int, optional): Seed to be used for the collector, pytorch and
            numpy. Default is 42.
        save_trainer_interval (int, optional): How often the trainer should be
            saved to disk. Default is 10000.
        save_trainer_file (path, optional): path where to save the trainer.
            Default is None (no saving)
    """

    # trackers
    _optim_count: int = 0
    _collected_frames: int = 0
    _last_log: dict = {}
    _last_save: int = 0
    _log_interval: int = 10000

    def __init__(
        self,
        collector: _DataCollector,
        total_frames: int,
        frame_skip: int,
        loss_module: Union[_LossModule, Callable[[_TensorDict], _TensorDict]],
        optimizer: optim.Optimizer,
        writer: Optional["SummaryWriter"] = None,
        optim_steps_per_batch: int = 500,
        clip_grad_norm: bool = True,
        clip_norm: float = 100.0,
        progress_bar: bool = True,
        seed: int = 42,
        save_trainer_interval: int = 10000,
        save_trainer_file: Optional[Union[str, pathlib.Path]] = None,
    ) -> None:

        # objects
        self.frame_skip = frame_skip
        self.collector = collector
        self.loss_module = loss_module
        self.optimizer = optimizer
        self.writer = writer
        self._params = []
        for p in self.optimizer.param_groups:
            self._params += p["params"]

        # seeding
        self.seed = seed
        self.set_seed()

        # constants
        self.optim_steps_per_batch = optim_steps_per_batch
        self.total_frames = total_frames
        self.clip_grad_norm = clip_grad_norm
        self.clip_norm = clip_norm
        if progress_bar and not _has_tqdm:
            warnings.warn(
                "tqdm library not found. "
                "Consider installing tqdm to use the Trainer progress bar."
            )
        self.progress_bar = progress_bar and _has_tqdm
        self.save_trainer_interval = save_trainer_interval
        self.save_trainer_file = save_trainer_file

        self._log_dict = defaultdict(lambda: [])

        self._batch_process_ops = []
        self._post_steps_ops = []
        self._post_steps_log_ops = []
        self._pre_steps_log_ops = []
        self._post_optim_log_ops = []
        self._pre_optim_ops = []
        self._post_loss_ops = []
        self._process_optim_batch_ops = []
        self._post_optim_ops = []

    def save_trainer(self, force_save: bool = False) -> None:
        _save = force_save
        if self.save_trainer_file is not None:
            if (self._collected_frames - self._last_save) > self.save_trainer_interval:
                self._last_save = self._collected_frames
                _save = True
        if _save and self.save_trainer_file:
            torch.save(self.state_dict(), self.save_trainer_file)

    def load_from_file(self, file: Union[str, pathlib.Path]) -> Trainer:
        loaded_dict: OrderedDict = torch.load(file)

        # checks that keys match
        expected_keys = {
            "env",
            "loss_module",
            "_last_log",
            "_last_save",
            "_optim_count",
        }
        actual_keys = set(loaded_dict.keys())
        if len(actual_keys.difference(expected_keys)) or len(
            expected_keys.difference(actual_keys)
        ):
            raise RuntimeError(
                f"Expected keys {expected_keys} in the loaded file but got"
                f" {actual_keys}"
            )
        self.collector.load_state_dict(loaded_dict["env"])
        self.model.load_state_dict(loaded_dict["model"])
        for key in [
            "_last_log",
            "_last_save",
            "_optim_count",
        ]:
            setattr(self, key, loaded_dict[key])
        return self

    def set_seed(self):
        seed = self.collector.set_seed(self.seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

    def state_dict(self) -> Dict:
        state_dict = OrderedDict(
            env=self.collector.state_dict(),
            loss_module=self.loss_module.state_dict(),
            _collected_frames=self._collected_frames,
            _last_log=self._last_log,
            _last_save=self._last_save,
            _optim_count=self._optim_count,
        )
        return state_dict

    def load_state_dict(self, state_dict: Dict) -> None:
        model_state_dict = state_dict["loss_module"]
        env_state_dict = state_dict["env"]
        self.loss_module.load_state_dict(model_state_dict)
        self.collector.load_state_dict(env_state_dict)

    @property
    def collector(self) -> _DataCollector:
        return self._collector

    @collector.setter
    def collector(self, collector: _DataCollector) -> None:
        self._collector = collector

    def register_op(self, dest: str, op: Callable, **kwargs) -> None:
        if dest == "batch_process":
            _check_input_output_typehint(op, input=_TensorDict, output=_TensorDict)
            self._batch_process_ops.append((op, kwargs))

        elif dest == "pre_optim_steps":
            _check_input_output_typehint(op, input=None, output=None)
            self._pre_optim_ops.append((op, kwargs))

        elif dest == "process_optim_batch":
            _check_input_output_typehint(op, input=_TensorDict, output=_TensorDict)
            self._process_optim_batch_ops.append((op, kwargs))

        elif dest == "post_loss":
            _check_input_output_typehint(op, input=_TensorDict, output=_TensorDict)
            self._post_loss_ops.append((op, kwargs))

        elif dest == "post_steps":
            _check_input_output_typehint(op, input=None, output=None)
            self._post_steps_ops.append((op, kwargs))

        elif dest == "post_optim":
            _check_input_output_typehint(op, input=None, output=None)
            self._post_optim_ops.append((op, kwargs))

        elif dest == "pre_steps_log":
            _check_input_output_typehint(
                op, input=_TensorDict, output=Tuple[str, float]
            )
            self._pre_steps_log_ops.append((op, kwargs))

        elif dest == "post_steps_log":
            _check_input_output_typehint(
                op, input=_TensorDict, output=Tuple[str, float]
            )
            self._post_steps_log_ops.append((op, kwargs))

        elif dest == "post_optim_log":
            _check_input_output_typehint(
                op, input=_TensorDict, output=Tuple[str, float]
            )
            self._post_optim_log_ops.append((op, kwargs))

        else:
            raise RuntimeError(
                f"The hook collection {dest} is not recognised. Choose from:"
                f"(batch_process, pre_steps, pre_step, post_loss, post_steps, "
                f"post_steps_log, post_optim_log)"
            )

    # Process batch
    def _process_batch_hook(self, batch: _TensorDict) -> _TensorDict:
        for op, kwargs in self._batch_process_ops:
            out = op(batch, **kwargs)
            if isinstance(out, _TensorDict):
                batch = out
        return batch

    def _post_steps_hook(self) -> None:
        for op, kwargs in self._post_steps_ops:
            op(**kwargs)

    def _post_optim_log(self, batch: _TensorDict) -> None:
        for op, kwargs in self._post_optim_log_ops:
            result = op(batch, **kwargs)
            if result is not None:
                self._log(**result)

    def _pre_optim_hook(self):
        for op, kwargs in self._pre_optim_ops:
            op(**kwargs)

    def _process_optim_batch_hook(self, batch):
        for op, kwargs in self._process_optim_batch_ops:
            out = op(batch, **kwargs)
            if isinstance(out, _TensorDict):
                batch = out
        return batch

    def _post_loss_hook(self, batch):
        for op, kwargs in self._post_loss_ops:
            out = op(batch, **kwargs)
            if isinstance(out, _TensorDict):
                batch = out
        return batch

    def _post_optim_hook(self):
        for op, kwargs in self._post_optim_ops:
            op(**kwargs)

    def _pre_steps_log_hook(self, batch: _TensorDict) -> None:
        for op, kwargs in self._pre_steps_log_ops:
            result = op(batch, **kwargs)
            if result is not None:
                self._log(**result)

    def _post_steps_log_hook(self, batch: _TensorDict) -> None:
        for op, kwargs in self._post_steps_log_ops:
            result = op(batch, **kwargs)
            if result is not None:
                self._log(**result)

    def train(self):
        if self.progress_bar:
            self._pbar = tqdm(total=self.total_frames)
            self._pbar_str = dict()

        self.collected_frames = 0

        for i, batch in enumerate(self.collector):
            batch = self._process_batch_hook(batch)
            self._pre_steps_log_hook(batch)
            current_frames = (
                batch.get("mask", torch.tensor(batch.numel())).sum().item()
                * self.frame_skip
            )
            self.collected_frames += current_frames

            if self.collected_frames > self.collector.init_random_frames:
                self.optim_steps(batch)
            self._post_steps_hook()

            self._post_steps_log_hook(batch)

            if self.progress_bar:
                self._pbar.update(current_frames)
                self._pbar_description()

            if self.collected_frames > self.total_frames:
                break
        self.save_trainer(force_save=True)

    def __del__(self):
        self.collector.shutdown()

    def shutdown(self):
        print("shutting down collector")
        self.collector.shutdown()

    def _optimizer_step(self, losses_td: _TensorDict) -> _TensorDict:
        # sum all keys that start with 'loss_'
        loss = sum([item for key, item in losses_td.items() if key.startswith("loss")])
        loss.backward()

        grad_norm = self._grad_clip()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return losses_td.detach().set("grad_norm", grad_norm)

    def optim_steps(self, batch: _TensorDict) -> None:
        # average_grad_norm = 0.0
        average_losses = None

        self._pre_optim_hook()

        for j in range(self.optim_steps_per_batch):
            self._optim_count += 1

            sub_batch = self._process_optim_batch_hook(batch)
            sub_batch_device = sub_batch.to(self.loss_module.device)
            losses_td = self.loss_module(sub_batch_device)
            self._post_loss_hook(sub_batch_device)

            losses_detached = self._optimizer_step(losses_td)
            self._post_optim_hook()
            self._post_optim_log(sub_batch_device)

            if average_losses is None:
                average_losses: _TensorDict = losses_detached
            else:
                for key, item in losses_detached.items():
                    val = average_losses.get(key)
                    average_losses.set(key, val * j / (j + 1) + item / (j + 1))

        if self.optim_steps_per_batch > 0:
            self._log(
                optim_steps=self._optim_count,
                **average_losses,
            )

    def _grad_clip(self) -> float:
        if self.clip_grad_norm:
            gn = nn.utils.clip_grad_norm_(self._params, self.clip_norm)
        else:
            gn = sum([p.grad.pow(2).sum() for p in self._params]).sqrt()
            nn.utils.clip_grad_value_(self._params, self.clip_norm)
        return float(gn)

    def _log(self, log_pbar=False, **kwargs) -> None:
        collected_frames = self.collected_frames
        for key, item in kwargs.items():
            self._log_dict[key].append(item)

            if (collected_frames - self._last_log.get(key, 0)) > self._log_interval:
                self._last_log[key] = collected_frames
                _log = True
            else:
                _log = False
            method = WRITER_METHODS.get(key, "add_scalar")
            if _log and self.writer is not None:
                getattr(self.writer, method)(key, item, global_step=collected_frames)
            if method == "add_scalar" and self.progress_bar and log_pbar:
                if isinstance(item, torch.Tensor):
                    item = item.item()
                self._pbar_str[key] = item

    def _pbar_description(self) -> None:
        if self.progress_bar:
            self._pbar.set_description(
                ", ".join(
                    [
                        f"{key}: {self._pbar_str[key] :{TYPE_DESCR.get(type(self._pbar_str[key]), '4.4f')}}"
                        for key in sorted(self._pbar_str.keys())
                    ]
                )
            )

    def __repr__(self) -> str:
        loss_str = indent(f"loss={self.loss_module}", 4 * " ")
        collector_str = indent(f"collector={self.collector}", 4 * " ")
        optimizer_str = indent(f"optimizer={self.optimizer}", 4 * " ")
        writer = indent(f"writer={self.writer}", 4 * " ")

        string = "\n".join(
            [
                loss_str,
                collector_str,
                optimizer_str,
                writer,
            ]
        )
        string = f"Trainer(\n{string})"
        return string


class SelectKeys:
    """Selects keys in a TensorDict batch.

    Args:
        keys (iterable of strings): keys to be selected in the tensordict.

    Examples:
        >>> trainer = make_trainer()
        >>> key1 = "first key"
        >>> key2 = "second key"
        >>> td = TensorDict(
        ...     {
        ...         key1: torch.randn(3),
        ...         key2: torch.randn(3),
        ...     },
        ...     [],
        ... )
        >>> trainer.register_op("batch_process", SelectKeys([key1]))
        >>> td_out = trainer._process_batch_hook(td)
        >>> assert key1 in td_out.keys()
        >>> assert key2 not in td_out.keys()

    """

    def __init__(self, keys: Sequence[str]):
        if isinstance(keys, str):
            raise RuntimeError(
                "Expected keys to be an iterable of str, got str instead"
            )
        self.keys = keys

    def __call__(self, batch: _TensorDict) -> _TensorDict:
        return batch.select(*self.keys)


class ReplayBufferTrainer:
    """Replay buffer hook provider.

    Args:
        replay_buffer (ReplayBuffer): replay buffer to be used.
        batch_size (int): batch size when sampling data from the
            latest collection or from the replay buffer.
        memmap (bool, optional): if True, a memmap tensordict is created.
            Default is False.
        device (device, optional): device where the samples must be placed.
            Default is cpu.

    Examples:
        >>> rb_trainer = ReplayBufferTrainer(replay_buffer=replay_buffer, batch_size=N)
        >>> trainer.register_op("batch_process", rb_trainer.extend)
        >>> trainer.register_op("process_optim_batch", rb_trainer.sample)
        >>> trainer.register_op("post_loss", rb_trainer.update_priority)

    """

    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        batch_size: int,
        memmap: bool = False,
        device: DEVICE_TYPING = "cpu",
    ) -> None:
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.memmap = memmap
        self.device = device

    def extend(self, batch: _TensorDict) -> _TensorDict:
        if "mask" in batch.keys():
            batch = batch[batch.get("mask").squeeze(-1)]
        else:
            batch = batch.reshape(-1)
        # reward_training = batch.get("reward").mean().item()
        batch = batch.cpu()
        if self.memmap:
            # We can already place the tensords on the device if they're memmap,
            # as this is a lazy op
            batch = batch.memmap_().to(self.device)
        self.replay_buffer.extend(batch)

    def sample(self, batch: _TensorDict) -> _TensorDict:
        sample = self.replay_buffer.sample(self.batch_size)
        sample = sample.contiguous()
        return sample.to(self.device)

    def update_priority(self, batch: _TensorDict) -> None:
        if isinstance(self.replay_buffer, TensorDictPrioritizedReplayBuffer):
            self.replay_buffer.update_priority(batch)


class LogReward:
    """Reward logger hook.

    Args:
        logname (str, optional): name of the rewards to be logged. Default is `"r_training"`.
        log_pbar (bool, optional): if True, the reward value will be logged on
            the progression bar. Default is `False`.

    Examples:
        >>> log_reward = LogReward("reward")
        >>> trainer.register_op("pre_steps_log", log_reward)

    """

    def __init__(self, logname="r_training", log_pbar: bool = False):
        self.logname = logname
        self.log_pbar = log_pbar

    def __call__(self, batch: _TensorDict) -> Dict:
        if "mask" in batch.keys():
            return {
                self.logname: batch.get("reward")[batch.get("mask").squeeze(-1)]
                .mean()
                .item(),
                "log_pbar": self.log_pbar,
            }
        return {
            self.logname: batch.get("reward").mean().item(),
            "log_pbar": self.log_pbar,
        }


class RewardNormalizer:
    """Reward normalizer hook.

    Args:
        decay (float, optional): exponential moving average decay parameter.
            Default is 0.999

    Examples:
        >>> reward_normalizer = RewardNormalizer()
        >>> trainer.register_op("batch_process", reward_normalizer.update_reward_stats)
        >>> trainer.register_op("process_optim_batch", reward_normalizer.normalize_reward)

    """

    def __init__(
        self, decay: float = 0.999, scale: float = 1.0, log_pbar: bool = False
    ):
        self._normalize_has_been_called = False
        self._update_has_been_called = False
        self._reward_stats = OrderedDict()
        self._reward_stats["decay"] = decay
        self.scale = scale

    @torch.no_grad()
    def update_reward_stats(self, batch: _TensorDict) -> None:
        reward = batch.get("reward")
        if "mask" in batch.keys():
            reward = reward[batch.get("mask").squeeze(-1)]
        if self._update_has_been_called and not self._normalize_has_been_called:
            # We'd like to check that rewards are normalized. Problem is that the trainer can collect data without calling steps...
            # raise RuntimeError(
            #     "There have been two consecutive calls to update_reward_stats without a call to normalize_reward. "
            #     "Check that normalize_reward has been registered in the trainer."
            # )
            pass
        decay = self._reward_stats.get("decay", 0.999)
        sum = self._reward_stats["sum"] = (
            decay * self._reward_stats.get("sum", 0.0) + reward.sum()
        )
        ssq = self._reward_stats["ssq"] = (
            decay * self._reward_stats.get("ssq", 0.0) + reward.pow(2).sum()
        )
        count = self._reward_stats["count"] = (
            decay * self._reward_stats.get("count", 0.0) + reward.numel()
        )

        self._reward_stats["mean"] = sum / count
        if count > 1:
            var = self._reward_stats["var"] = (ssq - sum.pow(2) / count) / (count - 1)
        else:
            var = self._reward_stats["var"] = torch.zeros_like(sum)

        self._reward_stats["std"] = var.clamp_min(1e-6).sqrt()
        self._update_has_been_called = True

    def normalize_reward(self, tensordict: _TensorDict) -> _TensorDict:
        tensordict = tensordict.to_tensordict()  # make sure it is not a SubTensorDict
        reward = tensordict.get("reward")
        reward = reward - self._reward_stats["mean"].to(tensordict.device)
        reward = reward / self._reward_stats["std"].to(tensordict.device)
        tensordict.set("reward", reward * self.scale)
        self._normalize_has_been_called = True
        return tensordict


def mask_batch(batch: _TensorDict) -> _TensorDict:
    """Batch masking hook.

    If a tensordict contained padded trajectories but only single events are
    needed, this hook can be used to select the valid events from the original
    tensordict.

    Args:
        batch:

    Examples:
        >>> trainer = mocking_trainer()
        >>> trainer.register_op("batch_process", mask_batch)

    """
    if "mask" in batch.keys():
        mask = batch.get("mask")
        return batch[mask.squeeze(-1)]
    return batch


class BatchSubSampler:
    """Data subsampler for online RL algorithms.

    This class subsamples a part of a whole batch of data just collected from the
    environment.

    Args:
        batch_size (int): sub-batch size to collect. The provided batch size
            must be equal to the total number of items in the output tensordict,
            which will have size [batch_size // sub_traj_len, sub_traj_len].
        sub_traj_len (int, optional): length of the trajectories that
            sub-samples must have in online settings. Default is -1 (i.e.
            takes the full length of the trajectory)
        min_sub_traj_len (int, optional): minimum value of `sub_traj_len`, in
            case some elements of the batch contain few steps.
            Default is -1 (i.e. no minimum value)

    Examples:
        >>> td = TensorDict(
        ...     {
        ...         key1: torch.stack([torch.arange(0, 10), torch.arange(10, 20)], 0),
        ...         key2: torch.stack([torch.arange(0, 10), torch.arange(10, 20)], 0),
        ...     },
        ...     [13, 10],
        ... )
        >>> trainer.register_op(
        ...     "process_optim_batch",
        ...     BatchSubSampler(batch_size=batch_size, sub_traj_len=sub_traj_len),
        ... )
        >>> td_out = trainer._process_optim_batch_hook(td)
        >>> assert td_out.shape == torch.Size([batch_size // sub_traj_len, sub_traj_len])

    """

    def __init__(
        self, batch_size: int, sub_traj_len: int = 0, min_sub_traj_len: int = 0
    ) -> None:
        self.batch_size = batch_size
        self.sub_traj_len = sub_traj_len
        self.min_sub_traj_len = min_sub_traj_len

    def __call__(self, batch: _TensorDict) -> _TensorDict:
        """Sub-sampled part of a batch randomly.

        If the batch has one dimension, a random subsample of length
        self.bach_size will be returned. If the batch has two or more
        dimensions, it is assumed that the first dimension represents the
        batch, and the second the time. If so, the resulting subsample will
        contain consecutive samples across time.
        """

        if batch.ndimension() == 1:
            return batch[torch.randperm(batch.shape[0])[: self.batch_size]]

        sub_traj_len = self.sub_traj_len if self.sub_traj_len > 0 else batch.shape[1]
        if "mask" in batch.keys():
            # if a valid mask is present, it's important to sample only
            # valid steps
            traj_len = batch.get("mask").sum(1).squeeze()
            sub_traj_len = max(
                self.min_sub_traj_len,
                min(sub_traj_len, traj_len.min().int().item()),
            )
        else:
            traj_len = (
                torch.ones(batch.shape[0], device=batch.device, dtype=torch.bool)
                * batch.shape[1]
            )
        len_mask = traj_len >= sub_traj_len
        valid_trajectories = torch.arange(batch.shape[0])[len_mask]

        batch_size = self.batch_size // sub_traj_len
        if batch_size == 0:
            raise RuntimeError(
                "Resulting batch size is zero. The batch size given to "
                "BatchSubSampler must be equal to the total number of elements "
                "that will result in a batch provided to the loss function."
            )
        traj_idx = valid_trajectories[
            torch.randint(
                valid_trajectories.numel(), (batch_size,), device=batch.device
            )
        ]

        if sub_traj_len < batch.shape[1]:
            _traj_len = traj_len[traj_idx]
            seq_idx = (
                torch.rand_like(_traj_len, dtype=torch.float)
                * (_traj_len - sub_traj_len)
            ).int()
            seq_idx = seq_idx.unsqueeze(-1).expand(-1, sub_traj_len)
        elif sub_traj_len == batch.shape[1]:
            seq_idx = torch.zeros(
                batch_size, sub_traj_len, device=batch.device, dtype=torch.long
            )
        else:
            raise ValueError(
                f"sub_traj_len={sub_traj_len} is not allowed. Accepted values "
                f"are in the range [1, {batch.shape[1]}]."
            )

        seq_idx = seq_idx + torch.arange(sub_traj_len, device=seq_idx.device)
        td = batch[traj_idx].clone()
        td = td.apply(
            lambda t: t.gather(
                dim=1,
                index=expand_right(seq_idx, (batch_size, sub_traj_len, *t.shape[2:])),
            ),
            batch_size=(batch_size, sub_traj_len),
        )
        if "mask" in batch.keys() and not td.get("mask").all():
            raise RuntimeError("Sampled invalid steps")
        return td


class Recorder:
    """Recorder hook for Trainer.

    Args:
        record_interval (int): total number of optimisation steps
            between two calls to the recorder for testing.
        record_frames (int): number of frames to be recorded during
            testing.
        frame_skip (int): frame_skip used in the environment. It is
            important to let the trainer know the number of frames skipped at
            each iteration, otherwise the frame count can be underestimated.
            For logging, this parameter is important to normalize the reward.
            Finally, to compare different runs with different frame_skip,
            one must normalize the frame count and rewards. Default is 1.
        policy_exploration (ProbabilisticTDModule): a policy
            instance used for

            (1) updating the exploration noise schedule;

            (2) testing the policy on the recorder.

            Given that this instance is supposed to both explore and render
            the performance of the policy, it should be possible to turn off
            the explorative behaviour by calling the
            `set_exploration_mode('mode')` context manager.
        recorder (_EnvClass): An environment instance to be used
            for testing.
        exploration_mode (str, optional): exploration mode to use for the
            policy. By default, no exploration is used and the value used is
            "mode". Set to "random" to enable exploration
        out_key (str, optional): reward key to set to the logger. Default is
            `"reward_evaluation"`.
        suffix (str, optional): suffix of the video to be recorded.
        log_pbar (bool, optional): if True, the reward value will be logged on
            the progression bar. Default is `False`.

    """

    def __init__(
        self,
        record_interval: int,
        record_frames: int,
        frame_skip: int,
        policy_exploration: TensorDictModule,
        recorder: _EnvClass,
        exploration_mode: str = "mode",
        log_keys: Optional[List[str]] = None,
        out_keys: Optional[Dict[str, str]] = None,
        suffix: Optional[str] = None,
        log_pbar: bool = False,
    ) -> None:

        self.policy_exploration = policy_exploration
        self.recorder = recorder
        self.record_frames = record_frames
        self.frame_skip = frame_skip
        self._count = 0
        self.record_interval = record_interval
        self.exploration_mode = exploration_mode
        if log_keys is None:
            log_keys = ["reward"]
        if out_keys is None:
            out_keys = KeyDependentDefaultDict()
            out_keys["reward"] = "r_evaluation"
        self.log_keys = log_keys
        self.out_keys = out_keys
        self.suffix = suffix
        self.log_pbar = log_pbar

    @torch.inference_mode()
    def __call__(self, batch: _TensorDict) -> Dict:
        out = None
        if self._count % self.record_interval == 0:
            with set_exploration_mode(self.exploration_mode):
                if isinstance(self.policy_exploration, torch.nn.Module):
                    self.policy_exploration.eval()
                self.recorder.eval()
                if isinstance(self.recorder, TransformedEnv):
                    self.recorder.transform.eval()
                td_record = self.recorder.rollout(
                    policy=self.policy_exploration,
                    max_steps=self.record_frames,
                    auto_reset=True,
                    auto_cast_to_device=True,
                ).clone()
                if isinstance(self.policy_exploration, torch.nn.Module):
                    self.policy_exploration.train()
                self.recorder.train()
                self.recorder.transform.dump(suffix=self.suffix)

                out = dict()
                for key in self.log_keys:
                    value = td_record.get(key).float().mean()
                    if key == "reward":
                        value = value / self.frame_skip
                    if key == "solved":
                        value = value.any().float()
                    out[self.out_keys[key]] = value
                out["log_pbar"] = self.log_pbar
        self._count += 1
        self.recorder.close()
        return out


class UpdateWeights:
    """A collector weights update hook class.

    This hook must be used whenever the collector policy weights sit on a
    different device than the policy weights being trained by the Trainer.
    In that case, those weights must be synced across devices at regular
    intervals. If the devices match, this will result in a no-op.

    Args:
        collector (_DataCollector): A data collector where the policy weights
            must be synced.
        update_weights_interval (int): Interval (in terms of number of batches
            collected) where the sync must take place.

    Examples:
        >>> update_weights = UpdateWeights(trainer.collector, T)
        >>> trainer.register_op("post_steps", update_weights)

    """

    def __init__(self, collector: _DataCollector, update_weights_interval: int):
        self.collector = collector
        self.update_weights_interval = update_weights_interval
        self.counter = 0

    def __call__(self):
        self.counter += 1
        if self.counter % self.update_weights_interval == 0:
            self.collector.update_policy_weights_()


class CountFramesLog:
    """A frame counter hook.

    Args:
        frame_skip (int): frame skip of the environment. This argument is
            important to keep track of the total number of frames, not the
            apparent one.
        log_pbar (bool, optional): if True, the reward value will be logged on
            the progression bar. Default is `False`.

    Examples:
        >>> count_frames = CountFramesLog(frame_skip=frame_skip)
        >>> trainer.register_op("pre_steps_log", count_frames)


    """

    def __init__(self, frame_skip: int, log_pbar: bool = False):
        self.frame_count = 0
        self.frame_skip = frame_skip
        self.log_pbar = log_pbar

    def __call__(self, batch: _TensorDict) -> Dict:
        if "mask" in batch.keys():
            current_frames = batch.get("mask").sum().item() * self.frame_skip
        else:
            current_frames = batch.numel() * self.frame_skip
        self.frame_count += current_frames
        return {"n_frames": self.frame_count, "log_pbar": self.log_pbar}


def _check_input_output_typehint(func: Callable, input: Type, output: Type):
    # Placeholder for a function that checks the types input / output against expectations
    return
