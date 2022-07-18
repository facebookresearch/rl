# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass
from typing import Optional

import torch

from torchrl.data import (
    DEVICE_TYPING,
    ReplayBuffer,
    TensorDictPrioritizedReplayBuffer,
    TensorDictReplayBuffer,
)

__all__ = ["make_replay_buffer"]

from torchrl.data.replay_buffers.storages import LazyMemmapStorage


def make_replay_buffer(device: DEVICE_TYPING, cfg: "DictConfig") -> ReplayBuffer:
    """Builds a replay buffer using the config built from ReplayArgsConfig."""
    device = torch.device(device)

    def collate_fn(x):
        return x.to_tensordict().to(device)

    if not cfg.prb:
        buffer = TensorDictReplayBuffer(
            cfg.buffer_size,
            collate_fn=collate_fn,
            pin_memory=device != torch.device("cpu"),
            prefetch=cfg.buffer_prefetch,
            storage=LazyMemmapStorage(
                cfg.buffer_size,
                scratch_dir=cfg.buffer_scratch_dir,
            ),
        )
    else:
        buffer = TensorDictPrioritizedReplayBuffer(
            cfg.buffer_size,
            alpha=0.7,
            beta=0.5,
            collate_fn=collate_fn,
            pin_memory=device != torch.device("cpu"),
            prefetch=cfg.buffer_prefetch,
            storage=LazyMemmapStorage(
                cfg.buffer_size,
                scratch_dir=cfg.buffer_scratch_dir,
            ),
        )
    return buffer


@dataclass
class ReplayArgsConfig:
    buffer_size: int = 1000000
    # buffer size, in number of frames stored. Default=1e6
    prb: bool = False
    # whether a Prioritized replay buffer should be used instead of a more basic circular one.
    buffer_scratch_dir: Optional[str] = None
    # directory where the buffer data should be stored. If none is passed, they will be placed in /tmp/
    buffer_prefetch: int = 10
    # prefetching queue length for the replay buffer
