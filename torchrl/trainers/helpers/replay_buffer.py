# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from argparse import ArgumentParser, Namespace

import torch

from torchrl.data import (
    DEVICE_TYPING,
    ReplayBuffer,
    TensorDictPrioritizedReplayBuffer,
    TensorDictReplayBuffer,
)

__all__ = ["make_replay_buffer", "parser_replay_args"]


def collate_fn(batch):
    batch = torch.stack(batch, 0)
    batch = batch.contiguous()
    return batch


def make_replay_buffer(device: DEVICE_TYPING, args: Namespace) -> ReplayBuffer:
    """Builds a replay buffer using the arguments build from the parser returned by parser_replay_args."""
    device = torch.device(device)
    if not args.prb:
        buffer = TensorDictReplayBuffer(
            args.buffer_size,
            collate_fn=collate_fn,
            pin_memory=device != torch.device("cpu"),
            prefetch=args.optim_steps_per_batch,
        )
    else:
        buffer = TensorDictPrioritizedReplayBuffer(
            args.buffer_size,
            alpha=0.7,
            beta=0.5,
            collate_fn=collate_fn,
            pin_memory=device != torch.device("cpu"),
            prefetch=args.optim_steps_per_batch,
        )
    return buffer


def parser_replay_args(parser: ArgumentParser) -> ArgumentParser:
    """
    Populates the argument parser to build a replay buffer.

    Args:
        parser (ArgumentParser): parser to be populated.

    """

    parser.add_argument(
        "--buffer_size",
        "--buffer-size",
        type=int,
        default=1000000,
        help="buffer size, in number of frames stored. Default=1e6",
    )
    parser.add_argument(
        "--prb",
        action="store_true",
        help="whether a Prioritized replay buffer should be used instead of a more basic circular one.",
    )
    return parser
