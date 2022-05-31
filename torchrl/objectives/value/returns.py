# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union

import torch
from torch import nn


def bellman_max(
    next_observation: torch.Tensor,
    reward: torch.Tensor,
    done: torch.Tensor,
    gamma: Union[float, torch.Tensor],
    value_model: nn.Module,
):
    qmax = value_model(next_observation).max(dim=-1)[0]
    nonterminal_target = reward + gamma * qmax
    terminal_target = reward
    target = done * terminal_target + (~done) * nonterminal_target
    return target
