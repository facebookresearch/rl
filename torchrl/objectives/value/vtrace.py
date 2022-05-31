# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Tuple, Union

import torch


def c_val(
    log_pi: torch.Tensor,
    log_mu: torch.Tensor,
    c: Union[float, torch.Tensor] = 1,
) -> torch.Tensor:
    return (log_pi - log_mu).clamp_max(math.log(c)).exp()


def dv_val(
    rewards: torch.Tensor,
    vals: torch.Tensor,
    gamma: Union[float, torch.Tensor],
    rho_bar: Union[float, torch.Tensor],
    log_pi: torch.Tensor,
    log_mu: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    rho = c_val(log_pi, log_mu, rho_bar)
    next_vals = torch.cat([vals[:, 1:], torch.zeros_like(vals[:, :1])], 1)
    dv = rho * (rewards + gamma * next_vals - vals)
    return dv, rho


def vtrace(
    rewards: torch.Tensor,
    vals: torch.Tensor,
    log_pi: torch.Tensor,
    log_mu: torch.Tensor,
    gamma: Union[torch.Tensor, float],
    rho_bar: Union[float, torch.Tensor] = 1.0,
    c_bar: Union[float, torch.Tensor] = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    T = vals.shape[1]
    if not isinstance(gamma, torch.Tensor):
        gamma = torch.full_like(vals, gamma)

    dv, rho = dv_val(rewards, vals, gamma, rho_bar, log_pi, log_mu)
    c = c_val(log_pi, log_mu, c_bar)

    v_out = []
    v_out.append(vals[:, -1] + dv[:, -1])
    for t in range(T - 2, -1, -1):
        _v_out = (
            vals[:, t] + dv[:, t] + gamma[:, t] * c[:, t] * (v_out[-1] - vals[:, t + 1])
        )
        v_out.append(_v_out)
    v_out = torch.stack(list(reversed(v_out)), 1)
    return v_out, rho
