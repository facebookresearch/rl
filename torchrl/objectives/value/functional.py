# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch

__all__ = [
    "generalized_advantage_estimate",
    "vec_td_lambda_return_estimate",
    "vec_td_lambda_advantage_estimate",
    "td_lambda_return_estimate",
    "td_lambda_advantage_estimate",
    "td_advantage_estimate",
]


def generalized_advantage_estimate(
    gamma: float,
    lmbda: float,
    state_value: torch.Tensor,
    next_state_value: torch.Tensor,
    reward: torch.Tensor,
    done: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get generalized advantage estimate of a trajectory
    Refer to "HIGH-DIMENSIONAL CONTINUOUS CONTROL USING GENERALIZED ADVANTAGE ESTIMATION"
    https://arxiv.org/pdf/1506.02438.pdf for more context.

    Args:
        gamma (scalar): exponential mean discount.
        lmbda (scalar): trajectory discount.
        state_value (Tensor): value function result with old_state input.
            must be a [Batch x TimeSteps x 1] or [Batch x TimeSteps] tensor
        next_state_value (Tensor): value function result with new_state input.
            must be a [Batch x TimeSteps x 1] or [Batch x TimeSteps] tensor
        reward (Tensor): reward of taking actions in the environment.
            must be a [Batch x TimeSteps x 1] or [Batch x TimeSteps] tensor
        done (Tensor): boolean flag for end of episode.
    """
    for tensor in (next_state_value, state_value, reward, done):
        if tensor.shape[-1] != 1:
            raise RuntimeError(
                "Last dimension of generalized_advantage_estimate inputs must be a singleton dimension."
            )
    not_done = 1 - done.to(next_state_value.dtype)
    *batch_size, time_steps = not_done.shape[:-1]
    device = state_value.device
    advantage = torch.empty(*batch_size, time_steps, 1, device=device)
    prev_advantage = 0
    for t in reversed(range(time_steps)):
        delta = (
            reward[..., t, :]
            + (gamma * next_state_value[..., t, :] * not_done[..., t, :])
            - state_value[..., t, :]
        )
        prev_advantage = advantage[..., t, :] = delta + (
            gamma * lmbda * prev_advantage * not_done[..., t, :]
        )

    value_target = advantage + state_value

    return advantage, value_target


def td_advantage_estimate(
    gamma: float,
    state_value: torch.Tensor,
    next_state_value: torch.Tensor,
    reward: torch.Tensor,
    done: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get generalized advantage estimate of a trajectory
    Refer to "HIGH-DIMENSIONAL CONTINUOUS CONTROL USING GENERALIZED ADVANTAGE ESTIMATION"
    https://arxiv.org/pdf/1506.02438.pdf for more context.

    Args:
        gamma (scalar): exponential mean discount.
        state_value (Tensor): value function result with old_state input.
            must be a [Batch x TimeSteps x 1] or [Batch x TimeSteps] tensor
        next_state_value (Tensor): value function result with new_state input.
            must be a [Batch x TimeSteps x 1] or [Batch x TimeSteps] tensor
        reward (Tensor): reward of taking actions in the environment.
            must be a [Batch x TimeSteps x 1] or [Batch x TimeSteps] tensor
        done (Tensor): boolean flag for end of episode.
    """
    for tensor in (next_state_value, state_value, reward, done):
        if tensor.shape[-1] != 1:
            raise RuntimeError(
                "Last dimension of generalized_advantage_estimate inputs must be a singleton dimension."
            )
    not_done = 1 - done.to(next_state_value.dtype)
    advantage = reward + gamma * not_done * next_state_value - state_value
    return advantage


def td_lambda_return_estimate(
    gamma: float,
    lmbda: float,
    next_state_value: torch.Tensor,
    reward: torch.Tensor,
    done: torch.Tensor,
) -> torch.Tensor:
    """TD(lambda) return estimate.

    Args:
        gamma (scalar): exponential mean discount.
        lmbda (scalar): trajectory discount.
        next_state_value (Tensor): value function result with new_state input.
            must be a [Batch x TimeSteps x 1] or [Batch x TimeSteps] tensor
        reward (Tensor): reward of taking actions in the environment.
            must be a [Batch x TimeSteps x 1] or [Batch x TimeSteps] tensor
        done (Tensor): boolean flag for end of episode.
    """
    for tensor in (next_state_value, reward, done):
        if tensor.shape[-1] != 1:
            raise RuntimeError(
                "Last dimension of generalized_advantage_estimate inputs must be a singleton dimension."
            )
    not_done = 1 - done.to(next_state_value.dtype)
    next_state_value = not_done * next_state_value

    returns = torch.empty_like(next_state_value)

    g = next_state_value[..., -1, :]
    T = returns.shape[-2]

    for i in reversed(range(T)):
        g = returns[..., i, :] = reward[..., i, :] + gamma * (
            (1 - lmbda) * next_state_value[..., i, :] + lmbda * g
        )
    return returns


def td_lambda_advantage_estimate(
    gamma: float,
    lmbda: float,
    state_value: torch.Tensor,
    next_state_value: torch.Tensor,
    reward: torch.Tensor,
    done: torch.Tensor,
) -> torch.Tensor:
    """TD(lambda) advantage estimate.

    Args:
        gamma (scalar): exponential mean discount.
        lmbda (scalar): trajectory discount.
        state_value (Tensor): value function result with old_state input.
            must be a [Batch x TimeSteps x 1] or [Batch x TimeSteps] tensor
        next_state_value (Tensor): value function result with new_state input.
            must be a [Batch x TimeSteps x 1] or [Batch x TimeSteps] tensor
        reward (Tensor): reward of taking actions in the environment.
            must be a [Batch x TimeSteps x 1] or [Batch x TimeSteps] tensor
        done (Tensor): boolean flag for end of episode.
    """
    if not state_value.shape == next_state_value.shape:
        raise RuntimeError("shape of state_value and next_state_value must match")
    returns = td_lambda_return_estimate(gamma, lmbda, next_state_value, reward, done)
    advantage = returns - state_value
    return advantage


def _custom_conv1d(tensor, filter):
    """Computes a conv1d filter over a value.
    This is usually used to compute a discounted return:

    Tensor:                         Filter                      Result (discounted return)
    [ r_0,                          [ 1.0,                      [ r_0 + g r_1 + g^2 r_2 + r^3 r_3,
      r_1,                            g,                          r_1 + g r_2 + g^2 r_3,
      r_2,                            g^2,                        r_2 + g r_3,
      r_3,                            g^3 ]                       r_3 ]
      0,      |                        |
      0,      |  zero padding          | direction of filter
      0 ]     |                        v

    This function takes care of applying the one-sided zero padding. In this example,
    `Filter_dim` = `Time` = 4, but in practice Filter_dim can be <= to `Time`.

    Args:
        tensor (torch.Tensor): a [ Batch x 1 x Time ] floating-point tensor
        filter (torch.Tensor): a [ Filter_dim x 1 ] floating-point filter

    Returns: a filtered tensor of the same shape as the input tensor.

    """
    val_pad = torch.cat(
        [
            tensor,
            torch.zeros(tensor.shape[0], 1, filter.shape[-2] - 1, device=tensor.device),
        ],
        -1,
    )

    # shape = val.shape
    filter = filter.squeeze(-1).unsqueeze(0).unsqueeze(0)  # 1 x 1 x T
    out = torch.conv1d(val_pad, filter)
    # out = out.view(shape)
    if not out.shape == tensor.shape:
        raise RuntimeError("wrong output shape")
    return out


def vec_td_lambda_advantage_estimate(
    gamma, lmbda, state_value, next_state_value, reward, done
):
    """Vectorized TD(lambda) advantage estimate.

    Args:
        gamma (scalar): exponential mean discount.
        lmbda (scalar): trajectory discount.
        state_value (Tensor): value function result with old_state input.
            must be a [Batch x TimeSteps x 1] or [Batch x TimeSteps] tensor
        next_state_value (Tensor): value function result with new_state input.
            must be a [Batch x TimeSteps x 1] or [Batch x TimeSteps] tensor
        reward (Tensor): reward of taking actions in the environment.
            must be a [Batch x TimeSteps x 1] or [Batch x TimeSteps] tensor
        done (Tensor): boolean flag for end of episode.
    """
    return (
        vec_td_lambda_return_estimate(gamma, lmbda, next_state_value, reward, done)
        - state_value
    )


def vec_td_lambda_return_estimate(gamma, lmbda, next_state_value, reward, done):
    """Vectorized TD(lambda) return estimate.

    Args:
        gamma (scalar): exponential mean discount.
        lmbda (scalar): trajectory discount.
        next_state_value (Tensor): value function result with new_state input.
            must be a [Batch x TimeSteps x 1] or [Batch x TimeSteps] tensor
        reward (Tensor): reward of taking actions in the environment.
            must be a [Batch x TimeSteps x 1] or [Batch x TimeSteps] tensor
        done (Tensor): boolean flag for end of episode.
    """

    shape = next_state_value.shape
    if not shape[-1] == 1:
        raise RuntimeError("last dimension of inputs shape must be singleton")

    next_state_value = next_state_value.view(-1, 1, shape[-2])
    reward = reward.view(-1, 1, shape[-2])
    done = done.view(-1, 1, shape[-2])

    """Vectorized version of td_lambda_advantage_estimate"""
    device = reward.device
    not_done = 1 - done.to(next_state_value.dtype)
    next_state_value = not_done * next_state_value

    T = shape[-2]

    first_below_thr_gamma = None

    gammas = torch.ones(T + 1, 1, device=device)
    gammas[1:] = gamma
    gammas = torch.cumprod(gammas, -2)

    lambdas = torch.ones(T + 1, 1, device=device)
    lambdas[1:] = lmbda
    lambdas = torch.cumprod(lambdas, -2)

    first_below_thr = gammas < 1e-7
    if first_below_thr.any():
        first_below_thr_gamma = first_below_thr.nonzero()[0, 0]
    first_below_thr = lambdas < 1e-7
    if first_below_thr.any() and first_below_thr_gamma is not None:
        first_below_thr = max(first_below_thr_gamma, first_below_thr.nonzero()[0, 0])
        gammas = gammas[:first_below_thr]
        lambdas = lambdas[:first_below_thr]

    gammas, gammas_prime = gammas[:-1], gammas[1:]
    lambdas, lambdas_prime = lambdas[:-1], lambdas[1:]

    rs = _custom_conv1d(reward, gammas * lambdas)
    vs = _custom_conv1d(next_state_value, gammas_prime * lambdas)
    gam_lam = gammas_prime * lambdas_prime
    mask = gam_lam.flip(-2)
    if mask.shape[-2] < next_state_value.shape[-1]:
        mask = torch.cat(
            # [torch.zeros_like(next_state_value[..., : -mask.shape[-2], :]), mask], -2
            [
                torch.zeros(
                    next_state_value.shape[-1] - mask.shape[-2], 1, device=device
                ),
                mask,
            ],
            -2,
        )
    vs2 = (
        _custom_conv1d(next_state_value, gam_lam)
        - mask.squeeze(-1) * next_state_value[..., -1:]
    )
    return (rs + vs - vs2).view(shape)
