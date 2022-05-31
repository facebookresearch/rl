# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch


def cross_entropy_loss(
    log_policy: torch.Tensor, action: torch.Tensor, inplace: bool = False
) -> torch.Tensor:
    """
    Returns the cross entropy loss defined as the log-softmax value indexed  by the action index.
    Supports discrete (integer) actions or one-hot encodings.

    Args:
        log_policy: Tensor of the log_softmax values of the policy.
        action: Integer or one-hot representation of the actions undertaken. Must have a shape log_policy.shape[:-1]
            (integer representation) or log_policy.shape (one-hot).
        inplace: fills log_policy in-place with 0.0 at non-selected actions before summing along the last dimensions.
            This is usually faster but it will change the value of log-policy in place, which may lead to unwanted
            behaviours.

    Returns:

    """
    if action.shape == log_policy.shape:
        if action.dtype not in (torch.bool, torch.long, torch.uint8):
            raise TypeError(
                f"Cross-entropy loss with {action.dtype} dtype is not permitted"
            )
        if not ((action == 1).sum(-1) == 1).all():
            raise RuntimeError(
                "Expected the action tensor to be a one hot encoding of the actions taken, "
                "but got more/less than one non-null boolean index on the last dimension"
            )
        if inplace:
            cross_entropy = log_policy.masked_fill_(action, 0.0).sum(-1)
        else:
            cross_entropy = (log_policy * action).sum(-1)
    elif action.shape == log_policy.shape[:-1]:
        cross_entropy = torch.gather(log_policy, dim=-1, index=action[..., None])
        cross_entropy.squeeze_(-1)
    else:
        raise RuntimeError(
            f"unexpected action shape in cross_entropy_loss with log_policy.shape={log_policy.shape} and"
            f"action.shape={action.shape}"
        )
    return cross_entropy
