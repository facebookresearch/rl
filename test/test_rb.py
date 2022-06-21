# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import numpy as np
import pytest
import torch
from _utils_internal import get_available_devices
from torchrl.data import TensorDict
from torchrl.data.replay_buffers import TensorDictPrioritizedReplayBuffer
from torchrl.data.tensordict.tensordict import assert_allclose_td


@pytest.mark.parametrize("priority_key", ["pk", "td_error"])
@pytest.mark.parametrize("contiguous", [True, False])
@pytest.mark.parametrize("device", get_available_devices())
def test_prb(priority_key, contiguous, device):
    torch.manual_seed(0)
    np.random.seed(0)
    rb = TensorDictPrioritizedReplayBuffer(
        5,
        alpha=0.7,
        beta=0.9,
        collate_fn=None if contiguous else lambda x: torch.stack(x, 0),
        priority_key=priority_key,
    )
    td1 = TensorDict(
        source={
            "a": torch.randn(3, 1),
            priority_key: torch.rand(3, 1) / 10,
            "_idx": torch.arange(3).view(3, 1),
        },
        batch_size=[3],
    ).to(device)
    rb.extend(td1)
    s = rb.sample(2)
    assert s.batch_size == torch.Size(
        [
            2,
        ]
    )
    assert (td1[s.get("_idx").squeeze()].get("a") == s.get("a")).all()
    assert_allclose_td(td1[s.get("_idx").squeeze()].select("a"), s.select("a"))

    # test replacement
    td2 = TensorDict(
        source={
            "a": torch.randn(5, 1),
            priority_key: torch.rand(5, 1) / 10,
            "_idx": torch.arange(5).view(5, 1),
        },
        batch_size=[5],
    ).to(device)
    rb.extend(td2)
    s = rb.sample(5)
    assert s.batch_size == torch.Size(
        [
            5,
        ]
    )
    assert (td2[s.get("_idx").squeeze()].get("a") == s.get("a")).all()
    assert_allclose_td(td2[s.get("_idx").squeeze()].select("a"), s.select("a"))

    # test strong update
    # get all indices that match first item
    idx = s.get("_idx")
    idx_match = (idx == idx[0]).nonzero()[:, 0]
    s.set_at_(
        priority_key,
        torch.ones(
            idx_match.numel(),
            1,
            device=device,
        )
        * 100000000,
        idx_match,
    )
    val = s.get("a")[0]

    idx0 = s.get("_idx")[0]
    rb.update_priority(s)
    s = rb.sample(5)
    assert (val == s.get("a")).sum() >= 1
    torch.testing.assert_allclose(
        td2[idx0].get("a").view(1), s.get("a").unique().view(1)
    )

    # test updating values of original td
    td2.set_("a", torch.ones_like(td2.get("a")))
    s = rb.sample(5)
    torch.testing.assert_allclose(
        td2[idx0].get("a").view(1), s.get("a").unique().view(1)
    )


@pytest.mark.parametrize("stack", [False, True])
def test_rb_trajectories(stack):
    traj_td = TensorDict(
        {"obs": torch.randn(3, 4, 5), "actions": torch.randn(3, 4, 2)},
        batch_size=[3, 4],
    )
    if stack:
        traj_td = torch.stack([td.to_tensordict() for td in traj_td], 0)

    rb = TensorDictPrioritizedReplayBuffer(
        5,
        alpha=0.7,
        beta=0.9,
        collate_fn=lambda x: torch.stack(x, 0),
        priority_key="td_error",
    )
    rb.extend(traj_td)
    sampled_td = rb.sample(3)
    sampled_td.set("td_error", torch.rand(3))
    rb.update_priority(sampled_td)
    sampled_td = rb.sample(3, return_weight=True)
    assert (sampled_td.get("_weight") > 0).all()
    assert sampled_td.batch_size == torch.Size([3])

    # set back the trajectory length
    sampled_td_filtered = sampled_td.to_tensordict().exclude(
        "_weight", "index", "td_error"
    )
    sampled_td_filtered.batch_size = [3, 4]


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
