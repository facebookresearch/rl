# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os.path
import re

import pytest
import torch
from _utils_internal import get_available_devices
from torch import multiprocessing as mp
from torchrl import prod
from torchrl.data import SavedTensorDict, TensorDict, MemmapTensor
from torchrl.data.tensordict.tensordict import (
    assert_allclose_td,
    LazyStackedTensorDict,
    stack as stack_td,
    pad,
    _TensorDict,
)
from torchrl.data.tensordict.utils import _getitem_batch_size, convert_ellipsis_to_idx


@pytest.mark.parametrize("device", get_available_devices())
def test_tensordict_set(device):
    torch.manual_seed(1)
    td = TensorDict({}, batch_size=(4, 5))
    td.set("key1", torch.randn(4, 5, device=device))
    assert td.device == torch.device(device)
    # by default inplace:
    with pytest.raises(RuntimeError):
        td.set("key1", torch.randn(5, 5, device=device))

    # robust to dtype casting
    td.set_("key1", torch.ones(4, 5, device=device, dtype=torch.double))
    assert (td.get("key1") == 1).all()

    # robust to device casting
    td.set("key_device", torch.ones(4, 5, device="cpu", dtype=torch.double))
    assert td.get("key_device").device == torch.device(device)

    with pytest.raises(
        AttributeError, match="for populating tensordict with new key-value pair"
    ):
        td.set_("smartypants", torch.ones(4, 5, device="cpu", dtype=torch.double))
    # test set_at_
    td.set("key2", torch.randn(4, 5, 6, device=device))
    x = torch.randn(6, device=device)
    td.set_at_("key2", x, (2, 2))
    assert (td.get("key2")[2, 2] == x).all()

    # test set_at_ with dtype casting
    x = torch.randn(6, dtype=torch.double, device=device)
    td.set_at_("key2", x, (2, 2))  # robust to dtype casting
    torch.testing.assert_allclose(td.get("key2")[2, 2], x.to(torch.float))

    td.set("key1", torch.zeros(4, 5, dtype=torch.double, device=device), inplace=True)
    assert (td.get("key1") == 0).all()
    td.set(
        "key1",
        torch.randn(4, 5, 1, 2, dtype=torch.double, device=device),
        inplace=False,
    )
    assert td._dict_meta["key1"].shape == td._tensordict["key1"].shape


def test_pad():
    dim0_left, dim0_right, dim1_left, dim1_right = [0, 1, 0, 2]
    td = TensorDict(
        {
            "a": torch.ones(3, 4, 1),
            "b": torch.zeros(3, 4, 1, 1),
        },
        batch_size=[3, 4],
    )

    padded_td = pad(td, [dim0_left, dim0_right, dim1_left, dim1_right], value=0.0)

    expected_a = torch.cat([torch.ones(3, 4, 1), torch.zeros(1, 4, 1)], dim=0)
    expected_a = torch.cat([expected_a, torch.zeros(4, 2, 1)], dim=1)

    assert padded_td["a"].shape == (4, 6, 1)
    assert padded_td["b"].shape == (4, 6, 1, 1)
    assert torch.equal(padded_td["a"], expected_a)
    padded_td._check_batch_size()


@pytest.mark.parametrize("device", get_available_devices())
def test_stack(device):
    torch.manual_seed(1)
    tds_list = [TensorDict(source={}, batch_size=(4, 5)) for _ in range(3)]
    tds = stack_td(tds_list, 0, contiguous=False)
    assert tds[0] is tds_list[0]

    td = TensorDict(
        source={"a": torch.randn(4, 5, 3, device=device)}, batch_size=(4, 5)
    )
    td_list = list(td)
    td_reconstruct = stack_td(td_list, 0)
    assert td_reconstruct.batch_size == td.batch_size
    assert (td_reconstruct == td).all()


@pytest.mark.parametrize("device", get_available_devices())
def test_tensordict_indexing(device):
    torch.manual_seed(1)
    td = TensorDict({}, batch_size=(4, 5))
    td.set("key1", torch.randn(4, 5, 1, device=device))
    td.set("key2", torch.randn(4, 5, 6, device=device, dtype=torch.double))

    td_select = td[2, 2]
    td_select._check_batch_size()

    td_select = td[2, :2]
    td_select._check_batch_size()

    td_select = td[None, :2]
    td_select._check_batch_size()

    td_reconstruct = stack_td([_td for _td in td], 0, contiguous=False)
    assert (
        td_reconstruct == td
    ).all(), f"td and td_reconstruct differ, got {td} and {td_reconstruct}"

    superlist = [stack_td([__td for __td in _td], 0, contiguous=False) for _td in td]
    td_reconstruct = stack_td(superlist, 0, contiguous=False)
    assert (
        td_reconstruct == td
    ).all(), f"td and td_reconstruct differ, got {td == td_reconstruct}"

    x = torch.randn(4, 5, device=device)
    td = TensorDict(
        source={"key1": torch.zeros(3, 4, 5, device=device)},
        batch_size=[3, 4],
    )
    td[0].set_("key1", x)
    torch.testing.assert_allclose(td.get("key1")[0], x)
    torch.testing.assert_allclose(td.get("key1")[0], td[0].get("key1"))

    y = torch.randn(3, 5, device=device)
    td[:, 0].set_("key1", y)
    torch.testing.assert_allclose(td.get("key1")[:, 0], y)
    torch.testing.assert_allclose(td.get("key1")[:, 0], td[:, 0].get("key1"))


@pytest.mark.parametrize("device", get_available_devices())
def test_subtensordict_construction(device):
    torch.manual_seed(1)
    td = TensorDict({}, batch_size=(4, 5))
    val1 = torch.randn(4, 5, 1, device=device)
    val2 = torch.randn(4, 5, 6, dtype=torch.double, device=device)
    val1_copy = val1.clone()
    val2_copy = val2.clone()
    td.set("key1", val1)
    td.set("key2", val2)
    std1 = td.get_sub_tensordict(2)
    std2 = std1.get_sub_tensordict(2)
    idx = (2, 2)
    std_control = td.get_sub_tensordict(idx)
    assert (std_control.get("key1") == std2.get("key1")).all()
    assert (std_control.get("key2") == std2.get("key2")).all()

    # write values
    with pytest.raises(RuntimeError, match="is prohibited for existing tensors"):
        std_control.set("key1", torch.randn(1, device=device))
    with pytest.raises(RuntimeError, match="is prohibited for existing tensors"):
        std_control.set("key2", torch.randn(6, device=device, dtype=torch.double))

    subval1 = torch.randn(1, device=device)
    subval2 = torch.randn(6, device=device, dtype=torch.double)
    std_control.set_("key1", subval1)
    std_control.set_("key2", subval2)
    assert (val1_copy[idx] != subval1).all()
    assert (td.get("key1")[idx] == subval1).all()
    assert (td.get("key1")[1, 1] == val1_copy[1, 1]).all()

    assert (val2_copy[idx] != subval2).all()
    assert (td.get("key2")[idx] == subval2).all()
    assert (td.get("key2")[1, 1] == val2_copy[1, 1]).all()

    assert (std_control.get("key1") == std2.get("key1")).all()
    assert (std_control.get("key2") == std2.get("key2")).all()

    assert std_control.get_parent_tensordict() is td
    assert (
        std_control.get_parent_tensordict()
        is std2.get_parent_tensordict().get_parent_tensordict()
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_mask_td(device):
    torch.manual_seed(1)
    d = {
        "key1": torch.randn(4, 5, 6, device=device),
        "key2": torch.randn(4, 5, 10, device=device),
    }
    mask = torch.zeros(4, 5, dtype=torch.bool, device=device).bernoulli_()
    mask_list = mask.cpu().numpy().tolist()
    td = TensorDict(batch_size=(4, 5), source=d)

    td_masked = torch.masked_select(td, mask)
    td_masked1 = td[mask_list]
    assert len(td_masked.get("key1")) == td_masked.shape[0]
    assert len(td_masked1.get("key1")) == td_masked1.shape[0]

    mask_list = [False, True, False, True]

    td_masked2 = td[mask_list, 0]
    torch.testing.assert_allclose(td.get("key1")[mask_list, 0], td_masked2.get("key1"))
    torch.testing.assert_allclose(td.get("key2")[mask_list, 0], td_masked2.get("key2"))


@pytest.mark.parametrize("device", get_available_devices())
def test_unbind_td(device):
    torch.manual_seed(1)
    d = {
        "key1": torch.randn(4, 5, 6, device=device),
        "key2": torch.randn(4, 5, 10, device=device),
    }
    td = TensorDict(batch_size=(4, 5), source=d)
    td_unbind = torch.unbind(td, dim=1)
    assert (
        td_unbind[0].batch_size == td[:, 0].batch_size
    ), f"got {td_unbind[0].batch_size} and {td[:, 0].batch_size}"


@pytest.mark.parametrize("device", get_available_devices())
def test_cat_td(device):
    torch.manual_seed(1)
    d = {
        "key1": torch.randn(4, 5, 6, device=device),
        "key2": torch.randn(4, 5, 10, device=device),
    }
    td1 = TensorDict(batch_size=(4, 5), source=d)
    d = {
        "key1": torch.randn(4, 10, 6, device=device),
        "key2": torch.randn(4, 10, 10, device=device),
    }
    td2 = TensorDict(batch_size=(4, 10), source=d)

    td_cat = torch.cat([td1, td2], 1)
    assert td_cat.batch_size == torch.Size([4, 15])
    d = {"key1": torch.randn(4, 15, 6), "key2": torch.randn(4, 15, 10)}
    td_out = TensorDict(batch_size=(4, 15), source=d)
    torch.cat([td1, td2], 1, out=td_out)
    assert td_out.batch_size == torch.Size([4, 15])


@pytest.mark.parametrize("device", get_available_devices())
def test_expand(device):
    torch.manual_seed(1)
    d = {
        "key1": torch.randn(4, 5, 6, device=device),
        "key2": torch.randn(4, 5, 10, device=device),
    }
    td1 = TensorDict(batch_size=(4, 5), source=d)
    td2 = td1.expand(3, 7)
    assert td2.batch_size == torch.Size([3, 7, 4, 5])
    assert td2.get("key1").shape == torch.Size([3, 7, 4, 5, 6])
    assert td2.get("key2").shape == torch.Size([3, 7, 4, 5, 10])


@pytest.mark.parametrize("device", get_available_devices())
def test_squeeze(device):
    torch.manual_seed(1)
    d = {
        "key1": torch.randn(4, 5, 6, device=device),
        "key2": torch.randn(4, 5, 10, device=device),
    }
    td1 = TensorDict(batch_size=(4, 5), source=d)
    td2 = torch.unsqueeze(td1, dim=1)
    assert td2.batch_size == torch.Size([4, 1, 5])

    td1b = torch.squeeze(td2, dim=1)
    assert td1b.batch_size == td1.batch_size


@pytest.mark.parametrize("device", get_available_devices())
def test_permute(device):
    torch.manual_seed(1)
    d = {
        "a": torch.randn(4, 5, 6, 9, device=device),
        "b": torch.randn(4, 5, 6, 7, device=device),
        "c": torch.randn(4, 5, 6, device=device),
    }
    td1 = TensorDict(batch_size=(4, 5, 6), source=d)
    td2 = torch.permute(td1, dims=(2, 1, 0))
    assert td2.shape == torch.Size((6, 5, 4))
    assert td2["a"].shape == torch.Size((6, 5, 4, 9))

    td2 = torch.permute(td1, dims=(-1, -3, -2))
    assert td2.shape == torch.Size((6, 4, 5))
    assert td2["c"].shape == torch.Size((6, 4, 5, 1))

    td2 = torch.permute(td1, dims=(0, 1, 2))
    assert td2["a"].shape == torch.Size((4, 5, 6, 9))

    t = TensorDict({"a": torch.randn(3, 4, 1)}, [3, 4])
    torch.permute(t, dims=(1, 0)).set("b", torch.randn(4, 3))
    assert t["b"].shape == torch.Size((3, 4, 1))

    torch.permute(t, dims=(1, 0)).fill_("a", 0.0)
    assert torch.sum(t["a"]) == torch.Tensor([0])


@pytest.mark.parametrize("device", get_available_devices())
def test_permute_exceptions(device):
    torch.manual_seed(1)
    d = {
        "a": torch.randn(4, 5, 6, 7, device=device),
        "b": torch.randn(4, 5, 6, 8, 9, device=device),
    }
    td1 = TensorDict(batch_size=(4, 5, 6), source=d)

    with pytest.raises(RuntimeError) as e_info:
        td2 = td1.permute(1, 1, 0)
        _ = td2.shape

    with pytest.raises(RuntimeError) as e_info:
        td2 = td1.permute(3, 2, 1, 0)
        _ = td2.shape

    with pytest.raises(RuntimeError) as e_info:
        td2 = td1.permute(2, -1, 0)
        _ = td2.shape

    with pytest.raises(IndexError) as e_info:
        td2 = td1.permute(2, 3, 0)
        _ = td2.shape

    with pytest.raises(IndexError) as e_info:
        td2 = td1.permute(2, -4, 0)
        _ = td2.shape

    with pytest.raises(RuntimeError) as e_info:
        td2 = td1.permute(2, 1)
        _ = td2.shape


@pytest.mark.parametrize("device", get_available_devices())
def test_permute_with_tensordict_operations(device):
    torch.manual_seed(1)
    d = {
        "a": torch.randn(20, 6, 9, device=device),
        "b": torch.randn(20, 6, 7, device=device),
        "c": torch.randn(20, 6, device=device),
    }
    td1 = TensorDict(batch_size=(20, 6), source=d).view(4, 5, 6).permute(2, 1, 0)
    assert td1.shape == torch.Size((6, 5, 4))

    d = {
        "a": torch.randn(4, 5, 6, 9, device=device),
        "b": torch.randn(4, 5, 6, 7, device=device),
        "c": torch.randn(4, 5, 6, device=device),
    }
    td1 = (
        TensorDict(batch_size=(4, 5, 6), source=d).to(SavedTensorDict).permute(2, 1, 0)
    )
    assert td1.shape == torch.Size((6, 5, 4))

    d = {
        "a": torch.randn(4, 5, 6, 7, 9, device=device),
        "b": torch.randn(4, 5, 6, 7, 7, device=device),
        "c": torch.randn(4, 5, 6, 7, device=device),
    }
    td1 = TensorDict(batch_size=(4, 5, 6, 7), source=d)[
        :, :, :, torch.tensor([1, 2])
    ].permute(3, 2, 1, 0)
    assert td1.shape == torch.Size((2, 6, 5, 4))

    d = {
        "a": torch.randn(4, 5, 9, device=device),
        "b": torch.randn(4, 5, 7, device=device),
        "c": torch.randn(4, 5, device=device),
    }
    td1 = stack_td(
        [TensorDict(batch_size=(4, 5), source=d).clone() for _ in range(6)],
        2,
        contiguous=False,
    ).permute(2, 1, 0)
    assert td1.shape == torch.Size((6, 5, 4))


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("stack_dim", [0, 1])
def test_stacked_td(stack_dim, device):
    tensordicts = [
        TensorDict(
            batch_size=[11, 12],
            source={
                "key1": torch.randn(11, 12, 5, device=device),
                "key2": torch.zeros(
                    11, 12, 50, device=device, dtype=torch.bool
                ).bernoulli_(),
            },
        )
        for _ in range(10)
    ]

    tensordicts0 = tensordicts[0]
    tensordicts1 = tensordicts[1]
    tensordicts2 = tensordicts[2]
    tensordicts3 = tensordicts[3]
    sub_td = LazyStackedTensorDict(*tensordicts, stack_dim=stack_dim)

    std_bis = stack_td(tensordicts, dim=stack_dim, contiguous=False)
    assert (sub_td == std_bis).all()

    item = tuple([*[slice(None) for _ in range(stack_dim)], 0])
    tensordicts0.zero_()
    assert (sub_td[item].get("key1") == sub_td.get("key1")[item]).all()
    assert (
        sub_td.contiguous()[item].get("key1") == sub_td.contiguous().get("key1")[item]
    ).all()
    assert (sub_td.contiguous().get("key1")[item] == 0).all()

    item = tuple([*[slice(None) for _ in range(stack_dim)], 1])
    std2 = sub_td[:5]
    tensordicts1.zero_()
    assert (std2[item].get("key1") == std2.get("key1")[item]).all()
    assert (
        std2.contiguous()[item].get("key1") == std2.contiguous().get("key1")[item]
    ).all()
    assert (std2.contiguous().get("key1")[item] == 0).all()

    std3 = sub_td[:5, :, :5]
    tensordicts2.zero_()
    item = tuple([*[slice(None) for _ in range(stack_dim)], 2])
    assert (std3[item].get("key1") == std3.get("key1")[item]).all()
    assert (
        std3.contiguous()[item].get("key1") == std3.contiguous().get("key1")[item]
    ).all()
    assert (std3.contiguous().get("key1")[item] == 0).all()

    std4 = sub_td.select("key1")
    tensordicts3.zero_()
    item = tuple([*[slice(None) for _ in range(stack_dim)], 3])
    assert (std4[item].get("key1") == std4.get("key1")[item]).all()
    assert (
        std4.contiguous()[item].get("key1") == std4.contiguous().get("key1")[item]
    ).all()
    assert (std4.contiguous().get("key1")[item] == 0).all()

    std5 = sub_td.unbind(1)[0]
    assert (std5.contiguous() == sub_td.contiguous().unbind(1)[0]).all()


@pytest.mark.parametrize("device", get_available_devices())
def test_savedtensordict(device):
    vals = [torch.randn(3, 1, device=device) for _ in range(4)]
    ss_list = [
        SavedTensorDict(
            source=TensorDict(
                source={"a": vals[i]},
                batch_size=[
                    3,
                ],
            )
        )
        for i in range(4)
    ]
    ss = stack_td(ss_list, 0)
    assert ss_list[1] is ss[1]
    torch.testing.assert_allclose(ss_list[1].get("a"), vals[1])
    torch.testing.assert_allclose(ss_list[1].get("a"), ss[1].get("a"))
    torch.testing.assert_allclose(ss[1].get("a"), ss.get("a")[1])
    assert ss.get("a").device == device


@pytest.mark.parametrize(
    "ellipsis_index, expected_index",
    [
        (..., (slice(None), slice(None), slice(None), slice(None), slice(None))),
        ((0, ..., 0), (0, slice(None), slice(None), slice(None), 0)),
        ((..., 0), (slice(None), slice(None), slice(None), slice(None), 0)),
        ((0, ...), (0, slice(None), slice(None), slice(None), slice(None))),
        (
            (slice(1, 2), ...),
            (slice(1, 2), slice(None), slice(None), slice(None), slice(None)),
        ),
    ],
)
def test_convert_ellipsis_to_idx_valid(ellipsis_index, expected_index):
    torch.manual_seed(1)
    batch_size = [3, 4, 5, 6, 7]

    assert convert_ellipsis_to_idx(ellipsis_index, batch_size) == expected_index


@pytest.mark.parametrize(
    "ellipsis_index, expectation",
    [
        ((..., 0, ...), pytest.raises(RuntimeError)),
        ((0, ..., 0, ...), pytest.raises(RuntimeError)),
    ],
)
def test_convert_ellipsis_to_idx_invalid(ellipsis_index, expectation):
    torch.manual_seed(1)
    batch_size = [3, 4, 5, 6, 7]

    with expectation:
        _ = convert_ellipsis_to_idx(ellipsis_index, batch_size)


TD_BATCH_SIZE = 4


@pytest.mark.parametrize(
    "td_name",
    [
        "td",
        "stacked_td",
        "sub_td",
        "sub_td2",
        "idx_td",
        "saved_td",
        "memmap_td",
        "unsqueezed_td",
        "squeezed_td",
        "td_reset_bs",
        "nested_td",
        "permute_td",
    ],
)
@pytest.mark.parametrize("device", get_available_devices())
class TestTensorDicts:
    def td(self, device):
        return TensorDict(
            source={
                "a": torch.randn(4, 3, 2, 1, 5, device=device),
                "b": torch.randn(4, 3, 2, 1, 10, device=device),
                "c": torch.randint(10, (4, 3, 2, 1, 3), device=device),
            },
            batch_size=[4, 3, 2, 1],
        )

    def nested_td(self, device):
        return TensorDict(
            source={
                "a": torch.randn(4, 3, 2, 1, 5, device=device),
                "b": torch.randn(4, 3, 2, 1, 10, device=device),
                "c": torch.randint(10, (4, 3, 2, 1, 3), device=device),
                "my_nested_td": TensorDict(
                    {"inner": torch.randn(4, 3, 2, 1, 2, device=device)}, [4, 3, 2, 1]
                ),
            },
            batch_size=[4, 3, 2, 1],
        )

    def stacked_td(self, device):
        td1 = TensorDict(
            source={
                "a": torch.randn(4, 3, 1, 5, device=device),
                "b": torch.randn(4, 3, 1, 10, device=device),
                "c": torch.randint(10, (4, 3, 1, 3), device=device),
            },
            batch_size=[4, 3, 1],
        )
        td2 = TensorDict(
            source={
                "a": torch.randn(4, 3, 1, 5, device=device),
                "b": torch.randn(4, 3, 1, 10, device=device),
                "c": torch.randint(10, (4, 3, 1, 3), device=device),
            },
            batch_size=[4, 3, 1],
        )
        return stack_td([td1, td2], 2)

    def idx_td(self, device):
        td = TensorDict(
            source={
                "a": torch.randn(2, 4, 3, 2, 1, 5, device=device),
                "b": torch.randn(2, 4, 3, 2, 1, 10, device=device),
                "c": torch.randint(10, (2, 4, 3, 2, 1, 3), device=device),
            },
            batch_size=[2, 4, 3, 2, 1],
        )
        return td[1]

    def sub_td(self, device):
        td = TensorDict(
            source={
                "a": torch.randn(2, 4, 3, 2, 1, 5, device=device),
                "b": torch.randn(2, 4, 3, 2, 1, 10, device=device),
                "c": torch.randint(10, (2, 4, 3, 2, 1, 3), device=device),
            },
            batch_size=[2, 4, 3, 2, 1],
        )
        return td.get_sub_tensordict(1)

    def sub_td2(self, device):
        td = TensorDict(
            source={
                "a": torch.randn(4, 2, 3, 2, 1, 5, device=device),
                "b": torch.randn(4, 2, 3, 2, 1, 10, device=device),
                "c": torch.randint(10, (4, 2, 3, 2, 1, 3), device=device),
            },
            batch_size=[4, 2, 3, 2, 1],
        )
        return td.get_sub_tensordict((slice(None), 1))

    def saved_td(self, device):
        return SavedTensorDict(source=self.td(device))

    def memmap_td(self, device):
        return self.td(device).memmap_()

    def permute_td(self, device):
        return TensorDict(
            source={
                "a": torch.randn(3, 1, 4, 2, 5, device=device),
                "b": torch.randn(3, 1, 4, 2, 10, device=device),
                "c": torch.randint(10, (3, 1, 4, 2, 3), device=device),
            },
            batch_size=[3, 1, 4, 2],
        ).permute(2, 0, 3, 1)
        # return TensorDict(
        #     source={
        #         "a": torch.randn(3, 1, 2, 4, 5, device=device),
        #         "b": torch.randn(3, 1, 2, 4, 10, device=device),
        #         "c": torch.randint(10, (3, 1, 2, 4, 3), device=device),
        #     },
        #     batch_size=[3, 1, 2, 4],
        # ).permute(2, 0, 1, 3)

    def unsqueezed_td(self, device):
        td = TensorDict(
            source={
                "a": torch.randn(4, 3, 2, 5, device=device),
                "b": torch.randn(4, 3, 2, 10, device=device),
                "c": torch.randint(10, (4, 3, 2, 3), device=device),
            },
            batch_size=[4, 3, 2],
        )
        return td.unsqueeze(-1)

    def squeezed_td(self, device):
        td = TensorDict(
            source={
                "a": torch.randn(4, 3, 1, 2, 1, 5, device=device),
                "b": torch.randn(4, 3, 1, 2, 1, 10, device=device),
                "c": torch.randint(10, (4, 3, 1, 2, 1, 3), device=device),
            },
            batch_size=[4, 3, 1, 2, 1],
        )
        return td.squeeze(2)

    def td_reset_bs(self, device):
        td = TensorDict(
            source={
                "a": torch.randn(4, 3, 2, 1, 5, device=device),
                "b": torch.randn(4, 3, 2, 1, 10, device=device),
                "c": torch.randint(10, (4, 3, 2, 1, 3), device=device),
            },
            batch_size=[4, 3, 2],
        )
        td.batch_size = torch.Size([4, 3, 2, 1])
        return td

    def test_to_tensordict(self, td_name, device):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        td2 = td.to_tensordict()
        assert (td2 == td).all()

    def test_select(self, td_name, device):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        td2 = td.select("a")
        assert td2 is not td
        assert len(list(td2.keys())) == 1 and "a" in td2.keys()
        assert len(list(td2.clone().keys())) == 1 and "a" in td2.clone().keys()

        td2 = td.select("a", inplace=True)
        assert td2 is td

    def test_exclude(self, td_name, device):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        td2 = td.exclude("a")
        assert td2 is not td
        assert (
            len(list(td2.keys())) == len(list(td.keys())) - 1 and "a" not in td2.keys()
        )
        assert (
            len(list(td2.clone().keys())) == len(list(td.keys())) - 1
            and "a" not in td2.clone().keys()
        )

        td2 = td.exclude("a", inplace=True)
        assert td2 is td

    def test_assert(self, td_name, device):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        with pytest.raises(
            ValueError,
            match="Converting a tensordict to boolean value is not permitted",
        ):
            assert td

    def test_expand(self, td_name, device):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        batch_size = td.batch_size
        new_td = td.expand(3)
        assert new_td.batch_size == torch.Size([3, *batch_size])
        assert all((_new_td == td).all() for _new_td in new_td)

    def test_cast(self, td_name, device):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        td_td = td.to(TensorDict)
        assert (td == td_td).all()

        td = getattr(self, td_name)(device)
        td_saved = td.to(SavedTensorDict)
        assert (td == td_saved).all()

    @pytest.mark.parametrize("call_del", [True, False])
    def test_remove(self, td_name, device, call_del):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        if call_del:
            del td["a"]
        else:
            td = td.del_("a")
        assert td is not None
        assert "a" not in td.keys()

    def test_set_unexisting(self, td_name, device):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        td.set("z", torch.ones_like(td.get("a")))
        assert (td.get("z") == 1).all()

    def test_fill_(self, td_name, device):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        new_td = td.fill_("a", 0.1)
        assert (td.get("a") == 0.1).all()
        assert new_td is td

    def test_masked_fill_(self, td_name, device):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        mask = torch.zeros(td.shape, dtype=torch.bool, device=device).bernoulli_()
        new_td = td.masked_fill_(mask, -10.0)
        assert new_td is td
        for k, item in td.items():
            assert (item[mask] == -10).all(), item[mask]

    def test_masked_fill(self, td_name, device):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        mask = torch.zeros(td.shape, dtype=torch.bool, device=device).bernoulli_()
        new_td = td.masked_fill(mask, -10.0)
        assert new_td is not td
        for k, item in new_td.items():
            assert (item[mask] == -10).all()

    def test_zero_(self, td_name, device):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        new_td = td.zero_()
        assert new_td is td
        for k in td.keys():
            assert (td.get(k) == 0).all()

    def test_from_empty(self, td_name, device):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        new_td = TensorDict({}, batch_size=td.batch_size, device=device)
        for key, item in td.items():
            new_td.set(key, item)
        assert_allclose_td(td, new_td)
        assert td.device == new_td.device
        assert td.shape == new_td.shape

    def test_masking(self, td_name, device):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        mask = torch.zeros(td.batch_size, dtype=torch.bool, device=device).bernoulli_(
            0.8
        )
        td_masked = td[mask]
        td_masked2 = torch.masked_select(td, mask)
        assert_allclose_td(td_masked, td_masked2)
        assert td_masked.batch_size[0] == mask.sum()
        assert td_masked.batch_dims == 1

        mask_list = mask.cpu().numpy().tolist()
        td_masked3 = td[mask_list]
        assert_allclose_td(td_masked3, td_masked2)
        assert td_masked3.batch_size[0] == mask.sum()
        assert td_masked3.batch_dims == 1

    @pytest.mark.parametrize("from_list", [True, False])
    def test_masking_set(self, td_name, device, from_list):
        def zeros_like(item, n, d):
            if isinstance(item, (MemmapTensor, torch.Tensor)):
                return torch.zeros(n, *item.shape[d:], dtype=item.dtype, device=device)
            elif isinstance(item, _TensorDict):
                batch_size = item.batch_size
                batch_size = [n, *batch_size[d:]]
                out = TensorDict(
                    {k: zeros_like(_item, n, d) for k, _item in item.items()},
                    batch_size,
                    device=device,
                )
                return out

        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        mask = torch.zeros(td.batch_size, dtype=torch.bool, device=device).bernoulli_(
            0.8
        )
        n = mask.sum()
        d = td.ndimension()
        pseudo_td = TensorDict(
            {k: zeros_like(item, n, d) for k, item in td.items()}, [n], device=device
        )
        if from_list:
            td_mask = mask.cpu().numpy().tolist()
        else:
            td_mask = mask
        if td_name == "stacked_td":
            with pytest.raises(RuntimeError, match="is not supported"):
                td[td_mask] = pseudo_td
        else:
            td[td_mask] = pseudo_td
            for k, item in td.items():
                assert (item[mask] == 0).all()

    @pytest.mark.skipif(
        torch.cuda.device_count() == 0, reason="No cuda device detected"
    )
    @pytest.mark.parametrize("device_cast", [0, "cuda:0", torch.device("cuda:0")])
    def test_pin_memory(self, td_name, device_cast, device):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        if td_name != "saved_td":
            td.pin_memory()
            td_device = td.to(device_cast)
            _device_cast = torch.device(device_cast)
            assert td_device.device == _device_cast
            assert td_device.clone().device == _device_cast
            if device != _device_cast:
                assert td_device is not td
            for k, item in td_device.items():
                assert item.device == _device_cast
            for k, item in td_device.clone().items():
                assert item.device == _device_cast
            # assert type(td_device) is type(td)
            assert_allclose_td(td, td_device.to(device))
        else:
            with pytest.raises(
                RuntimeError,
                match="pin_memory requires tensordicts that live in memory",
            ):
                td.pin_memory()

    @pytest.mark.skipif(
        torch.cuda.device_count() == 0, reason="No cuda device detected"
    )
    @pytest.mark.parametrize("device_cast", get_available_devices())
    def test_cast_device(self, td_name, device, device_cast):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        td_device = td.to(device_cast)

        for k, item in td_device.items_meta():
            assert item.device == device_cast
        for k, item in td_device.items():
            assert item.device == device_cast
        for k, item in td_device.clone().items():
            assert item.device == device_cast

        assert td_device.device == device_cast, (
            f"td_device first tensor device is " f"{next(td_device.items())[1].device}"
        )
        assert td_device.clone().device == device_cast
        if device_cast != td.device:
            assert td_device is not td
        assert td_device.to(device_cast) is td_device
        assert td.to(device) is td
        assert_allclose_td(td, td_device.to(device))

    @pytest.mark.skipif(
        torch.cuda.device_count() == 0, reason="No cuda device detected"
    )
    def test_cpu_cuda(self, td_name, device):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        td_device = td.cuda()
        td_back = td_device.cpu()
        assert td_device.device == torch.device("cuda:0")
        assert td_back.device == torch.device("cpu")

    def test_unbind(self, td_name, device):
        if td_name not in ["sub_td", "idx_td", "td_reset_bs"]:
            torch.manual_seed(1)
            td = getattr(self, td_name)(device)
            td_unbind = torch.unbind(td, dim=0)
            assert (td == stack_td(td_unbind, 0).contiguous()).all()
            assert (td[0] == td_unbind[0]).all()

    @pytest.mark.parametrize("squeeze_dim", [0, 1])
    def test_unsqueeze(self, td_name, device, squeeze_dim):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        td_unsqueeze = torch.unsqueeze(td, dim=squeeze_dim)
        tensor = torch.ones_like(td.get("a").unsqueeze(squeeze_dim))
        if td_name in ("sub_td", "sub_td2"):
            td_unsqueeze.set_("a", tensor)
        else:
            td_unsqueeze.set("a", tensor)
        assert (td_unsqueeze.get("a") == tensor).all()
        assert (td.get("a") == tensor.squeeze(squeeze_dim)).all()
        assert td_unsqueeze.squeeze(squeeze_dim) is td
        assert (td_unsqueeze.get("a") == 1).all()
        assert (td.get("a") == 1).all()

    def test_squeeze(self, td_name, device, squeeze_dim=-1):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        td_squeeze = torch.squeeze(td, dim=-1)
        tensor_squeeze_dim = td.batch_dims + squeeze_dim
        tensor = torch.ones_like(td.get("a").squeeze(tensor_squeeze_dim))
        if td_name in ("sub_td", "sub_td2"):
            td_squeeze.set_("a", tensor)
        else:
            td_squeeze.set("a", tensor)
        assert td.batch_size[squeeze_dim] == 1
        assert (td_squeeze.get("a") == tensor).all()
        assert (td.get("a") == tensor.unsqueeze(tensor_squeeze_dim)).all()
        if td_name != "unsqueezed_td":
            assert td_squeeze.unsqueeze(squeeze_dim) is td
        else:
            assert td_squeeze is td._source
        assert (td_squeeze.get("a") == 1).all()
        assert (td.get("a") == 1).all()

    def test_pad(self, td_name, device):
        td = getattr(self, td_name)(device)
        paddings = [
            [0, 1, 0, 2],
            [1, 0, 0, 2],
            [1, 0, 2, 1],
        ]

        for pad_size in paddings:
            padded_td = pad(td, pad_size)
            padded_td._check_batch_size()
            amount_expanded = [0] * (len(pad_size) // 2)
            for i in range(0, len(pad_size), 2):
                amount_expanded[i // 2] = pad_size[i] + pad_size[i + 1]

            for key in padded_td.keys():
                expected_dims = tuple(
                    sum(p)
                    for p in zip(
                        td[key].shape,
                        amount_expanded
                        + [0] * (len(td[key].shape) - len(amount_expanded)),
                    )
                )
                assert padded_td[key].shape == expected_dims

        with pytest.raises(RuntimeError):
            pad(td, [0] * 100)

        with pytest.raises(RuntimeError):
            pad(td, [0])

    def test_view(self, td_name, device):
        if td_name in ("permute_td", "sub_td2"):
            pytest.skip("view incompatible with stride / permutation")
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        td_view = td.view(-1)
        tensor = td.get("a")
        tensor = tensor.view(-1, tensor.numel() // prod(td.batch_size))
        tensor = torch.ones_like(tensor)
        if td_name == "sub_td":
            td_view.set_("a", tensor)
        else:
            td_view.set("a", tensor)
        assert (td_view.get("a") == tensor).all()
        assert (td.get("a") == tensor.view(td.get("a").shape)).all()
        assert td_view.view(td.shape) is td
        assert td_view.view(*td.shape) is td
        assert (td_view.get("a") == 1).all()
        assert (td.get("a") == 1).all()

    def test_clone_td(self, td_name, device):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        assert (torch.clone(td) == td).all()
        assert td.batch_size == torch.clone(td).batch_size
        if td_name in (
            "stacked_td",
            "saved_td",
            "squeezed_td",
            "unsqueezed_td",
            "sub_td",
            "sub_td2",
            "permute_td",
        ):
            with pytest.raises(AssertionError):
                assert td.clone(recursive=False).get("a") is td.get("a")
        else:
            assert td.clone(recursive=False).get("a") is td.get("a")

    def test_rename_key(self, td_name, device) -> None:
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        with pytest.raises(KeyError, match="already present in TensorDict"):
            td.rename_key("a", "b", safe=True)
        a = td.get("a")
        td.rename_key("a", "z")
        with pytest.raises(KeyError):
            td.get("a")
        assert "a" not in td.keys()

        z = td.get("z")
        if isinstance(a, MemmapTensor):
            a = a._tensor
        if isinstance(z, MemmapTensor):
            z = z._tensor
        torch.testing.assert_allclose(a, z)

        new_z = torch.randn_like(z)
        if td_name in ("sub_td", "sub_td2"):
            td.set_("z", new_z)
        else:
            td.set("z", new_z)

        torch.testing.assert_allclose(new_z, td.get("z"))

        new_z = torch.randn_like(z)
        td.set_("z", new_z)
        torch.testing.assert_allclose(new_z, td.get("z"))

    def test_set_nontensor(self, td_name, device):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        r = torch.randn_like(td.get("a"))
        td.set("numpy", r.cpu().numpy())
        torch.testing.assert_allclose(td.get("numpy"), r)

    @pytest.mark.parametrize(
        "actual_index,expected_index",
        [
            (..., (slice(None),) * TD_BATCH_SIZE),
            ((..., 0), (slice(None),) * (TD_BATCH_SIZE - 1) + (0,)),
            ((0, ...), (0,) + (slice(None),) * (TD_BATCH_SIZE - 1)),
            ((0, ..., 0), (0,) + (slice(None),) * (TD_BATCH_SIZE - 2) + (0,)),
        ],
    )
    def test_getitem_ellipsis(self, td_name, device, actual_index, expected_index):
        torch.manual_seed(1)

        td = getattr(self, td_name)(device)

        actual_td = td[actual_index]
        expected_td = td[expected_index]
        assert expected_td.shape == _getitem_batch_size(
            td.batch_size, convert_ellipsis_to_idx(actual_index, td.batch_size)
        )
        assert_allclose_td(actual_td, expected_td)

    @pytest.mark.parametrize("actual_index", [..., (..., 0), (0, ...), (0, ..., 0)])
    def test_setitem_ellipsis(self, td_name, device, actual_index):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)

        idx = actual_index
        td_clone = td.clone()
        actual_td = td_clone[idx].clone().zero_()

        for key in actual_td.keys():
            assert (actual_td.get(key) == 0).all()
        td_clone[idx] = actual_td
        for key in td_clone.keys():
            assert (td_clone[idx].get(key) == 0).all()

    @pytest.mark.parametrize("idx", [slice(1), torch.tensor([0]), torch.tensor([0, 1])])
    def test_setitem(self, td_name, device, idx):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        if isinstance(idx, torch.Tensor) and idx.numel() > 1 and td.shape[0] == 1:
            pytest.mark.skip("cannot index tensor with desired index")
            return

        td_clone = td[idx].clone().zero_()
        td[idx] = td_clone
        assert (td[idx].get("a") == 0).all()

        td_clone = torch.cat([td_clone, td_clone], 0)
        with pytest.raises(RuntimeError, match="differs from the source batch size"):
            td[idx] = td_clone

    def test_setitem_string(self, td_name, device):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        td["d"] = torch.randn(4, 3, 2, 1, 5)
        assert "d" in td.keys()

    def test_getitem_string(self, td_name, device):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        assert isinstance(td["a"], (MemmapTensor, torch.Tensor))

    def test_delitem(self, td_name, device):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        del td["a"]
        assert "a" not in td.keys()

    @pytest.mark.filterwarnings("error")
    def test_stack_tds_on_subclass(self, td_name, device):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        tds_count = td.batch_size[0]
        tds_batch_size = td.batch_size[1:]
        tds_list = [
            TensorDict(
                source={
                    "a": torch.ones(*tds_batch_size, 5),
                    "b": torch.ones(*tds_batch_size, 10),
                    "c": torch.ones(*tds_batch_size, 3, dtype=torch.long),
                },
                batch_size=tds_batch_size,
                device=device,
            )
            for _ in range(tds_count)
        ]
        stacked_td = torch.stack(tds_list, 0, out=td)
        assert stacked_td.batch_size == td.batch_size
        assert stacked_td is td
        for key in ("a", "b", "c"):
            assert (stacked_td[key] == 1).all()

    @pytest.mark.filterwarnings("error")
    def test_stack_subclasses_on_td(self, td_name, device):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        td = td.expand(3).to_tensordict().clone().zero_()
        tds_list = [getattr(self, td_name)(device) for _ in range(3)]
        stacked_td = stack_td(tds_list, 0, out=td)
        assert stacked_td.batch_size == td.batch_size
        for key in ("a", "b", "c"):
            assert (stacked_td[key] == td[key]).all()

    @pytest.mark.parametrize("dim", [0, 1])
    @pytest.mark.parametrize("chunks", [1, 2])
    def test_chunk(self, td_name, device, dim, chunks):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        if len(td.shape) - 1 < dim:
            pytest.mark.skip(f"no dim {dim} in td")
            return

        chunks = min(td.shape[dim], chunks)
        td_chunks = td.chunk(chunks, dim)
        assert len(td_chunks) == chunks
        assert sum([_td.shape[dim] for _td in td_chunks]) == td.shape[dim]
        assert (torch.cat(td_chunks, dim) == td).all()

    def test_items_values_keys(self, td_name, device):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        keys = list(td.keys())
        values = list(td.values())
        items = list(td.items())

        # Test td.items()
        constructed_td1 = TensorDict({}, batch_size=td.shape)
        for key, value in items:
            constructed_td1.set(key, value)

        assert (td == constructed_td1).all()

        # Test td.keys() and td.values()
        # items = [key, value] should be verified
        assert len(values) == len(items)
        assert len(keys) == len(items)
        constructed_td2 = TensorDict({}, batch_size=td.shape)
        for key, value in list(zip(td.keys(), td.values())):
            constructed_td2.set(key, value)

        assert (td == constructed_td2).all()

        # Test that keys is sorted
        assert all(keys[i] <= keys[i + 1] for i in range(len(keys) - 1))

        # Add new element to tensor
        a = td.get("a")
        td.set("x", torch.randn_like(a))
        keys = list(td.keys())
        values = list(td.values())
        items = list(td.items())

        # Test that keys is still sorted after adding the element
        assert all(keys[i] <= keys[i + 1] for i in range(len(keys) - 1))

        # Test td.items()
        # after adding the new element
        constructed_td1 = TensorDict({}, batch_size=td.shape)
        for key, value in items:
            constructed_td1.set(key, value)

        assert (td == constructed_td1).all()

        # Test td.keys() and td.values()
        # items = [key, value] should be verified
        # even after adding the new element
        assert len(values) == len(items)
        assert len(keys) == len(items)

        constructed_td2 = TensorDict({}, batch_size=td.shape)
        for key, value in list(zip(td.keys(), td.values())):
            constructed_td2.set(key, value)

        assert (td == constructed_td2).all()

        # Test the methods values_meta() and items_meta()
        values_meta = list(td.values_meta())
        items_meta = list(td.items_meta())
        assert len(values_meta) == len(items_meta)

    def test_set_requires_grad(self, td_name, device):
        td = getattr(self, td_name)(device)
        assert not td.get("a").requires_grad
        assert not td._get_meta("a").requires_grad
        if td_name in ("sub_td", "sub_td2"):
            td.set_("a", torch.randn_like(td.get("a")).requires_grad_())
        else:
            td.set("a", torch.randn_like(td.get("a")).requires_grad_())
        assert td.get("a").requires_grad
        assert td._get_meta("a").requires_grad

    def test_nested_td_emptyshape(self, td_name, device):
        td = getattr(self, td_name)(device)
        tdin = TensorDict({"inner": torch.randn(*td.shape, 1)}, [], device=device)
        td["inner_td"] = tdin
        tdin.batch_size = td.batch_size
        assert (td["inner_td"] == tdin).all()

    def test_nested_td(self, td_name, device):
        td = getattr(self, td_name)(device)
        tdin = TensorDict({"inner": torch.randn(td.shape)}, td.shape, device=device)
        td.set("inner_td", tdin)
        assert (td["inner_td"] == tdin).all()

    def test_nested_td_index(self, td_name, device):
        td = getattr(self, td_name)(device)

        sub_td = TensorDict({}, [*td.shape, 2], device=device)
        a = torch.zeros([*td.shape, 2, 2], device=device)
        sub_sub_td = TensorDict({"a": a}, [*td.shape, 2, 2], device=device)
        sub_td.set("sub_sub_td", sub_sub_td)
        td.set("sub_td", sub_td)
        assert (td["sub_td", "sub_sub_td", "a"] == 0).all()
        assert (
            td["sub_td"]["sub_sub_td"]["a"] == td["sub_td", "sub_sub_td", "a"]
        ).all()

        a = torch.ones_like(a)
        other_sub_sub_td = TensorDict({"a": a}, [*td.shape, 2, 2])
        td["sub_td", "sub_sub_td"] = other_sub_sub_td
        assert (td["sub_td", "sub_sub_td", "a"] == 1).all()
        assert (
            td["sub_td"]["sub_sub_td"]["a"] == td["sub_td", "sub_sub_td", "a"]
        ).all()

        b = torch.ones_like(a)
        other_sub_sub_td = TensorDict({"b": b}, [*td.shape, 2, 2])

        if td_name in ("sub_td", "sub_td2"):
            with pytest.raises(
                RuntimeError, match="with another one with non-matching keys"
            ):
                td["sub_td", "sub_sub_td"] = other_sub_sub_td
        else:
            td["sub_td", "sub_sub_td"] = other_sub_sub_td
            assert (td["sub_td", "sub_sub_td", "b"] == 1).all()
            assert (
                td["sub_td"]["sub_sub_td"]["b"] == td["sub_td", "sub_sub_td", "b"]
            ).all()

    @pytest.mark.parametrize("inplace", [True, False])
    @pytest.mark.parametrize("separator", [",", "-"])
    def test_flatten_keys(self, td_name, device, inplace, separator):
        td = getattr(self, td_name)(device)
        nested_nested_tensordict = TensorDict(
            {
                "a": torch.zeros(*td.shape, 2, 3),
            },
            [*td.shape, 2],
        )
        nested_tensordict = TensorDict(
            {
                "a": torch.zeros(*td.shape, 2),
                "nested_nested_tensordict": nested_nested_tensordict,
            },
            td.shape,
        )
        td["nested_tensordict"] = nested_tensordict

        td_flatten = td.flatten_keys(inplace=inplace, separator=separator)
        for key, value in td_flatten.items():
            assert not isinstance(value, _TensorDict)
        assert (
            separator.join(["nested_tensordict", "nested_nested_tensordict", "a"])
            in td_flatten.keys()
        )
        if inplace:
            assert td_flatten is td
        else:
            assert td_flatten is not td

    @pytest.mark.parametrize("inplace", [True, False])
    @pytest.mark.parametrize("separator", [",", "-"])
    def test_unflatten_keys(self, td_name, device, inplace, separator):
        td = getattr(self, td_name)(device)
        nested_nested_tensordict = TensorDict(
            {
                "a": torch.zeros(*td.shape, 2, 3),
            },
            [*td.shape, 2],
        )
        nested_tensordict = TensorDict(
            {
                "a": torch.zeros(*td.shape, 2),
                "nested_nested_tensordict": nested_nested_tensordict,
            },
            td.shape,
        )
        td["nested_tensordict"] = nested_tensordict

        td_flatten = td.flatten_keys(inplace=inplace, separator=separator)
        td_unflatten = td_flatten.unflatten_keys(inplace=inplace, separator=separator)
        assert (td == td_unflatten).all()
        if inplace:
            assert td is td_unflatten

    def test_repr(self, td_name, device):
        td = getattr(self, td_name)(device)
        _ = str(td)


@pytest.mark.parametrize(
    "td_name",
    [
        "td",
        "stacked_td",
        "sub_td",
        "idx_td",
        "unsqueezed_td",
        "td_reset_bs",
    ],
)
@pytest.mark.parametrize(
    "device",
    get_available_devices(),
)
class TestTensorDictsRequiresGrad:
    def td(self, device):
        return TensorDict(
            source={
                "a": torch.randn(3, 1, 5, device=device),
                "b": torch.randn(3, 1, 10, device=device, requires_grad=True),
                "c": torch.randint(10, (3, 1, 3), device=device),
            },
            batch_size=[3, 1],
        )

    def stacked_td(self, device):
        return stack_td([self.td(device) for _ in range(2)], 0)

    def idx_td(self, device):
        return self.td(device)[0]

    def sub_td(self, device):
        return self.td(device).get_sub_tensordict(0)

    def unsqueezed_td(self, device):
        return self.td(device).unsqueeze(0)

    def td_reset_bs(self, device):
        td = self.td(device)
        td = td.unsqueeze(-1).to_tensordict()
        td.batch_size = torch.Size([3, 1])
        return td

    def test_init_requires_grad(self, td_name, device):
        td = getattr(self, td_name)(device)
        assert td._get_meta("b").requires_grad

    def test_view(self, td_name, device):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        td_view = td.view(-1)
        assert td_view._get_meta("b").requires_grad

    def test_expand(self, td_name, device):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        batch_size = td.batch_size
        new_td = td.expand(3)
        assert new_td._get_meta("b").requires_grad

    def test_cast(self, td_name, device):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        td_td = td.to(TensorDict)
        assert td_td._get_meta("b").requires_grad

    def test_clone_td(self, td_name, device):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        assert torch.clone(td)._get_meta("b").requires_grad

    def test_squeeze(self, td_name, device, squeeze_dim=-1):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        assert torch.squeeze(td, dim=-1)._get_meta("b").requires_grad


def test_batchsize_reset():
    td = TensorDict(
        {"a": torch.randn(3, 4, 5, 6), "b": torch.randn(3, 4, 5)}, batch_size=[3, 4]
    )
    # smoke-test
    td.batch_size = torch.Size([3])

    # test with list
    td.batch_size = [3]

    # test with tuple
    td.batch_size = (3,)

    # incompatible size
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "the tensor a has shape torch.Size([3, 4, 5, "
            "6]) which is incompatible with the new shape torch.Size([3, 5])"
        ),
    ):
        td.batch_size = [3, 5]

    # test set
    td.set("c", torch.randn(3))

    # test index
    subtd = td[torch.tensor([1, 2])]
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "The shape torch.Size([3]) is incompatible with the index (slice(None, None, None), 0)."
        ),
    ):
        td[:, 0]

    # test a greater batch_size
    td = TensorDict(
        {"a": torch.randn(3, 4, 5, 6), "b": torch.randn(3, 4, 5)}, batch_size=[3, 4]
    )
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "TensorDict requires tensors that have at least one more dimension than the batch_size"
        ),
    ):
        td.batch_size = torch.Size([3, 4, 5])
    del td["b"]
    td.batch_size = torch.Size([3, 4, 5])

    td.set("c", torch.randn(3, 4, 5, 6))
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "batch dimension mismatch, "
            "got self.batch_size=torch.Size([3, 4, 5]) and tensor.shape[:self.batch_dims]=torch.Size([3, 4, 2])"
        ),
    ):
        td.set("d", torch.randn(3, 4, 2))

    # test with saved tensordict
    td = SavedTensorDict(TensorDict({"a": torch.randn(3, 4)}, [3, 4]))
    td.batch_size = [3]
    assert td.to_tensordict().batch_size == torch.Size([3])

    # test that lazy tds return an exception
    td_stack = stack_td([TensorDict({"a": torch.randn(3)}, [3]) for _ in range(2)])
    td_stack.to_tensordict().batch_size = [2]
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "modifying the batch size of a lazy repesentation "
            "of a tensordict is not permitted. Consider instantiating the tensordict fist by calling `td = td.to_tensordict()` before resetting the batch size."
        ),
    ):
        td_stack.batch_size = [2]

    td = TensorDict({"a": torch.randn(3, 4)}, [3, 4])
    subtd = td[:, torch.tensor([1, 2])]
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "modifying the batch size of a lazy repesentation of a tensordict is not permitted. Consider instantiating the tensordict fist by calling `td = td.to_tensordict()` before resetting the batch size."
        ),
    ):
        subtd.batch_size = [3, 2]
    subtd.to_tensordict().batch_size = [3, 2]

    td = TensorDict({"a": torch.randn(3, 4)}, [3, 4])
    td_u = td.unsqueeze(0)
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "modifying the batch size of a lazy repesentation of a tensordict is not permitted. Consider instantiating the tensordict fist by calling `td = td.to_tensordict()` before resetting the batch size."
        ),
    ):
        td_u.batch_size = [1]
    td_u.to_tensordict().batch_size = [1]


@pytest.mark.parametrize("index0", [None, slice(None)])
def test_set_sub_key(index0):
    # tests that parent tensordict is affected when subtensordict is set with a new key
    batch_size = [10, 10]
    source = {"a": torch.randn(10, 10, 10), "b": torch.ones(10, 10, 2)}
    td = TensorDict(source, batch_size=batch_size)
    idx0 = (index0, 0) if index0 is not None else 0
    td0 = td.get_sub_tensordict(idx0)
    idx = (index0, slice(2, 4)) if index0 is not None else slice(2, 4)
    sub_td = td.get_sub_tensordict(idx)
    if index0 is None:
        c = torch.randn(2, 10, 10)
    else:
        c = torch.randn(10, 2, 10)
    sub_td.set("c", c)
    assert (td.get("c")[idx] == sub_td.get("c")).all()
    assert (sub_td.get("c") == c).all()
    assert (td.get("c")[idx0] == 0).all()
    assert (td.get_sub_tensordict(idx0).get("c") == 0).all()
    assert (td0.get("c") == 0).all()


@pytest.mark.skipif(not torch.cuda.device_count(), reason="no cuda")
def test_create_on_device():
    device = torch.device(0)

    # TensorDict
    td = TensorDict({}, [5])
    with pytest.raises(RuntimeError):
        td.device
    td.set("a", torch.randn(5, device=device))
    assert td.device == device

    td = TensorDict({}, [5], device="cuda:0")
    td.set("a", torch.randn(5, 1))
    assert td.get("a").device == device

    # stacked TensorDict
    td1 = TensorDict({}, [5])
    td2 = TensorDict({}, [5])
    stackedtd = stack_td([td1, td2], 0)
    with pytest.raises(RuntimeError):
        stackedtd.device
    stackedtd.set("a", torch.randn(2, 5, device=device))
    assert stackedtd.device == device
    assert td1.device == device
    assert td2.device == device

    td1 = TensorDict({}, [5], device="cuda:0")
    td2 = TensorDict({}, [5], device="cuda:0")
    stackedtd = stack_td([td1, td2], 0)
    stackedtd.set("a", torch.randn(2, 5, 1))
    assert stackedtd.get("a").device == device
    assert td1.get("a").device == device
    assert td2.get("a").device == device

    # TensorDict, indexed
    td = TensorDict({}, [5])
    subtd = td[1]
    with pytest.raises(RuntimeError):
        subtd.device
    subtd.set("a", torch.randn(1, device=device))
    assert subtd.device == device

    td = TensorDict({}, [5], device="cuda:0")
    subtd = td[1]
    subtd.set("a", torch.randn(1))
    assert subtd.get("a").device == device

    # TensorDict, indexed, slice
    td = TensorDict({}, [5])
    subtd = td[1:3]
    with pytest.raises(RuntimeError):
        subtd.device
    subtd.set("a", torch.randn(2, device=device))
    assert subtd.device == device

    td = TensorDict({}, [5], device="cuda:0")
    subtd = td[1:3]
    subtd.set("a", torch.randn(2))
    assert subtd.get("a").device == device

    # SavedTensorDict
    td = TensorDict({}, [5])
    savedtd = td.to(SavedTensorDict)
    with pytest.raises(RuntimeError):
        savedtd.device
    savedtd.set("a", torch.randn(5, device=device))
    assert savedtd.device == device

    td = TensorDict({}, [5], device="cuda:0")
    savedtd = td.to(SavedTensorDict)
    savedtd.set("a", torch.randn(5))
    assert savedtd.get("a").device == device

    # ViewedTensorDict
    td = TensorDict({}, [6])
    viewedtd = td.view(2, 3)
    with pytest.raises(RuntimeError):
        viewedtd.device
    viewedtd.set("a", torch.randn(2, 3, device=device))
    assert viewedtd.device == device

    td = TensorDict({}, [6], device="cuda:0")
    viewedtd = td.view(2, 3)
    a = torch.randn(2, 3)
    viewedtd.set("a", a)
    assert viewedtd.get("a").device == device
    assert (a.unsqueeze(-1).to(device) == viewedtd.get("a")).all()


def _remote_process(worker_id, command_pipe_child, command_pipe_parent, tensordict):
    command_pipe_parent.close()
    while True:
        cmd, val = command_pipe_child.recv()
        if cmd == "recv":
            b = tensordict.get("b")
            assert (b == val).all()
            command_pipe_child.send("done")
        elif cmd == "send":
            a = torch.ones(2) * val
            tensordict.set_("a", a)
            assert (
                tensordict.get("a") == a
            ).all(), f'found {a} and {tensordict.get("a")}'
            command_pipe_child.send("done")
        elif cmd == "set_done":
            tensordict.set_("done", torch.ones(1, dtype=torch.bool))
            command_pipe_child.send("done")
        elif cmd == "set_undone_":
            tensordict.set_("done", torch.zeros(1, dtype=torch.bool))
            command_pipe_child.send("done")
        elif cmd == "update":
            tensordict.update_(
                TensorDict(
                    source={"a": tensordict.get("a").clone() + 1},
                    batch_size=tensordict.batch_size,
                )
            )
            command_pipe_child.send("done")
        elif cmd == "update_":
            tensordict.update_(
                TensorDict(
                    source={"a": tensordict.get("a").clone() - 1},
                    batch_size=tensordict.batch_size,
                )
            )
            command_pipe_child.send("done")

        elif cmd == "close":
            command_pipe_child.close()
            break


def _driver_func(tensordict, tensordict_unbind):
    procs = []
    children = []
    parents = []

    for i in range(2):
        command_pipe_parent, command_pipe_child = mp.Pipe()
        proc = mp.Process(
            target=_remote_process,
            args=(i, command_pipe_child, command_pipe_parent, tensordict_unbind[i]),
        )
        proc.start()
        command_pipe_child.close()
        parents.append(command_pipe_parent)
        children.append(command_pipe_child)
        procs.append(proc)

    b = torch.ones(2, 1) * 10
    tensordict.set_("b", b)
    for i in range(2):
        parents[i].send(("recv", 10))
        is_done = parents[i].recv()
        assert is_done == "done"

    for i in range(2):
        parents[i].send(("send", i))
        is_done = parents[i].recv()
        assert is_done == "done"
    a = tensordict.get("a").clone()
    assert (a[0] == 0).all()
    assert (a[1] == 1).all()

    assert not tensordict.get("done").any()
    for i in range(2):
        parents[i].send(("set_done", i))
        is_done = parents[i].recv()
        assert is_done == "done"
    assert tensordict.get("done").all()

    for i in range(2):
        parents[i].send(("set_undone_", i))
        is_done = parents[i].recv()
        assert is_done == "done"
    assert not tensordict.get("done").any()

    a_prev = tensordict.get("a").clone().contiguous()
    for i in range(2):
        parents[i].send(("update_", i))
        is_done = parents[i].recv()
        assert is_done == "done"
    new_a = tensordict.get("a").clone().contiguous()
    torch.testing.assert_allclose(a_prev - 1, new_a)

    a_prev = tensordict.get("a").clone().contiguous()
    for i in range(2):
        parents[i].send(("update", i))
        is_done = parents[i].recv()
        assert is_done == "done"
    new_a = tensordict.get("a").clone().contiguous()
    torch.testing.assert_allclose(a_prev + 1, new_a)

    for i in range(2):
        parents[i].send(("close", None))
        procs[i].join()


@pytest.mark.parametrize(
    "td_type", ["contiguous", "stack", "saved", "memmap", "memmap_stack"]
)
def test_mp(td_type):
    tensordict = TensorDict(
        source={
            "a": torch.randn(2, 2),
            "b": torch.randn(2, 1),
            "done": torch.zeros(2, 1, dtype=torch.bool),
        },
        batch_size=[2],
    )
    if td_type == "contiguous":
        tensordict = tensordict.share_memory_()
    elif td_type == "stack":
        tensordict = stack_td(
            [
                tensordict[0].clone().share_memory_(),
                tensordict[1].clone().share_memory_(),
            ],
            0,
        )
    elif td_type == "saved":
        tensordict = tensordict.clone().to(SavedTensorDict)
    elif td_type == "memmap":
        tensordict = tensordict.memmap_()
    elif td_type == "memmap_stack":
        tensordict = stack_td(
            [tensordict[0].clone().memmap_(), tensordict[1].clone().memmap_()], 0
        )
    else:
        raise NotImplementedError
    _driver_func(tensordict, tensordict.unbind(0))


def test_saved_delete():
    td = TensorDict(source={"a": torch.randn(3)}, batch_size=[])
    td = td.to(SavedTensorDict)
    file = td.file.name
    assert os.path.isfile(file)
    del td
    assert not os.path.isfile(file)


def test_stack_keys():
    td1 = TensorDict(source={"a": torch.randn(3)}, batch_size=[])
    td2 = TensorDict(
        source={
            "a": torch.randn(3),
            "b": torch.randn(3),
            "c": torch.randn(4),
            "d": torch.randn(5),
        },
        batch_size=[],
    )
    td = stack_td([td1, td2], 0)
    assert "a" in td.keys()
    assert "b" not in td.keys()
    assert "b" in td[1].keys()
    td.set("b", torch.randn(2, 10), inplace=False)  # overwrites
    with pytest.raises(KeyError):
        td.set_("c", torch.randn(2, 10))  # overwrites
    td.set_("b", torch.randn(2, 10))  # b has been set before

    td1.set("c", torch.randn(4))
    td[
        "c"
    ]  # we must first query that key for the stacked tensordict to update the list
    assert "c" in td.keys(), list(td.keys())  # now all tds have the key c
    td.get("c")

    td1.set("d", torch.randn(6))
    with pytest.raises(RuntimeError):
        td.get("d")

    td["e"] = torch.randn(2, 4)
    assert "e" in td.keys()  # now all tds have the key c
    td.get("e")


@pytest.mark.parametrize(
    "idx",
    [
        (slice(None),),
        slice(None),
        (3, 4),
        (3, slice(None), slice(2, 2, 2)),
        (torch.tensor([1, 2, 3]),),
        ([1, 2, 3]),
        (
            torch.tensor([1, 2, 3]),
            torch.tensor([2, 3, 4]),
            torch.tensor([0, 10, 2]),
            torch.tensor([2, 4, 1]),
        ),
        torch.zeros(10, 7, 11, 5, dtype=torch.bool).bernoulli_(),
        torch.zeros(10, 7, 11, dtype=torch.bool).bernoulli_(),
        (0, torch.zeros(7, dtype=torch.bool).bernoulli_()),
    ],
)
def test_getitem_batch_size(idx):
    shape = [
        10,
        7,
        11,
        5,
    ]
    mocking_tensor = torch.zeros(*shape)
    expected_shape = mocking_tensor[idx].shape
    resulting_shape = _getitem_batch_size(shape, idx)
    assert expected_shape == resulting_shape, (idx, expected_shape, resulting_shape)


@pytest.mark.parametrize("device", get_available_devices())
def test_requires_grad(device):
    torch.manual_seed(1)
    # Just one of the tensors have requires_grad
    tensordicts = [
        TensorDict(
            batch_size=[11, 12],
            source={
                "key1": torch.randn(
                    11, 12, 5, device=device, requires_grad=True if i == 5 else False
                ),
                "key2": torch.zeros(
                    11, 12, 50, device=device, dtype=torch.bool
                ).bernoulli_(),
            },
        )
        for i in range(10)
    ]
    stacked_td = LazyStackedTensorDict(*tensordicts, stack_dim=0)
    # First stacked tensor has requires_grad == True
    assert list(stacked_td.values_meta())[0].requires_grad is True
    td0 = SavedTensorDict(tensordicts[0])
    with pytest.raises(
        Exception,
        match=re.escape(
            "SavedTensorDicts is not compatible with gradients, one of Tensors has requires_grad equals True"
        ),
    ):
        td5 = SavedTensorDict(tensordicts[5])


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize(
    "td_type", ["tensordict", "view", "unsqueeze", "squeeze", "saved", "stack"]
)
@pytest.mark.parametrize("update", [True, False])
def test_filling_empty_tensordict(device, td_type, update):
    if td_type == "tensordict":
        td = TensorDict({}, batch_size=[16], device=device)
    elif td_type == "view":
        td = TensorDict({}, batch_size=[4, 4], device=device).view(-1)
    elif td_type == "unsqueeze":
        td = TensorDict({}, batch_size=[16], device=device).unsqueeze(-1)
    elif td_type == "squeeze":
        td = TensorDict({}, batch_size=[16, 1], device=device).squeeze(-1)
    elif td_type == "saved":
        td = TensorDict({}, batch_size=[16], device=device).to(SavedTensorDict)
    elif td_type == "stack":
        td = torch.stack([TensorDict({}, [], device=device) for _ in range(16)], 0)
    else:
        raise NotImplementedError

    for i in range(16):
        other_td = TensorDict({"a": torch.randn(10), "b": torch.ones(1)}, [])
        if td_type == "unsqueeze":
            other_td = other_td.unsqueeze(-1).to_tensordict()
        if update:
            subtd = td.get_sub_tensordict(i)
            subtd.update(other_td, inplace=True)
        else:
            td[i] = other_td

    assert td.device == device
    assert td.get("a").device == device
    assert (td.get("b") == 1).all()
    if td_type == "view":
        assert td._source["a"].shape == torch.Size([4, 4, 10])
    elif td_type == "unsqueeze":
        assert td._source["a"].shape == torch.Size([16, 10])
    elif td_type == "squeeze":
        assert td._source["a"].shape == torch.Size([16, 1, 10])
    elif td_type == "stack":
        assert (td[-1] == other_td.to(device)).all()


def test_getitem_nested():
    tensor = torch.randn(4, 5, 6, 7)
    tensordict = TensorDict({}, [4])
    sub_tensordict = TensorDict({}, [4, 5])
    sub_sub_tensordict = TensorDict({"c": tensor}, [4, 5, 6])
    with pytest.raises(RuntimeError, match="The nested tensordict had not device"):
        tensordict["a"] = sub_tensordict
    sub_tensordict["b"] = sub_sub_tensordict
    tensordict["a"] = sub_tensordict

    # check that content match
    assert tensordict["a"] is sub_tensordict
    assert tensordict["a", "b"] is sub_sub_tensordict
    assert tensordict["a", "b", "c"] is tensor

    # check that shapes are kept
    assert tensordict.shape == torch.Size([4])
    assert sub_tensordict.shape == torch.Size([4, 5])
    assert sub_sub_tensordict.shape == torch.Size([4, 5, 6])

    # check that device are tracked
    assert tensordict.device == torch.device("cpu")
    assert sub_tensordict.device == torch.device("cpu")
    assert sub_sub_tensordict.device == torch.device("cpu")


def test_setitem_nested():
    tensor = torch.randn(4, 5, 6, 7)
    tensor2 = torch.ones(4, 5, 6, 7)
    tensordict = TensorDict({}, [4])
    sub_tensordict = TensorDict({}, [4, 5])
    sub_sub_tensordict = TensorDict({"c": tensor}, [4, 5, 6])
    sub_sub_tensordict2 = TensorDict({"c": tensor2}, [4, 5, 6])
    sub_tensordict["b"] = sub_sub_tensordict
    tensordict["a"] = sub_tensordict
    assert tensordict["a", "b"] is sub_sub_tensordict

    tensordict["a", "b"] = sub_sub_tensordict2
    assert tensordict["a", "b"] is sub_sub_tensordict2
    assert (tensordict["a", "b", "c"] == 1).all()


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
