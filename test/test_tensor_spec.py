# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import numpy as np
import pytest
import torch
from torchrl.data.tensor_specs import (
    NdUnboundedContinuousTensorSpec,
    NdBoundedTensorSpec,
    CompositeSpec,
    MultOneHotDiscreteTensorSpec,
    BinaryDiscreteTensorSpec,
    BoundedTensorSpec,
    UnboundedContinuousTensorSpec,
    OneHotDiscreteTensorSpec,
    CustomNdOneHotDiscreteTensorSpec,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.float64, None])
def test_bounded(dtype):
    torch.manual_seed(0)
    np.random.seed(0)
    for _ in range(100):
        bounds = torch.randn(2).sort()[0]
        ts = BoundedTensorSpec(bounds[0].item(), bounds[1].item(), dtype=dtype)
        _dtype = dtype
        if dtype is None:
            _dtype = torch.get_default_dtype()

        r = ts.rand()
        assert ts.is_in(r)
        assert r.dtype is _dtype
        ts.is_in(ts.encode(bounds.mean()))
        ts.is_in(ts.encode(bounds.mean().item()))
        assert (ts.encode(ts.to_numpy(r)) == r).all()


def test_onehot():
    torch.manual_seed(0)
    np.random.seed(0)

    ts = OneHotDiscreteTensorSpec(10)
    for _ in range(100):
        r = ts.rand()
        ts.to_numpy(r)
        ts.encode(torch.tensor([5]))
        ts.encode(torch.tensor([5]).numpy())
        ts.encode(9)
        with pytest.raises(RuntimeError):
            ts.encode(torch.tensor([11]))  # out of bounds
        assert ts.is_in(r)
        assert (ts.encode(ts.to_numpy(r)) == r).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.float64, None])
def test_unbounded(dtype):
    torch.manual_seed(0)
    np.random.seed(0)
    ts = UnboundedContinuousTensorSpec(dtype=dtype)

    if dtype is None:
        dtype = torch.get_default_dtype()
    for _ in range(100):
        r = ts.rand()
        ts.to_numpy(r)
        assert ts.is_in(r)
        assert r.dtype is dtype
        assert (ts.encode(ts.to_numpy(r)) == r).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.float64, None])
@pytest.mark.parametrize(
    "shape",
    [
        [],
        torch.Size(
            [
                3,
            ]
        ),
    ],
)
def test_ndbounded(dtype, shape):
    torch.manual_seed(0)
    np.random.seed(0)

    for _ in range(100):
        lb = torch.rand(10) - 1
        ub = torch.rand(10) + 1
        ts = NdBoundedTensorSpec(lb, ub, dtype=dtype)
        _dtype = dtype
        if dtype is None:
            _dtype = torch.get_default_dtype()

        r = ts.rand(shape)
        assert r.dtype is _dtype
        assert r.shape == torch.Size([*shape, 10])
        assert (r >= lb.to(dtype)).all() and (
            r <= ub.to(dtype)
        ).all(), f"{r[r <= lb] - lb.expand_as(r)[r <= lb]} -- {r[r >= ub] - ub.expand_as(r)[r >= ub]} "
        ts.to_numpy(r)
        assert ts.is_in(r)
        ts.encode(lb + torch.rand(10) * (ub - lb))
        ts.encode((lb + torch.rand(10) * (ub - lb)).numpy())
        assert (ts.encode(ts.to_numpy(r)) == r).all()
        with pytest.raises(AssertionError):
            ts.encode(torch.rand(10) + 3)  # out of bounds
        with pytest.raises(AssertionError):
            ts.to_numpy(torch.rand(10) + 3)  # out of bounds


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.float64, None])
@pytest.mark.parametrize("n", range(3, 10))
@pytest.mark.parametrize(
    "shape",
    [
        [],
        torch.Size(
            [
                3,
            ]
        ),
    ],
)
def test_ndunbounded(dtype, n, shape):
    torch.manual_seed(0)
    np.random.seed(0)

    ts = NdUnboundedContinuousTensorSpec(
        shape=[
            n,
        ],
        dtype=dtype,
    )

    if dtype is None:
        dtype = torch.get_default_dtype()

    for _ in range(100):
        r = ts.rand(shape)
        assert r.shape == torch.Size(
            [
                *shape,
                n,
            ]
        )
        ts.to_numpy(r)
        assert ts.is_in(r)
        assert r.dtype is dtype
        assert (ts.encode(ts.to_numpy(r)) == r).all()


@pytest.mark.parametrize("n", range(3, 10))
@pytest.mark.parametrize(
    "shape",
    [
        [],
        torch.Size(
            [
                3,
            ]
        ),
    ],
)
def test_binary(n, shape):
    torch.manual_seed(0)
    np.random.seed(0)

    ts = BinaryDiscreteTensorSpec(n)
    for _ in range(100):
        r = ts.rand(shape)
        assert r.shape == torch.Size(
            [
                *shape,
                n,
            ]
        )
        assert ts.is_in(r)
        assert ((r == 0) | (r == 1)).all()
        assert (ts.encode(r.numpy()) == r).all()
        assert (ts.encode(ts.to_numpy(r)) == r).all()


@pytest.mark.parametrize(
    "ns",
    [
        [
            5,
        ],
        [5, 2, 3],
        [4, 4, 1],
    ],
)
@pytest.mark.parametrize(
    "shape",
    [
        [],
        torch.Size(
            [
                3,
            ]
        ),
    ],
)
def test_mult_onehot(shape, ns):
    torch.manual_seed(0)
    np.random.seed(0)
    ts = MultOneHotDiscreteTensorSpec(nvec=ns)
    for _ in range(100):
        r = ts.rand(shape)
        assert r.shape == torch.Size(
            [
                *shape,
                sum(ns),
            ]
        )
        assert ts.is_in(r)
        assert ((r == 0) | (r == 1)).all()
        rsplit = r.split(ns, dim=-1)
        for _r, _n in zip(rsplit, ns):
            assert (_r.sum(-1) == 1).all()
            assert _r.shape[-1] == _n
        np_r = ts.to_numpy(r)
        assert (ts.encode(np_r) == r).all()


@pytest.mark.parametrize("n", range(10, 12))
@pytest.mark.parametrize("shape", [torch.Size([]), torch.Size([10])])
def test_custom_ndonehot(n, shape):
    torch.manual_seed(0)
    np.random.seed(0)

    with pytest.raises(RuntimeError):
        mask = torch.zeros(*shape, n).bernoulli_()
        ts = CustomNdOneHotDiscreteTensorSpec(mask)
    mask = torch.zeros(*shape, n, dtype=torch.bool).bernoulli_()
    ts = CustomNdOneHotDiscreteTensorSpec(mask)

    for _ in range(100):
        r = ts.rand([10])
        assert r.shape == torch.Size([10, *shape, n])
        assert ts.is_in(r), r
        assert ((r == 0) | (r == 1)).all()
        r_numpy = r.argmax(-1).numpy()
        assert (ts.encode(r_numpy) == r).all()
        assert (ts.encode(ts.to_numpy(r)) == r).all()
        assert (r.sum(-1) == 1).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.float64, None])
@pytest.mark.parametrize(
    "shape",
    [
        [],
        torch.Size(
            [
                3,
            ]
        ),
    ],
)
def test_composite(shape, dtype):
    torch.manual_seed(0)
    np.random.seed(0)

    ts = CompositeSpec(
        obs=NdBoundedTensorSpec(
            torch.zeros(3, 32, 32), torch.ones(3, 32, 32), dtype=dtype
        ),
        act=NdUnboundedContinuousTensorSpec((7,), dtype=dtype),
    )
    if dtype is None:
        dtype = torch.get_default_dtype()

    rand_td = ts.rand(shape)
    assert rand_td.shape == torch.Size(shape)
    assert rand_td.get("obs").shape == torch.Size([*shape, 3, 32, 32])
    assert rand_td.get("obs").dtype == dtype
    assert rand_td.get("act").shape == torch.Size([*shape, 7])
    assert rand_td.get("act").dtype == dtype


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
