import contextlib
from typing import Callable, Optional, Tuple

import torch
from torch.utils._pytree import tree_map


@contextlib.contextmanager
def no_dispatch():
    guard = torch._C._DisableTorchDispatch()
    try:
        yield
    finally:
        del guard


class FiniteTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, elem: torch.Tensor, *args, **kwargs):
        if not torch.isfinite(elem).all():
            raise RuntimeError("FiniteTensor encountered a non-finite tensor.")
        return torch.Tensor._make_subclass(cls, elem, elem.requires_grad)

    def __repr__(self) -> str:
        return f"FiniteTensor({super().__repr__()})"

    @classmethod
    def __torch_dispatch__(
        cls,
        func: Callable,
        types,
        args: Tuple = (),
        kwargs: Optional[dict] = None,
    ):
        # TODO: also explicitly recheck invariants on inplace/out mutation
        if kwargs:
            raise Exception("Expected empty kwargs")
        with no_dispatch():
            rs = func(*args)
        return tree_map(
            lambda e: FiniteTensor(e) if isinstance(e, torch.Tensor) else e, rs
        )
