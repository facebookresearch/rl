# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import functools
import os
import tempfile
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import torch

from torchrl import prod
from torchrl.data.tensordict.utils import _getitem_batch_size
from torchrl.data.utils import (
    DEVICE_TYPING,
    INDEX_TYPING,
    torch_to_numpy_dtype_dict,
)

MEMMAP_HANDLED_FN = {}

__all__ = ["MemmapTensor", "set_transfer_ownership"]


def implements_for_memmap(torch_function) -> Callable:
    """Register a torch function override for ScalarTensor"""

    @functools.wraps(torch_function)
    def decorator(func):
        MEMMAP_HANDLED_FN[torch_function] = func
        return func

    return decorator


def to_numpy(tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    else:
        return tensor


class MemmapTensor(object):
    """A torch.tensor interface with a np.memmap array.

    A temporary file is created and cleared once the object is out-of-scope.
    This class is aimed at being used for data transfer in between processes
    and remote workers that have access to
    a common storage, and as such it supports serialization and
    deserialization. It is possible to choose if the ownership is
    transferred upon serialization / deserialization: If owenership is not
    transferred (transfer_ownership=False, default), then the process where
    the MemmapTensor was created will be responsible of clearing it once it
    gets out of scope (in that process). Otherwise, the process that
    deserialize the MemmapTensor will be responsible of clearing the files
    once the object is out of scope.

    Supports (almost) all tensor operations.

    Args:
        *tensor_or_size (torch.Tensor, MemmapTensor, torch.Size or sequence of integers):
            If a size is provided (with a sequence of integers, a torch.Size object
            or a list/tuple of integers) it indicates the size of the MemmapTensor created.
            If a te is provided, its content will be stored on physical storage.
            If MemmapTensor, a new MemmapTensor is created and the same data is stored in it.
        device (torch.device or equivalent, optional): device where the loaded
            tensor will be sent. This should not be used with MemmapTensors
            created from torch.Tensor objects. Default is "cpu".
        dtype (torch.dtype, optional): dtype of the loaded tensor.
            This should not be used with MemmapTensors created from torch.Tensor
            objects. Default is `torch.get_default_dtype()`.
        transfer_ownership (bool, optional): affects the ownership after serialization:
            if True, the current process looses ownership immediately after
            serialization. If False, the current process keeps the ownership
            of the temporary file.
            Default: False.
        prefix (str or path, optional): prefix of the file location.

    Examples:
        >>> x = torch.ones(3,4)
        >>> x_memmap = MemmapTensor(x)
        >>> # indexing
        >>> x0 = x_memmap[0]
        >>> x0[:] = 2
        >>> assert (x_memmap[0]==2).all()
        >>>
        >>> # device
        >>> x = x.to('cuda:0')
        >>> x_memmap = MemmapTensor(x)
        >>> assert (x_memmap.clone()).device == torch.device('cuda:0')
        >>>
        >>> # operations
        >>> assert (x_memmap + 1 == x+1).all()
        >>> assert (x_memmap / 2 == x/2).all()
        >>> assert (x_memmap * 2 == x*2).all()
        >>>
        >>> # temp file clearance
        >>> filename = x_memmap.filename
        >>> assert os.path.isfile(filename)
        >>> del x_memmap
        >>> assert not os.path.isfile(filename)

    """

    requires_grad = False

    def __init__(
        self,
        elem: Union[torch.Tensor, MemmapTensor],
        *size: int,
        device: DEVICE_TYPING = None,
        dtype: torch.dtype = None,
        transfer_ownership: bool = False,
        prefix: Optional[str] = None,
    ):
        self.idx = None
        self._memmap_array = None
        self.prefix = prefix
        self.is_meta = False
        self.file = tempfile.NamedTemporaryFile(prefix=prefix, delete=False)
        self.filename = self.file.name
        self.file.close()  # we close the file for now, but don't delete it

        if isinstance(elem, (torch.Tensor, MemmapTensor, np.ndarray)):
            if device is not None:
                raise TypeError(
                    "device cannot be passed when creating a MemmapTensor from a tensor"
                )
            if dtype is not None:
                raise TypeError(
                    "dtype cannot be passed when creating a MemmapTensor from a tensor"
                )
            self._init_tensor(elem, transfer_ownership)
        else:
            if not isinstance(elem, int) and size:
                raise TypeError(
                    "Valid init methods for MemmapTensor are: "
                    "\n- MemmapTensor(tensor, ...)"
                    "\n- MemmapTensor(size, ...)"
                    "\n- MemmapTensor(*size, ...)"
                )
            shape = (
                torch.Size([elem] + list(size))
                if isinstance(elem, int)
                else torch.Size(elem)
            )
            device = device if device is not None else torch.device("cpu")
            dtype = dtype if dtype is not None else torch.get_default_dtype()
            self._init_shape(shape, device, dtype, transfer_ownership)

    def _init_shape(
        self,
        shape: torch.Size,
        device: DEVICE_TYPING,
        dtype: torch.dtype,
        transfer_ownership: bool,
    ):
        self._device = device
        self._shape = shape
        self.transfer_ownership = transfer_ownership
        self.np_shape = tuple(self._shape)
        self._dtype = dtype
        self._ndim = len(shape)
        self._numel = prod(shape)
        self.mode = "r+"
        self._has_ownership = True

        self._tensor_dir = torch.zeros(1, device=device, dtype=dtype).__dir__()
        self._save_item(shape)

    def _init_tensor(
        self, elem: Union[torch.Tensor, MemmapTensor], transfer_ownership: bool
    ):
        if not isinstance(elem, (torch.Tensor, MemmapTensor)):
            raise TypeError(
                "convert input to torch.Tensor before calling MemmapTensor() " "on it."
            )

        if elem.requires_grad:
            raise RuntimeError(
                "MemmapTensor is incompatible with tensor.requires_grad. "
                "Consider calling tensor.detach() first."
            )

        self._device = elem.device
        self._shape = elem.shape
        self.transfer_ownership = transfer_ownership
        self.np_shape = tuple(self._shape)
        self._dtype = elem.dtype
        self._tensor_dir = elem.__dir__()
        self._ndim = elem.ndimension()
        self._numel = elem.numel()
        self.mode = "r+"
        self._has_ownership = True
        self._had_ownership = True
        if isinstance(elem, MemmapTensor):
            prev_filename = elem.filename
            self._copy_item(prev_filename)
            if self.memmap_array is elem.memmap_array:
                raise RuntimeError
        else:
            if elem.requires_grad:
                raise Exception(
                    "memmap is not compatible with gradients, Tensor has requires_grad equals True"
                )
            self._save_item(elem)

    def _get_memmap_array(self) -> np.memmap:
        if self._memmap_array is None:
            self._memmap_array = np.memmap(
                self.filename,
                dtype=torch_to_numpy_dtype_dict[self.dtype],
                mode=self.mode,
                shape=self.np_shape,
            )
        return self._memmap_array

    def _set_memmap_array(self, value: np.memmap) -> None:
        self._memmap_array = value

    memmap_array = property(_get_memmap_array, _set_memmap_array)

    def _save_item(
        self,
        value: Union[torch.Tensor, torch.Size, MemmapTensor, np.ndarray],
        idx: Optional[int] = None,
    ):
        if isinstance(value, (torch.Tensor,)):
            np_array = value.cpu().numpy()
        elif isinstance(value, torch.Size):
            # create the memmap array on disk
            _ = self.memmap_array
            return
        else:
            np_array = value
        memmap_array = self.memmap_array
        if idx is None:
            memmap_array[:] = np_array
        else:
            memmap_array[idx] = np_array

    def _copy_item(self, filename: Union[bytes, str]) -> None:
        self.memmap_array[:] = np.memmap(
            filename,
            dtype=torch_to_numpy_dtype_dict[self.dtype],
            mode="r",
            shape=self.np_shape,
        )

    def _load_item(
        self,
        idx: Optional[int] = None,
        memmap_array: Optional[np.ndarray] = None,
        from_numpy: bool = False,
    ) -> torch.Tensor:
        if memmap_array is None:
            memmap_array = self.memmap_array
        if idx is not None:
            if isinstance(idx, torch.Tensor):
                idx = idx.cpu()
            elif isinstance(idx, tuple) and any(
                isinstance(_idx, torch.Tensor) for _idx in idx
            ):
                idx = tuple(
                    _idx.cpu() if isinstance(_idx, torch.Tensor) else _idx
                    for _idx in idx
                )
            memmap_array = memmap_array[idx]
        out = self._np_to_tensor(memmap_array, from_numpy=from_numpy)
        if (
            idx is not None
            and not isinstance(idx, (int, np.integer))
            and len(idx) == 1
            and not (isinstance(idx, torch.Tensor) and idx.dtype is torch.bool)
        ):  # and isinstance(idx, torch.Tensor) and len(idx) == 1:
            size = _getitem_batch_size(self.shape, idx)
            out = out.view(size)
        return out

    def _np_to_tensor(self, memmap_array: np.ndarray, from_numpy: bool) -> torch.Tensor:
        if from_numpy:
            return torch.from_numpy(memmap_array)
        return torch.as_tensor(memmap_array, device=self.device)

    @classmethod
    def __torch_function__(
        cls,
        func: Callable,
        types,
        args: Tuple = (),
        kwargs: Optional[dict] = None,
    ):
        if kwargs is None:
            kwargs = {}
        if func not in MEMMAP_HANDLED_FN:
            args = tuple(a._tensor if hasattr(a, "_tensor") else a for a in args)
            ret = func(*args, **kwargs)
            return ret

        return MEMMAP_HANDLED_FN[func](*args, **kwargs)

    @property
    def _tensor(self) -> torch.Tensor:
        return self._load_item()

    @property
    def _tensor_from_numpy(self) -> torch.Tensor:
        # a tensor created with `from_numpy` to make sure that changes are done in-place
        return self._load_item(from_numpy=True)

    def ndimension(self) -> int:
        return self._ndim

    def numel(self) -> int:
        return self._numel

    def clone(self) -> MemmapTensor:
        """Clones the MemmapTensor onto another MemmapTensor

        Returns:
            a new MemmapTensor with the same data but a new storage.

        """
        return MemmapTensor(self)

    def contiguous(self) -> torch.Tensor:
        """Copies the MemmapTensor onto a torch.Tensor object.

        Returns:
            a torch.Tensor instance with the data of the MemmapTensor
        stored on the desired device.

        """
        return self._tensor

    @property
    def device(self) -> torch.device:
        return self._device

    @device.setter
    def device(self, device):
        self._device = torch.device(device)

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def shape(self) -> torch.Size:
        return self._shape

    def cpu(self) -> torch.Tensor:
        """Defines the device of the MemmapTensor as "cpu"

        Returns: a MemmapTensor where device has been modified in-place

        """
        self.device = torch.device("cpu")
        return self

    def cuda(self) -> torch.Tensor:
        """Defines the device of the MemmapTensor as "cuda"

        Returns: a MemmapTensor where device has been modified in-place

        """
        self.device = torch.device("cuda")
        return self

    def numpy(self) -> np.ndarray:
        return self._tensor.numpy()

    def copy_(self, other: Union[torch.Tensor, MemmapTensor]) -> MemmapTensor:
        self._save_item(other)
        return self

    def set_transfer_ownership(self, value: bool = True) -> MemmapTensor:
        """Controls whether the ownership will be transferred to another
        process upon serialization/deserialization

        Args:
            value (bool): if True, the ownership will be transferred.
                Otherwise the process will keep ownership of the
                MemmapTensor temp file.
                Default = True

        Returns:
            the MemmapTensor

        """
        if not isinstance(value, bool):
            raise TypeError(
                f"value provided to set_transfer_ownership should be a "
                f"boolean, got {type(value)}"
            )
        self.transfer_ownership = value
        return self

    def __del__(self) -> None:
        if "_has_ownership" in self.__dir__() and self._has_ownership:
            os.unlink(self.filename)

    def __eq__(self, other: Any) -> torch.Tensor:
        if not isinstance(other, (MemmapTensor, torch.Tensor, float, int, np.ndarray)):
            raise NotImplementedError(f"Unknown type {type(other)}")
        return self._tensor == other

    def __getattr__(self, attr: str) -> Any:
        if attr in self.__dir__():
            return self.__getattribute__(
                attr
            )  # make sure that appropriate exceptions are raised

        if ("_tensor_dir" not in self.__dir__()) or (
            attr not in self.__getattribute__("_tensor_dir")
        ):
            raise AttributeError(f"{attr} not found")
        _tensor = self.__getattribute__("_tensor")
        return getattr(_tensor, attr)

    def masked_fill_(self, mask: torch.Tensor, value: float):
        self.memmap_array[mask.cpu().numpy()] = value
        return self

    def __len__(self):
        return self.shape[0] if len(self.shape) else 0

    def is_shared(self) -> bool:
        return False

    def __add__(self, other: Union[float, MemmapTensor, torch.Tensor]) -> torch.Tensor:
        return torch.add(self, other)

    def __truediv__(
        self, other: Union[float, MemmapTensor, torch.Tensor]
    ) -> torch.Tensor:
        return torch.div(self, other)

    def __neg__(self: Union[float, MemmapTensor, torch.Tensor]) -> torch.Tensor:
        return torch.neg(self)

    def __sub__(self, other: Union[float, MemmapTensor, torch.Tensor]) -> torch.Tensor:
        return torch.sub(self, other)

    def __matmul__(
        self, other: Union[float, MemmapTensor, torch.Tensor]
    ) -> torch.Tensor:
        return torch.matmul(self, other)

    def __mul__(self, other: Union[float, MemmapTensor, torch.Tensor]) -> torch.Tensor:
        return torch.mul(self, other)

    def __pow__(self, other: Union[float, MemmapTensor, torch.Tensor]) -> torch.Tensor:
        return torch.pow(self, other)

    def __repr__(self) -> str:
        return (
            f"MemmapTensor(shape={self.shape}, device={self.device}, "
            f"dtype={self.dtype})"
        )

    def __getitem__(self, item: INDEX_TYPING) -> torch.Tensor:
        # return self._load_item(memmap_array=self.memmap_array[item])#[item]
        # return self._load_item()[item]
        return self._load_item(idx=item)

    def __setitem__(self, idx: INDEX_TYPING, value: torch.Tensor):
        if self.device == torch.device("cpu"):
            self._load_item()[idx] = value
        else:
            if isinstance(idx, torch.Tensor):
                idx = idx.cpu()
            elif isinstance(idx, tuple) and any(
                isinstance(_idx, torch.Tensor) for _idx in idx
            ):
                idx = tuple(
                    _idx.cpu() if isinstance(_idx, torch.Tensor) else _idx
                    for _idx in idx
                )
            self.memmap_array[idx] = to_numpy(value)

    def __setstate__(self, state: dict) -> None:
        if state["file"] is None:
            # state["_had_ownership"] = state["_had_ownership"]
            # state["_has_ownership"] = delete
            tmpfile = tempfile.NamedTemporaryFile(delete=False)
            tmpfile.close()
            tmpfile.name = state["filename"]
            tmpfile._closer.name = state["filename"]
            state["file"] = tmpfile
        self.__dict__.update(state)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["file"] = None
        state["_memmap_array"] = None
        state["_fake"] = None
        state["_has_ownership"] = (
            state["transfer_ownership"] and state["_had_ownership"]
        )
        self._had_ownership = self._has_ownership
        # self._had_ownership = self._has_ownership = state["_had_ownership"]
        return state

    def __reduce__(self, *args, **kwargs):
        if self.transfer_ownership and self._has_ownership:
            self._has_ownership = False
            # those values should already be False
            # self.file.delete = False
            # self.file._closer.delete = False
        return super(MemmapTensor, self).__reduce__(*args, **kwargs)

    def to(
        self, dest: Union[DEVICE_TYPING, torch.dtype]
    ) -> Union[torch.Tensor, MemmapTensor]:
        """Maps a MemmapTensor to a given dtype or device.

        Args:
            dest (device indicator or torch.dtype): where to cast the
                MemmapTensor. For devices, this is a lazy operation
                (as the data is stored on physical memory). For dtypes, the
                tensor will be retrieved, mapped to the
                desired dtype and cast to a new MemmapTensor.

        Returns:

        """
        if isinstance(dest, (int, str, torch.device)):
            dest = torch.device(dest)
            self.device = dest
            return self
        elif isinstance(dest, torch.dtype):
            return MemmapTensor(self._tensor.to(dest))
        else:
            raise NotImplementedError(
                f"argument dest={dest} to MemmapTensor.to(dest) is not "
                f"handled. "
                f"Please provide a dtype or a device."
            )

    def unbind(self, dim: int) -> Tuple[torch.Tensor, ...]:
        """Unbinds a MemmapTensor along the desired dimension.

        Args:
            dim (int): dimension along which the MemmapTensor will be split.

        Returns:
            A tuple of indexed MemmapTensors that share the same storage.

        """
        idx = [
            (tuple(slice(None) for _ in range(dim)) + (i,))
            for i in range(self.shape[dim])
        ]
        return tuple(self[_idx] for _idx in idx)


@implements_for_memmap(torch.stack)
def stack(
    list_of_memmap: List[MemmapTensor],
    dim: int,
    out: Optional[Union[torch.Tensor]] = None,
) -> torch.Tensor:
    list_of_tensors = [
        a._tensor if isinstance(a, MemmapTensor) else a for a in list_of_memmap
    ]
    if isinstance(out, MemmapTensor):
        list_of_tensors = [tensor.cpu() for tensor in list_of_tensors]
        return torch.stack(list_of_tensors, dim, out=out._tensor_from_numpy)
    else:
        return torch.stack(list_of_tensors, dim, out=out)


@implements_for_memmap(torch.unbind)
def unbind(memmap: MemmapTensor, dim: int) -> Tuple[torch.Tensor, ...]:
    return memmap.unbind(dim)


@implements_for_memmap(torch.cat)
def cat(
    list_of_memmap: List[MemmapTensor],
    dim: int,
    out: Optional[Union[torch.Tensor, MemmapTensor]] = None,
) -> torch.Tensor:
    list_of_tensors = [
        a._tensor if isinstance(a, MemmapTensor) else a for a in list_of_memmap
    ]
    print("mm: ", [t.shape for t in list_of_memmap])
    print("tensors: ", [t.shape for t in list_of_tensors])
    print("dim: ", dim, "shape: ", torch.cat(list_of_tensors, dim, out=out).shape)
    return torch.cat(list_of_tensors, dim, out=out)


def set_transfer_ownership(memmap: MemmapTensor, value: bool = True) -> None:
    if isinstance(memmap, MemmapTensor):
        memmap.set_transfer_ownership(value)
