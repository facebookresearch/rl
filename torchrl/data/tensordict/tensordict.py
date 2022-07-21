# pad_size, value Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import abc
import functools
import tempfile
import textwrap
import uuid
from collections import defaultdict
from collections.abc import Mapping
from copy import copy, deepcopy
from numbers import Number
from textwrap import indent
from typing import (
    Callable,
    Dict,
    Generator,
    Iterator,
    KeysView,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)
from warnings import warn

import numpy as np
import torch

from torchrl import KeyDependentDefaultDict, prod
from torchrl.data.tensordict.memmap import MemmapTensor
from torchrl.data.tensordict.metatensor import MetaTensor
from torchrl.data.tensordict.utils import (
    _getitem_batch_size,
    _sub_index,
    convert_ellipsis_to_idx,
)
from torchrl.data.utils import (
    DEVICE_TYPING,
    expand_right,
    expand_as_right,
    INDEX_TYPING,
)

__all__ = [
    "TensorDict",
    "SubTensorDict",
    "merge_tensordicts",
    "LazyStackedTensorDict",
    "SavedTensorDict",
]

TD_HANDLED_FUNCTIONS: Dict = dict()
COMPATIBLE_TYPES = Union[
    torch.Tensor,
    MemmapTensor,
]  # None? # leaves space for _TensorDict

_STR_MIXED_INDEX_ERROR = "Received a mixed string-non string index. Only string-only or string-free indices are supported."


class _TensorDict(Mapping, metaclass=abc.ABCMeta):
    """
    _TensorDict is an abstract parent class for TensorDicts, the torchrl
    data container.
    """

    _safe = False
    _lazy = False
    is_meta = False

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        del state["_dict_meta"]
        return state

    def __setstate__(self, state: dict) -> None:
        state["_dict_meta"] = KeyDependentDefaultDict(self._make_meta)
        self.__dict__.update(state)

    def __init__(self):
        self._dict_meta = KeyDependentDefaultDict(self._make_meta)

    @abc.abstractmethod
    def _make_meta(self, key: str) -> MetaTensor:
        raise NotImplementedError

    @property
    def shape(self) -> torch.Size:
        """See _TensorDict.batch_size"""
        return self.batch_size

    @property
    @abc.abstractmethod
    def batch_size(self) -> torch.Size:
        """Shape of (or batch_size) of a TensorDict.
        The shape of a tensordict corresponds to the common N first
        dimensions of the tensors it contains, where N is an arbitrary
        number. The TensorDict shape is controlled by the user upon
        initialization (i.e. it is not inferred from the tensor shapes) and
        it should not be changed dynamically.

        Returns:
            a torch.Size object describing the TensorDict batch size.

        """
        raise NotImplementedError

    @property
    def requires_grad(self):
        return any(v.requires_grad for v in self._dict_meta.values())

    def _batch_size_setter(self, new_batch_size: torch.Size) -> None:
        if new_batch_size == self.batch_size:
            return
        if self._lazy:
            raise RuntimeError(
                "modifying the batch size of a lazy repesentation of a "
                "tensordict is not permitted. Consider instantiating the "
                "tensordict fist by calling `td = td.to_tensordict()` before "
                "resetting the batch size."
            )
        if self.batch_size == new_batch_size:
            return
        if not isinstance(new_batch_size, torch.Size):
            new_batch_size = torch.Size(new_batch_size)
        self._check_new_batch_size(new_batch_size)
        self._change_batch_size(new_batch_size)

    @property
    def batch_dims(self) -> int:
        """Length of the tensordict batch size.

        Returns:
            int describing the number of dimensions of the tensordict.

        """
        return len(self.batch_size)

    def ndimension(self) -> int:
        return self.batch_dims

    def dim(self) -> int:
        return self.batch_dims

    @property
    @abc.abstractmethod
    def device(self) -> torch.device:
        """Device of a TensorDict. All tensors of a tensordict must live on the
        same device.

        Returns:
            torch.device object indicating the device where the tensors
            are placed.

        """
        raise NotImplementedError

    @device.setter
    @abc.abstractmethod
    def device(self, value: DEVICE_TYPING) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def _device_safe(self) -> Union[None, torch.device]:
        raise NotImplementedError

    def is_shared(self, no_check: bool = True) -> bool:
        """Checks if tensordict is in shared memory.

        This is always True for CUDA tensordicts, except when stored as
        MemmapTensors.

        Args:
            no_check (bool, optional): whether to use cached value or not
                Default is True

        """
        if no_check:
            if self._is_shared is None:
                if self.keys():
                    _is_shared = all(value.is_shared() for value in self.values_meta())
                else:
                    _is_shared = None
                self._is_shared = _is_shared
            return self._is_shared
        return all(item.is_shared() for item in self.values_meta())

    def is_memmap(self, no_check: bool = True) -> bool:
        """Checks if tensordict is stored with MemmapTensors.

        Args:
            no_check (bool, optional): whether to use cached value or not
                Default is True

        """
        if no_check:
            if self._is_memmap is None:
                if self.keys():
                    _is_memmap = all(value.is_memmap() for value in self.values_meta())
                else:
                    _is_memmap = None
                self._is_memmap = _is_memmap
            return self._is_memmap
        return all(item.is_memmap() for item in self.values_meta())

    def numel(self) -> int:
        """Total number of elements in the batch."""
        return max(1, prod(self.batch_size))

    def _check_batch_size(self) -> None:
        bs = [value.shape[: self.batch_dims] for key, value in self.items_meta()] + [
            self.batch_size
        ]
        if len(set(bs)) > 1:
            raise RuntimeError(
                f"batch_size are incongruent, got {list(set(bs))}, "
                f"-- expected {self.batch_size}"
            )

    def _check_is_shared(self) -> bool:
        raise NotImplementedError(f"{self.__class__.__name__}")

    def _check_device(self) -> None:
        raise NotImplementedError(f"{self.__class__.__name__}")

    def set(
        self, key: str, item: COMPATIBLE_TYPES, inplace: bool = False, **kwargs
    ) -> _TensorDict:
        """Sets a new key-value pair.

        Args:
            key (str): name of the value
            item (torch.Tensor): value to be stored in the tensordict
            inplace (bool, optional): if True and if a key matches an existing
                key in the tensordict, then the update will occur in-place
                for that key-value pair. Default is `False`.

        Returns:
            self

        """
        raise NotImplementedError(f"{self.__class__.__name__}")

    @abc.abstractmethod
    def set_(
        self, key: str, item: COMPATIBLE_TYPES, no_check: bool = False
    ) -> _TensorDict:
        """Sets a value to an existing key while keeping the original storage.

        Args:
            key (str): name of the value
            item (torch.Tensor): value to be stored in the tensordict
            no_check (bool, optional): if True, it is assumed that device and shape
                match the original tensor and that the keys is in the tensordict.

        Returns:
            self

        """
        raise NotImplementedError(f"{self.__class__.__name__}")

    @abc.abstractmethod
    def _stack_onto_(
        self,
        key: str,
        list_item: List[COMPATIBLE_TYPES],
        dim: int,
    ) -> _TensorDict:
        """Stacks a list of values onto an existing key while keeping the original storage.

        Args:
            key (str): name of the value
            list_item (list of torch.Tensor): value to be stacked and stored in the tensordict.
            dim (int): dimension along which the tensors should be stacked.
            no_check (bool, optional): if True, it is assumed that device and shape
                match the original tensor and that the keys is in the tensordict.

        Returns:
            self

        """
        raise NotImplementedError(f"{self.__class__.__name__}")

    def _stack_onto_at_(
        self,
        key: str,
        list_item: List[COMPATIBLE_TYPES],
        dim: int,
        idx: INDEX_TYPING,
    ) -> _TensorDict:
        """Similar to _stack_onto_ but on a specific index. Only works with regular TensorDicts."""
        raise RuntimeError(
            f"Cannot call _stack_onto_at_ with {self.__class__.__name__}. "
            "This error is probably caused by a call to a lazy operation before stacking. "
            "Make sure your sub-classed tensordicts are turned into regular tensordicts by calling to_tensordict() "
            "before calling __getindex__ and stack."
        )

    def _default_get(
        self, key: str, default: Union[str, COMPATIBLE_TYPES] = "_no_default_"
    ) -> COMPATIBLE_TYPES:
        if not isinstance(default, str):
            return default
        if default == "_no_default_":
            raise KeyError(
                f'key "{key}" not found in {self.__class__.__name__} with '
                f"keys {sorted(list(self.keys()))}"
            )
        else:
            raise ValueError(
                f"default should be None or a torch.Tensor instance, " f"got {default}"
            )

    @abc.abstractmethod
    def get(
        self, key: str, default: Union[str, COMPATIBLE_TYPES] = "_no_default_"
    ) -> COMPATIBLE_TYPES:
        """
        Gets the value stored with the input key.

        Args:
            key (str): key to be queried.
            default: default value if the key is not found in the tensordict.

        """
        raise NotImplementedError(f"{self.__class__.__name__}")

    def _get_meta(self, key) -> MetaTensor:
        if not isinstance(key, str):
            raise TypeError(f"Expected key to be a string but found {type(key)}")

        try:
            return self._dict_meta[key]
        except KeyError:
            raise KeyError(
                f"key {key} not found in {self.__class__.__name__} with keys"
                f" {sorted(list(self.keys()))}"
            )

    def apply_(self, fn: Callable) -> _TensorDict:
        """Applies a callable to all values stored in the tensordict and
        re-writes them in-place.

        Args:
            fn (Callable): function to be applied to the tensors in the
                tensordict.

        Returns:
            self

        """
        for key, item in self.items():
            item_trsf = fn(item)
            if item_trsf is not None:
                self.set(key, item_trsf, inplace=True)
        return self

    def apply(
        self, fn: Callable, batch_size: Optional[Sequence[int]] = None
    ) -> _TensorDict:
        """Applies a callable to all values stored in the tensordict and sets
        them in a new tensordict.

        Args:
            fn (Callable): function to be applied to the tensors in the
                tensordict.
            batch_size (sequence of int, optional): if provided,
                the resulting TensorDict will have the desired batch_size.
                The `batch_size` argument should match the batch_size after
                the transformation.

        Returns:
            a new tensordict with transformed_in tensors.

        """
        if batch_size is None:
            td = TensorDict({}, batch_size=self.batch_size, device=self._device_safe())
        else:
            td = TensorDict(
                {}, batch_size=torch.Size(batch_size), device=self._device_safe()
            )
        for key, item in self.items():
            item_trsf = fn(item)
            td.set(key, item_trsf)
        return td

    def update(
        self,
        input_dict_or_td: Union[Dict[str, COMPATIBLE_TYPES], _TensorDict],
        clone: bool = False,
        inplace: bool = False,
        **kwargs,
    ) -> _TensorDict:
        """Updates the TensorDict with values from either a dictionary or
            another TensorDict.

        Args:
            input_dict_or_td (_TensorDict or dict): Does not keyword arguments
                (unlike `dict.update()`).
            clone (bool, optional): whether the tensors in the input (
                tensor) dict should be cloned before being set. Default is
                `False`.
            inplace (bool, optional): if True and if a key matches an existing
                key in the tensordict, then the update will occur in-place
                for that key-value pair. Default is `False`.
            **kwargs: keyword arguments for the `TensorDict.set` method

        Returns:
            self

        """
        if input_dict_or_td is self:
            # no op
            return self
        for key, value in input_dict_or_td.items():
            if not isinstance(value, _accepted_classes):
                raise TypeError(
                    f"Expected value to be one of types "
                    f"{_accepted_classes} but got {type(value)}"
                )
            if clone:
                value = value.clone()
            self.set(key, value, inplace=inplace, **kwargs)
        return self

    def update_(
        self,
        input_dict_or_td: Union[Dict[str, COMPATIBLE_TYPES], _TensorDict],
        clone: bool = False,
    ) -> _TensorDict:
        """Updates the TensorDict in-place with values from either a dictionary
        or another TensorDict.

        Unlike TensorDict.update, this function will
        throw an error if the key is unknown to the TensorDict

        Args:
            input_dict_or_td (_TensorDict or dict): Does not keyword
                arguments (unlike `dict.update()`).
            clone (bool, optional): whether the tensors in the input (
                tensor) dict should be cloned before being set. Default is
                `False`.

        Returns:
            self

        """
        if input_dict_or_td is self:
            # no op
            return self
        for key, value in input_dict_or_td.items():
            if not isinstance(value, _accepted_classes):
                raise TypeError(
                    f"Expected value to be one of types {_accepted_classes} "
                    f"but got {type(value)}"
                )
            if clone:
                value = value.clone()
            self.set_(key, value)
        return self

    def update_at_(
        self,
        input_dict_or_td: Union[Dict[str, COMPATIBLE_TYPES], _TensorDict],
        idx: INDEX_TYPING,
        clone: bool = False,
    ) -> _TensorDict:
        """Updates the TensorDict in-place at the specified index with
        values from either a dictionary or another TensorDict.

        Unlike  TensorDict.update, this function will throw an error if the
        key is unknown to the TensorDict.

        Args:
            input_dict_or_td (_TensorDict or dict): Does not keyword arguments
                (unlike `dict.update()`).
            idx (int, torch.Tensor, iterable, slice): index of the tensordict
                where the update should occur.
            clone (bool, optional): whether the tensors in the input (
                tensor) dict should be cloned before being set. Default is
                `False`.

        Returns:
            self

        Examples:
            >>> td = TensorDict(source={'a': torch.zeros(3, 4, 5),
            ...    'b': torch.zeros(3, 4, 10)}, batch_size=[3, 4])
            >>> td.update_at_(
            ...    TensorDict(source={'a': torch.ones(1, 4, 5),
            ...        'b': torch.ones(1, 4, 10)}, batch_size=[1, 4]),
            ...    slice(1, 2))
            TensorDict(
                fields={a: Tensor(torch.Size([3, 4, 5]), dtype=torch.float32),
                    b: Tensor(torch.Size([3, 4, 10]),\
dtype=torch.float32)},
                shared=False,
                batch_size=torch.Size([3, 4]),
                device=cpu)

        """

        for key, value in input_dict_or_td.items():
            if not isinstance(value, _accepted_classes):
                raise TypeError(
                    f"Expected value to be one of types {_accepted_classes} "
                    f"but got {type(value)}"
                )
            if clone:
                value = value.clone()
            self.set_at_(
                key,
                value,
                idx,
            )
        return self

    def _convert_to_tensor(
        self, array: np.ndarray
    ) -> Union[torch.Tensor, MemmapTensor]:
        return torch.as_tensor(array, device=self.device)

    def _process_tensor(
        self,
        input: Union[COMPATIBLE_TYPES, np.ndarray],
        check_device: bool = True,
        check_tensor_shape: bool = True,
        check_shared: bool = False,
    ) -> Union[torch.Tensor, MemmapTensor]:

        if not isinstance(input, _accepted_classes):
            tensor = self._convert_to_tensor(input)
        else:
            tensor = input

        if check_device and self._device_safe() is not None:
            device = self.device
            tensor = tensor.to(device)
        elif self._device_safe() is None:
            if isinstance(tensor, _TensorDict):
                device = tensor._device_safe()
                if device is not None:
                    self.device = device
                else:
                    raise RuntimeError(
                        "The nested tensordict had not device. "
                        "When nesting tensordicts, the nested tensordict must have "
                        "a defined device (either by specifying it upon creation or "
                        "by populating it with at tensor)."
                    )
            else:
                self.device = tensor.device

        if check_shared:
            raise DeprecationWarning("check_shared is not authorized anymore")

        if check_tensor_shape and tensor.shape[: self.batch_dims] != self.batch_size:
            # if TensorDict, let's try to map it to the desired shape
            if (
                isinstance(tensor, _TensorDict)
                and tensor.batch_size[: self.batch_dims] != self.batch_size
            ):
                tensor = tensor.clone(recursive=False)
                tensor.batch_size = self.batch_size
            else:
                raise RuntimeError(
                    f"batch dimension mismatch, got self.batch_size"
                    f"={self.batch_size} and tensor.shape[:self.batch_dims]"
                    f"={tensor.shape[: self.batch_dims]}"
                )

        # minimum ndimension is 1
        if tensor.ndimension() == self.ndimension() and not isinstance(
            tensor, _TensorDict
        ):
            tensor = tensor.unsqueeze(-1)

        return tensor

    @abc.abstractmethod
    def pin_memory(self) -> _TensorDict:
        """Calls pin_memory() on the stored tensors."""
        raise NotImplementedError(f"{self.__class__.__name__}")

    # @abc.abstractmethod
    # def is_pinned(self) -> bool:
    #     """Checks if tensors are pinned."""
    #     raise NotImplementedError(f"{self.__class__.__name__}")

    def items(self) -> Iterator[Tuple[str, COMPATIBLE_TYPES]]:
        """
        Returns a generator of key-value pairs for the tensordict.

        """
        for k in self.keys():
            yield k, self.get(k)

    def values(self) -> Iterator[COMPATIBLE_TYPES]:
        """
        Returns a generator representing the values for the tensordict.

        """
        for k in self.keys():
            yield self.get(k)

    def items_meta(self, make_unset: bool = True) -> Iterator[Tuple[str, MetaTensor]]:
        """Returns a generator of key-value pairs for the tensordict, where the
        values are MetaTensor instances corresponding to the stored tensors.

        """
        if make_unset:
            for k in self.keys():
                yield k, self._get_meta(k)
        else:
            return self._dict_meta.items()

    def values_meta(self, make_unset: bool = True) -> Iterator[MetaTensor]:
        """Returns a generator representing the values for the tensordict, those
        values are MetaTensor instances corresponding to the stored tensors.

        """
        if make_unset:
            for k in self.keys():
                yield self._get_meta(k)
        else:
            return self._dict_meta.values()

    @abc.abstractmethod
    def keys(self) -> KeysView:
        """Returns a generator of tensordict keys."""

        raise NotImplementedError(f"{self.__class__.__name__}")

    def expand(self, *shape: int) -> _TensorDict:
        """Expands each tensors of the tensordict according to
        `tensor.expand(*shape, *tensor.shape)`

        Examples:
            >>> td = TensorDict(source={'a': torch.zeros(3, 4, 5),
            ...     'b': torch.zeros(3, 4, 10)}, batch_size=[3, 4])
            >>> td_expand = td.expand(10)
            >>> assert td_expand.shape == torch.Size([10, 3, 4])
            >>> assert td_expand.get("a").shape == torch.Size([10, 3, 4, 5])
        """
        d = dict()
        for key, value in self.items():
            if isinstance(value, _TensorDict):
                d[key] = value.expand(*shape)
            else:
                d[key] = value.expand(*shape, *value.shape)
        return TensorDict(
            source=d,
            batch_size=[*shape, *self.batch_size],
            device=self._device_safe(),
        )

    def __bool__(self) -> bool:
        raise ValueError("Converting a tensordict to boolean value is not permitted")

    def __ne__(self, other: object) -> _TensorDict:
        """XOR operation over two tensordicts, for evey key. The two
        tensordicts must have the same key set.

        Returns:
            a new TensorDict instance with all tensors are boolean
            tensors of the same shape as the original tensors.

        """

        if not isinstance(other, _TensorDict):
            raise TypeError(
                f"TensorDict comparision requires both objects to be "
                f"_TensorDict subclass, got {type(other)}"
            )
        keys1 = set(self.keys())
        keys2 = set(other.keys())
        if len(keys1.difference(keys2)) or len(keys1) != len(keys2):
            raise KeyError(
                f"keys in {self} and {other} mismatch, got {keys1} and {keys2}"
            )
        d = dict()
        for (key, item1) in self.items():
            d[key] = item1 != other.get(key)
        return TensorDict(
            batch_size=self.batch_size, source=d, device=self._device_safe()
        )

    def __eq__(self, other: object) -> _TensorDict:
        """Compares two tensordicts against each other, for evey key. The two
        tensordicts must have the same key set.

        Returns:
            a new TensorDict instance with all tensors are boolean
            tensors of the same shape as the original tensors.

        """
        if not isinstance(other, (_TensorDict, float, int)):
            raise TypeError(
                f"TensorDict comparision requires both objects to be "
                f"_TensorDict subclass, got {type(other)}"
            )
        if not isinstance(other, _TensorDict):
            return TensorDict(
                {key: value == other for key, value in self.items()}, self.batch_size
            )
        keys1 = set(self.keys())
        keys2 = set(other.keys())
        if len(keys1.difference(keys2)) or len(keys1) != len(keys2):
            raise KeyError(f"keys in tensordicts mismatch, got {keys1} and {keys2}")
        d = dict()
        for (key, item1) in self.items():
            d[key] = item1 == other.get(key)
        return TensorDict(
            batch_size=self.batch_size, source=d, device=self._device_safe()
        )

    @abc.abstractmethod
    def del_(self, key: str) -> _TensorDict:
        """Deletes a key of the tensordict.

        Args:
            key (str): key to be deleted

        Returns:
            self

        """
        raise NotImplementedError(f"{self.__class__.__name__}")

    @abc.abstractmethod
    def select(self, *keys: str, inplace: bool = False) -> _TensorDict:
        """Selects the keys of the tensordict and returns an new tensordict
        with only the selected keys.

        The values are not copied: in-place modifications a tensor of either
        of the original or new tensordict will result in a change in both
        tensordicts.

        Args:
            *keys (str): keys to select
            inplace (bool): if True, the tensordict is pruned in place.
                Default is `False`.

        Returns:
            A new tensordict with the selected keys only.

        """
        raise NotImplementedError(f"{self.__class__.__name__}")

    def exclude(self, *keys: str, inplace: bool = False) -> _TensorDict:
        keys = [key for key in self.keys() if key not in keys]
        return self.select(*keys, inplace=inplace)

    @abc.abstractmethod
    def set_at_(
        self, key: str, value: COMPATIBLE_TYPES, idx: INDEX_TYPING
    ) -> _TensorDict:
        """Sets the values in-place at the index indicated by `idx`.

        Args:
            key (str): key to be modified.
            value (torch.Tensor): value to be set at the index `idx`
            idx (int, tensor or tuple): index where to write the values.

        Returns:
            self

        """
        raise NotImplementedError(f"{self.__class__.__name__}")

    def copy_(self, tensordict: _TensorDict) -> _TensorDict:
        """See `_TensorDict.update_`."""
        return self.update_(tensordict)

    def copy_at_(self, tensordict: _TensorDict, idx: INDEX_TYPING) -> _TensorDict:
        """See `_TensorDict.update_at_`."""
        return self.update_at_(tensordict, idx)

    def get_at(
        self, key: str, idx: INDEX_TYPING, default: COMPATIBLE_TYPES = "_no_default_"
    ) -> COMPATIBLE_TYPES:
        """Get the value of a tensordict from the key `key` at the index `idx`.

        Args:
            key (str): key to be retrieved.
            idx (int, slice, torch.Tensor, iterable): index of the tensor.
            default (torch.Tensor): default value to return if the key is
                not present in the tensordict.

        Returns:
            indexed tensor.

        """
        value = self.get(key, default=default)
        if value is not default:
            return value[idx]
        return value

    @abc.abstractmethod
    def share_memory_(self) -> _TensorDict:
        """Places all the tensors in shared memory.

        Returns:
            self.

        """
        raise NotImplementedError(f"{self.__class__.__name__}")

    @abc.abstractmethod
    def memmap_(self, prefix=None) -> _TensorDict:
        """Writes all tensors onto a MemmapTensor.

        Args:
            prefix (str): directory prefix where the memmap tensors will have to
                be stored.

        Returns:
            self.

        """

        raise NotImplementedError(f"{self.__class__.__name__}")

    @abc.abstractmethod
    def detach_(self) -> _TensorDict:
        """Detach the tensors in the tensordict in-place.

        Returns:
            self.

        """
        raise NotImplementedError(f"{self.__class__.__name__}")

    def detach(self) -> _TensorDict:
        """Detach the tensors in the tensordict.

        Returns:
            a new tensordict with no tensor requiring gradient.

        """

        return TensorDict(
            {key: item.detach() for key, item in self.items()},
            batch_size=self.batch_size,
            device=self._device_safe(),
        )

    def to_tensordict(self):
        """Returns a regular TensorDict instance from the _TensorDict.

        Returns:
            a new TensorDict object containing the same values.

        """
        return self.to(
            TensorDict,
            batch_size=self.batch_size,
            device=self.device,
            _meta_source=dict(self._dict_meta),
            _run_checks=False,
        )

    def zero_(self) -> _TensorDict:
        """Zeros all tensors in the tensordict in-place."""
        for key in self.keys():
            self.fill_(key, 0)
        return self

    def unbind(self, dim: int) -> Tuple[_TensorDict, ...]:
        """Returns a tuple of indexed tensordicts unbound along the
        indicated dimension. Resulting tensordicts will share
        the storage of the initial tensordict.

        """
        idx = [
            (tuple(slice(None) for _ in range(dim)) + (i,))
            for i in range(self.shape[dim])
        ]
        return tuple(self[_idx] for _idx in idx)

    def chunk(self, chunks: int, dim: int = 0) -> Tuple[_TensorDict, ...]:
        """Attempts to split a tendordict into the specified number of
        chunks. Each chunk is a view of the input tensordict.

        Args:
            chunks (int): number of chunks to return
            dim (int, optional): dimension along which to split the
                tensordict. Default is 0.

        """
        if chunks < 1:
            raise ValueError(
                f"chunks must be a strictly positive integer, got {chunks}."
            )
        indices = []
        _idx_start = 0
        if chunks > 1:
            interval = _idx_end = self.batch_size[dim] // chunks
        else:
            interval = _idx_end = self.batch_size[dim]
        for c in range(chunks):
            indices.append(slice(_idx_start, _idx_end))
            _idx_start = _idx_end
            if c < chunks - 2:
                _idx_end = _idx_end + interval
            else:
                _idx_end = self.batch_size[dim]
        if dim < 0:
            dim = len(self.batch_size) + dim
        return tuple(self[(*[slice(None) for _ in range(dim)], idx)] for idx in indices)

    def clone(self, recursive: bool = True) -> _TensorDict:
        """Clones a _TensorDict subclass instance onto a new TensorDict.

        Args:
            recursive (bool, optional): if True, each tensor contained in the
                TensorDict will be copied too. Default is `True`.
        """
        return TensorDict(
            source={
                key: value.clone() if recursive else value
                for key, value in self.items()
            },
            batch_size=self.batch_size,
            device=self._device_safe(),
        )

    @classmethod
    def __torch_function__(
        cls,
        func: Callable,
        types,
        args: Tuple = (),
        kwargs: Optional[dict] = None,
    ) -> Callable:
        if kwargs is None:
            kwargs = {}
        if func not in TD_HANDLED_FUNCTIONS or not all(
            issubclass(t, (torch.Tensor, _TensorDict)) for t in types
        ):
            return NotImplemented
        return TD_HANDLED_FUNCTIONS[func](*args, **kwargs)

    @abc.abstractmethod
    def to(self, dest: Union[DEVICE_TYPING, Type, torch.Size], **kwargs) -> _TensorDict:
        """Maps a _TensorDict subclass either on a new device or to another
        _TensorDict subclass (if permitted). Casting tensors to a new dtype
        is not allowed, as tensordicts are not bound to contain a single
        tensor dtype.

        Args:
            dest (device, size or _TensorDict subclass): destination of the
                tensordict. If it is a torch.Size object, the batch_size
                will be updated provided that it is compatible with the
                stored tensors.

        Returns:
            a new tensordict. If device indicated by dest differs from
            the tensordict device, this is a no-op.

        """
        raise NotImplementedError

    def _check_new_batch_size(self, new_size: torch.Size):
        n = len(new_size)
        for key, meta_tensor in self.items_meta():
            if (meta_tensor.ndimension() <= n) or (meta_tensor.shape[:n] != new_size):
                if meta_tensor.ndimension() == n and meta_tensor.shape == new_size:
                    raise RuntimeError(
                        "TensorDict requires tensors that have at least one more "
                        f'dimension than the batch_size. The tensor "{key}" has shape '
                        f"{meta_tensor.shape} which is the same as the new size."
                    )
                raise RuntimeError(
                    f"the tensor {key} has shape {meta_tensor.shape} which "
                    f"is incompatible with the new shape {new_size}."
                )

    @abc.abstractmethod
    def _change_batch_size(self, new_size: torch.Size):
        raise NotImplementedError

    def cpu(self) -> _TensorDict:
        """Casts a tensordict to cpu (if not already on cpu)."""
        return self.to("cpu")

    def cuda(self, device: int = 0) -> _TensorDict:
        """Casts a tensordict to a cuda device (if not already on it)."""
        return self.to(f"cuda:{device}")

    @abc.abstractmethod
    def masked_fill_(
        self, mask: torch.Tensor, value: Union[float, bool]
    ) -> _TensorDict:
        """Fills the values corresponding to the mask with the desired value.

        Args:
            mask (boolean torch.Tensor): mask of values to be filled. Shape
                must match tensordict shape.
            value: value to used to fill the tensors.

        Returns:
            self

        Examples:
            >>> td = TensorDict(source={'a': torch.zeros(3, 4)},
            ...     batch_size=[3])
            >>> mask = torch.tensor([True, False, False])
            >>> _ = td.masked_fill_(mask, 1.0)
            >>> td.get("a")
            tensor([[1., 1., 1., 1.],
                    [0., 0., 0., 0.],
                    [0., 0., 0., 0.]])
        """
        raise NotImplementedError

    @abc.abstractmethod
    def masked_fill(self, mask: torch.Tensor, value: Union[float, bool]) -> _TensorDict:
        """Out-of-place version of masked_fill

        Args:
            mask (boolean torch.Tensor): mask of values to be filled. Shape
                must match tensordict shape.
            value: value to used to fill the tensors.

        Returns:
            self

        Examples:
            >>> td = TensorDict(source={'a': torch.zeros(3, 4)},
            ...     batch_size=[3])
            >>> mask = torch.tensor([True, False, False])
            >>> td1 = td.masked_fill(mask, 1.0)
            >>> td1.get("a")
            tensor([[1., 1., 1., 1.],
                    [0., 0., 0., 0.],
                    [0., 0., 0., 0.]])
        """
        raise NotImplementedError

    def masked_select(self, mask: torch.Tensor) -> _TensorDict:
        """Masks all tensors of the TensorDict and return a new TensorDict
        instance with similar keys pointing to masked values.

        Args:
            mask (torch.Tensor): boolean mask to be used for the tensors.
                Shape must match the TensorDict batch_size.

        Examples:
            >>> td = TensorDict(source={'a': torch.zeros(3, 4)},
            ...    batch_size=[3])
            >>> mask = torch.tensor([True, False, False])
            >>> td_mask = td.masked_select(mask)
            >>> td_mask.get("a")
            tensor([[0., 0., 0., 0.]])

        """
        d = dict()
        for key, value in self.items():
            while mask.ndimension() > self.batch_dims:
                mask_expand = mask.squeeze(-1)
            else:
                mask_expand = mask
            value_select = value[mask_expand]
            d[key] = value_select
        dim = int(mask.sum().item())
        return TensorDict(
            device=self._device_safe(), source=d, batch_size=torch.Size([dim])
        )

    @abc.abstractmethod
    def is_contiguous(self) -> bool:
        """

        Returns:
            boolean indicating if all the tensors are contiguous.

        """
        raise NotImplementedError

    @abc.abstractmethod
    def contiguous(self) -> _TensorDict:
        """

        Returns:
            a new tensordict of the same type with contiguous values (
            or self if values are already contiguous).

        """
        raise NotImplementedError

    def to_dict(self) -> dict:
        """

        Returns:
            dictionary with key-value pairs matching those of the
            tensordict.

        """
        return {key: value for key, value in self.items()}

    def unsqueeze(self, dim: int) -> _TensorDict:
        """Unsqueeze all tensors for a dimension comprised in between
        `-td.batch_dims` and `td.batch_dims` and returns them in a new
        tensordict.

        Args:
            dim (int): dimension along which to unsqueeze

        """
        if dim < 0:
            dim = self.batch_dims + dim + 1

        if (dim > self.batch_dims) or (dim < 0):
            raise RuntimeError(
                f"unsqueezing is allowed for dims comprised between "
                f"`-td.batch_dims` and `td.batch_dims` only. Got "
                f"dim={dim} with a batch size of {self.batch_size}."
            )
        return UnsqueezedTensorDict(
            source=self,
            custom_op="unsqueeze",
            inv_op="squeeze",
            custom_op_kwargs={"dim": dim},
            inv_op_kwargs={"dim": dim},
        )

    def squeeze(self, dim: int) -> _TensorDict:
        """Squeezes all tensors for a dimension comprised in between
        `-td.batch_dims+1` and `td.batch_dims-1` and returns them
        in a new tensordict.

        Args:
            dim (int): dimension along which to squeeze

        """
        if dim < 0:
            dim = self.batch_dims + dim

        if self.batch_dims and (dim >= self.batch_dims or dim < 0):
            raise RuntimeError(
                f"squeezing is allowed for dims comprised between 0 and "
                f"td.batch_dims only. Got dim={dim} and batch_size"
                f"={self.batch_size}."
            )

        if dim >= self.batch_dims or self.batch_size[dim] != 1:
            return self
        return SqueezedTensorDict(
            source=self,
            custom_op="squeeze",
            inv_op="unsqueeze",
            custom_op_kwargs={"dim": dim},
            inv_op_kwargs={"dim": dim},
        )

    def reshape(
        self,
        *shape: int,
        size: Optional[Union[List, Tuple, torch.Size]] = None,
    ) -> _TensorDict:
        """Returns a contiguous, reshaped tensor of the desired shape.

        Args:
            *shape (int): new shape of the resulting tensordict.
            size: iterable

        Returns:
            A TensorDict with reshaped keys

        """
        if len(shape) == 0 and size is not None:
            return self.view(*size)
        elif len(shape) == 1 and isinstance(shape[0], (list, tuple, torch.Size)):
            return self.view(*shape[0])
        elif not isinstance(shape, torch.Size):
            shape = torch.Size(shape)

        d = {}
        for key, item in self.items():
            d[key] = item.reshape(*shape, *item.shape[self.ndimension() :])
        if len(d):
            batch_size = d[key].shape[: len(shape)]
        else:
            if any(not isinstance(i, int) or i < 0 for i in shape):
                raise RuntimeError(
                    "Implicit reshaping is not permitted with empty " "tensordicts"
                )
            batch_size = shape
        return TensorDict(d, batch_size, device=self._device_safe())

    def view(
        self,
        *shape: int,
        size: Optional[Union[List, Tuple, torch.Size]] = None,
    ) -> _TensorDict:
        """Returns a tensordict with views of the tensors according to a new
        shape, compatible with the tensordict batch_size.

        Args:
            *shape (int): new shape of the resulting tensordict.
            size: iterable

        Returns:
            a new tensordict with the desired batch_size.

        Examples:
            >>> td = TensorDict(source={'a': torch.zeros(3,4,5),
            ...    'b': torch.zeros(3,4,10,1)}, batch_size=torch.Size([3, 4]))
            >>> td_view = td.view(12)
            >>> print(td_view.get("a").shape)  # torch.Size([12, 5])
            >>> print(td_view.get("b").shape)  # torch.Size([12, 10, 1])
            >>> td_view = td.view(-1, 4, 3)
            >>> print(td_view.get("a").shape)  # torch.Size([1, 4, 3, 5])
            >>> print(td_view.get("b").shape)  # torch.Size([1, 4, 3, 10, 1])

        """
        if len(shape) == 0 and size is not None:
            return self.view(*size)
        elif len(shape) == 1 and isinstance(shape[0], (list, tuple, torch.Size)):
            return self.view(*shape[0])
        elif not isinstance(shape, torch.Size):
            shape = torch.Size(shape)
        return ViewedTensorDict(
            source=self,
            custom_op="view",
            inv_op="view",
            custom_op_kwargs={"size": shape},
            inv_op_kwargs={"size": self.batch_size},
        )

    def permute(
        self,
        *dims_list: int,
        dims=None,
    ) -> _TensorDict:
        """Returns a view of a tensordict with the batch dimensions permuted according to dims

        Args:
            *dims_list (int): the new ordering of the batch dims of the tensordict. Alternatively,
                a single iterable of integers can be provided.
            dims (list of int): alternative way of calling permute(...).

        Returns:
            a new tensordict with the batch dimensions in the desired order.

        Examples:
            >>> tensordict = TensorDict({"a": torch.randn(3, 4, 5)}, [3, 4])
            >>> print(tensordict.permute([1, 0]))
            PermutedTensorDict(
                source=TensorDict(
                    fields={
                        a: Tensor(torch.Size([3, 4, 5]), dtype=torch.float32)},
                    batch_size=torch.Size([3, 4]),
                    device=cpu,
                    is_shared=False),
                op=permute(dims=[1, 0]))
            >>> print(tensordict.permute(1, 0))
            PermutedTensorDict(
                source=TensorDict(
                    fields={
                        a: Tensor(torch.Size([3, 4, 5]), dtype=torch.float32)},
                    batch_size=torch.Size([3, 4]),
                    device=cpu,
                    is_shared=False),
                op=permute(dims=[1, 0]))
            >>> print(tensordict.permute(dims=[1, 0]))
            PermutedTensorDict(
                source=TensorDict(
                    fields={
                        a: Tensor(torch.Size([3, 4, 5]), dtype=torch.float32)},
                    batch_size=torch.Size([3, 4]),
                    device=cpu,
                    is_shared=False),
                op=permute(dims=[1, 0]))
        """
        if len(dims_list) == 0:
            dims_list = dims
        elif len(dims_list) == 1 and not isinstance(dims_list[0], int):
            dims_list = dims_list[0]
        if len(dims_list) != len(self.shape):
            raise RuntimeError(
                f"number of dims don't match in permute (got {len(dims_list)}, expected {len(self.shape)}"
            )

        min_dim, max_dim = -self.batch_dims, self.batch_dims - 1
        seen = [False for dim in range(max_dim + 1)]
        for idx in dims_list:
            if idx < min_dim or idx > max_dim:
                raise IndexError(
                    f"dimension out of range (expected to be in range of [{min_dim}, {max_dim}], but got {idx})"
                )
            if seen[idx]:
                raise RuntimeError("repeated dim in permute")
            seen[idx] = True

        return PermutedTensorDict(
            source=self,
            custom_op="permute",
            inv_op="permute",
            custom_op_kwargs={"dims": dims_list},
            inv_op_kwargs={"dims": dims_list},
        )

    def __repr__(self) -> str:
        fields = _td_fields(self)
        field_str = indent(f"fields={{{fields}}}", 4 * " ")
        batch_size_str = indent(f"batch_size={self.batch_size}", 4 * " ")
        device_str = indent(f"device={self.device}", 4 * " ")
        is_shared_str = indent(f"is_shared={self.is_shared()}", 4 * " ")
        string = ",\n".join([field_str, batch_size_str, device_str, is_shared_str])
        return f"{type(self).__name__}(\n{string})"

    def all(self, dim: int = None) -> Union[bool, _TensorDict]:
        """Checks if all values are True/non-null in the tensordict.

        Args:
            dim (int, optional): if None, returns a boolean indicating
                whether all tensors return `tensor.all() == True`
                If integer, all is called upon the dimension specified if
                and only if this dimension is compatible with the tensordict
                shape.

        """
        if dim is not None and (dim >= self.batch_dims or dim <= -self.batch_dims):
            raise RuntimeError(
                "dim must be greater than -tensordict.batch_dims and smaller "
                "than tensordict.batchdims"
            )
        if dim is not None:
            if dim < 0:
                dim = self.batch_dims + dim
            return TensorDict(
                source={key: value.all(dim=dim) for key, value in self.items()},
                batch_size=[b for i, b in enumerate(self.batch_size) if i != dim],
                device=self._device_safe(),
            )
        return all(value.all() for value in self.values())

    def any(self, dim: int = None) -> Union[bool, _TensorDict]:
        """Checks if any value is True/non-null in the tensordict.

        Args:
            dim (int, optional): if None, returns a boolean indicating
                whether all tensors return `tensor.any() == True`.
                If integer, all is called upon the dimension specified if
                and only if this dimension is compatible with
                the tensordict shape.

        """
        if dim is not None and (dim >= self.batch_dims or dim <= -self.batch_dims):
            raise RuntimeError(
                "dim must be greater than -tensordict.batch_dims and smaller "
                "than tensordict.batchdims"
            )
        if dim is not None:
            if dim < 0:
                dim = self.batch_dims + dim
            return TensorDict(
                source={key: value.any(dim=dim) for key, value in self.items()},
                batch_size=[b for i, b in enumerate(self.batch_size) if i != dim],
                device=self._device_safe(),
            )
        return any([value.any() for key, value in self.items()])

    def get_sub_tensordict(self, idx: INDEX_TYPING) -> _TensorDict:
        """Returns a SubTensorDict with the desired index."""
        sub_td = SubTensorDict(
            source=self,
            idx=idx,
        )
        return sub_td

    def __iter__(self) -> Generator:
        if not self.batch_dims:
            raise StopIteration
        length = self.batch_size[0]
        for i in range(length):
            yield self[i]

    def flatten_keys(self, separator: str = ",", inplace: bool = True) -> _TensorDict:
        to_flatten = []
        for key, meta_value in self.items_meta():
            if meta_value.is_tensordict():
                to_flatten.append(key)

        if inplace:
            for key in to_flatten:
                inner_tensordict = self.get(key).flatten_keys(
                    separator=separator, inplace=inplace
                )
                for inner_key, inner_item in inner_tensordict.items():
                    self.set(separator.join([key, inner_key]), inner_item)
            for key in to_flatten:
                del self[key]
            return self
        else:
            tensordict_out = TensorDict(
                {}, batch_size=self.batch_size, device=self.device
            )
            for key, value in self.items():
                if key in to_flatten:
                    inner_tensordict = self.get(key).flatten_keys(
                        separator=separator, inplace=inplace
                    )
                    for inner_key, inner_item in inner_tensordict.items():
                        tensordict_out.set(separator.join([key, inner_key]), inner_item)
                else:
                    tensordict_out.set(key, value)
            return tensordict_out

    def unflatten_keys(self, separator: str = ",", inplace: bool = True) -> _TensorDict:
        to_unflatten = defaultdict(lambda: list())
        for key in self.keys():
            if separator in key[1:-1]:
                split_key = key.split(separator)
                to_unflatten[split_key[0]].append((key, separator.join(split_key[1:])))

        if not inplace:
            out = TensorDict(
                {
                    key: value
                    for key, value in self.items()
                    if separator not in key[1:-1]
                },
                batch_size=self.batch_size,
                device=self.device,
            )
        else:
            out = self

        for key, list_of_keys in to_unflatten.items():
            tensordict = TensorDict({}, batch_size=self.batch_size, device=self.device)
            for (old_key, new_key) in list_of_keys:
                value = self[old_key]
                tensordict[new_key] = value
                if inplace:
                    del self[old_key]
            out.set(key, tensordict.unflatten_keys(separator=separator))
        return out

    def __len__(self) -> int:
        """

        Returns:
            Length of first dimension, if there is, otherwise 0.

        """
        return self.shape[0] if self.batch_dims else 0

    def __getitem__(self, idx: INDEX_TYPING) -> _TensorDict:
        """Indexes all tensors according to idx and returns a new tensordict
        where the values share the storage of the original tensors (even
        when the index is a torch.Tensor). Any in-place modification to the
        resulting tensordict will impact the parent tensordict too.

        Examples:
            >>> td = TensorDict(source={'a': torch.zeros(3,4,5)},
            ...     batch_size=torch.Size([3, 4]))
            >>> subtd = td[torch.zeros(1, dtype=torch.long)]
            >>> assert subtd.shape == torch.Size([1,4])
            >>> subtd.set("a", torch.ones(1,4,5))
            >>> print(td.get("a"))  # first row is full of 1
            >>> # Warning: this will not work as expected
            >>> subtd.get("a")[:] = 2.0
            >>> print(td.get("a"))  # values have not changed

        """
        if isinstance(idx, list):
            idx = torch.tensor(idx, device=self.device)
        if isinstance(idx, tuple) and any(
            isinstance(sub_index, list) for sub_index in idx
        ):
            idx = tuple(
                torch.tensor(sub_index, device=self.device)
                if isinstance(sub_index, list)
                else sub_index
                for sub_index in idx
            )
        if isinstance(idx, str):
            return self.get(idx)
        if isinstance(idx, tuple) and sum(
            isinstance(_idx, str) for _idx in idx
        ) not in [len(idx), 0]:
            raise IndexError(_STR_MIXED_INDEX_ERROR)
        elif isinstance(idx, Number):
            idx = (idx,)
        elif isinstance(idx, tuple) and all(
            isinstance(sub_idx, str) for sub_idx in idx
        ):
            out = self.get(idx[0])
            if len(idx) > 1:
                return out[idx[1:]]
            else:
                return out
        # elif isinstance(idx, torch.Tensor) and idx.dtype == torch.bool:
        #     return self.masked_select(idx)

        contiguous_input = (int, slice)
        return_simple_view = isinstance(idx, contiguous_input) or (
            isinstance(idx, tuple)
            and all(isinstance(_idx, contiguous_input) for _idx in idx)
        )
        if not self.batch_size:
            raise RuntimeError(
                "indexing a tensordict with td.batch_dims==0 is not permitted"
            )

        if isinstance(idx, np.ndarray):
            idx = torch.tensor(idx, device=self.device)
        if idx is Ellipsis or (isinstance(idx, tuple) and Ellipsis in idx):
            idx = convert_ellipsis_to_idx(idx, self.batch_size)

        if return_simple_view and not self.is_memmap():
            # We exclude memmap tensordicts such that indexing is achieved only when needed
            #  as  SubTenssorDicts are lazy objects
            return TensorDict(
                source={key: item[idx] for key, item in self.items()},
                _meta_source={key: item[idx] for key, item in self.items_meta()},
                batch_size=_getitem_batch_size(self.batch_size, idx),
                device=self._device_safe(),
            )
        # SubTensorDict keeps the same storage as TensorDict
        # in all cases not accounted for above
        return self.get_sub_tensordict(idx)

    def __setitem__(self, index: INDEX_TYPING, value: _TensorDict) -> None:
        if index is Ellipsis or (isinstance(index, tuple) and Ellipsis in index):
            index = convert_ellipsis_to_idx(index, self.batch_size)
        if isinstance(index, list):
            index = torch.tensor(index, device=self.device)
        if isinstance(index, tuple) and any(
            isinstance(sub_index, list) for sub_index in index
        ):
            index = tuple(
                torch.tensor(sub_index, device=self.device)
                if isinstance(sub_index, list)
                else sub_index
                for sub_index in index
            )
        if isinstance(index, tuple) and sum(
            isinstance(_index, str) for _index in index
        ) not in [len(index), 0]:
            raise IndexError(_STR_MIXED_INDEX_ERROR)
        if isinstance(index, str):
            self.set(index, value, inplace=False)
        elif isinstance(index, tuple) and all(
            isinstance(sub_index, str) for sub_index in index
        ):
            try:
                if len(index) == 1:
                    return self.set(
                        index[0], value, inplace=isinstance(self, SubTensorDict)
                    )
                self[index[:-1]] = self[index[:-1]].set(
                    index[-1], value, inplace=isinstance(self, SubTensorDict)
                )
            except AttributeError as err:
                if "for populating tensordict with new key-value pair" in str(err):
                    raise RuntimeError(
                        "Trying to replace an existing nested tensordict with "
                        "another one with non-matching keys. This leads to "
                        "unspecified behaviours and is prohibited."
                    )
                raise err
        else:
            indexed_bs = _getitem_batch_size(self.batch_size, index)
            if value.batch_size != indexed_bs:
                raise RuntimeError(
                    f"indexed destination TensorDict batch size is {indexed_bs} "
                    f"(batch_size = {self.batch_size}, index={index}), "
                    f"which differs from the source batch size {value.batch_size}"
                )
            keys = set(self.keys())
            if not all(key in keys for key in value.keys()):
                subtd = self.get_sub_tensordict(index)
            for key, item in value.items():
                if key in keys:
                    self.set_at_(key, item, index)
                else:
                    subtd.set(key, item)

    def __delitem__(self, index: INDEX_TYPING) -> _TensorDict:
        if isinstance(index, str):
            return self.del_(index)
        raise IndexError(f"Index has to a string but received {index}.")

    @abc.abstractmethod
    def rename_key(self, old_key: str, new_key: str, safe: bool = False) -> _TensorDict:
        """Renames a key with a new string.

        Args:
            old_key (str): key to be renamed
            new_key (str): new name
            safe (bool, optional): if True, an error is thrown when the new
                key is already present in the TensorDict.

        Returns:
            self

        """
        raise NotImplementedError

    def fill_(self, key: str, value: Union[float, bool]) -> _TensorDict:
        """Fills a tensor pointed by the key with the a given value.

        Args:
            key (str): key to be remaned
            value (Number, bool): value to use for the filling

        Returns:
            self

        """
        meta_tensor = self._get_meta(key)
        shape = meta_tensor.shape
        device = meta_tensor.device
        dtype = meta_tensor.dtype
        if meta_tensor.is_tensordict():
            tensordict = self.get(key)
            tensordict.apply_(lambda x: x.fill_(value))
            self.set_(key, tensordict)
        else:
            tensor = torch.full(shape, value, device=device, dtype=dtype)
            self.set_(key, tensor)
        return self

    def empty(self) -> _TensorDict:
        """Returns a new, empty tensordict with the same device and batch size."""
        return self.select()

    def is_empty(self):
        for i in self.items_meta():
            return False
        return True

    @property
    def is_locked(self):
        if not hasattr(self, "_is_locked"):
            self._is_locked = False
        return self._is_locked

    @is_locked.setter
    def is_locked(self, value: bool):
        self._is_locked = value


class TensorDict(_TensorDict):
    """A batched dictionary of tensors.

    TensorDict is a tensor container where all tensors are stored in a
    key-value pair fashion and where each element shares at least the
    following features:
    - device;
    - memory location (shared, memory-mapped array, ...);
    - batch size (i.e. n^th first dimensions).

    TensorDict instances support many regular tensor operations as long as
    they are dtype-independent (as a TensorDict instance can contain tensors
    of many different dtypes). Those operations include (but are not limited
    to):

    - operations on shape: when a shape operation is called (indexing,
      reshape, view, expand, transpose, permute,
      unsqueeze, squeeze, masking etc), the operations is done as if it
      was done on a tensor of the same shape as the batch size then
      expended to the right, e.g.:

        >>> td = TensorDict({'a': torch.zeros(3,4,5)}, batch_size=[3, 4])
        >>> # returns a TensorDict of batch size [3, 4, 1]
        >>> td_unsqueeze = td.unsqueeze(-1)
        >>> # returns a TensorDict of batch size [12]
        >>> td_view = td.view(-1)
        >>> # returns a tensor of batch size [12, 4]
        >>> a_view = td.view(-1).get("a")

    - casting operations: a TensorDict can be cast on a different device
      or another TensorDict type using

        >>> td_cpu = td.to("cpu")
        >>> td_savec = td.to(SavedTensorDict)  # TensorDict saved on disk
        >>> dictionary = td.to_dict()

      A call of the `.to()` method with a dtype will return an error.

    - Cloning, contiguous

    - Reading: `td.get(key)`, `td.get_at(key, index)`

    - Content modification: `td.set(key, value)`, `td.set_(key, value)`,
      `td.update(td_or_dict)`, `td.update_(td_or_dict)`, `td.fill_(key,
      value)`, `td.rename_key(old_name, new_name)`, etc.

    - Operations on multiple tensordicts: `torch.cat(tensordict_list, dim)`,
      `torch.stack(tensordict_list, dim)`, `td1 == td2` etc.

    Args:
        source (TensorDict or dictionary): a data source. If empty, the
            tensordict can be populated subsequently.
        batch_size (iterable of int, optional): a batch size for the
            tensordict. The batch size is immutable and can only be modified
            by calling operations that create a new TensorDict. Unless the
            source is another TensorDict, the batch_size argument must be
            provided as it won't be inferred from the data.
        device (torch.device or compatible type, optional): a device for the
            TensorDict. If the source is non-empty and the device is
            missing, it will be inferred from the input dictionary, assuming
            that all tensors are on the same device.

    Examples:
        >>> import torch
        >>> from torchrl.data import TensorDict
        >>> source = {'random': torch.randn(3, 4),
        ...     'zeros': torch.zeros(3, 4, 5)}
        >>> batch_size = [3]
        >>> td = TensorDict(source, batch_size)
        >>> print(td.shape)  # equivalent to td.batch_size
        torch.Size([3])
        >>> td_unqueeze = td.unsqueeze(-1)
        >>> print(td_unqueeze.get("zeros").shape)
        torch.Size([3, 1, 4, 5])
        >>> print(td_unqueeze[0].shape)
        torch.Size([1])
        >>> print(td_unqueeze.view(-1).shape)
        torch.Size([3])
        >>> print((td.clone()==td).all())
        True

    """

    @classmethod
    def __new__(cls, *args, **kwargs):
        cls._safe = True
        cls._lazy = False
        cls._is_shared = None
        cls._is_memmap = None
        return _TensorDict.__new__(cls)

    def __init__(
        self,
        source: Union[_TensorDict, dict],
        batch_size: Optional[Union[Sequence[int], torch.Size, int]] = None,
        device: Optional[DEVICE_TYPING] = None,
        _meta_source: Optional[dict] = None,
        _run_checks: bool = True,
        _is_shared: Optional[bool] = None,
        _is_memmap: Optional[bool] = None,
    ) -> object:
        super().__init__()

        self._tensordict: Dict = dict()

        self._is_shared = _is_shared
        self._is_memmap = _is_memmap

        if not isinstance(source, (_TensorDict, dict)):
            raise ValueError(
                "A TensorDict source is expected to be a _TensorDict "
                f"sub-type or a dictionary, found type(source)={type(source)}."
            )
        if isinstance(
            batch_size,
            (
                Number,
                Sequence,
            ),
        ):
            if not isinstance(batch_size, torch.Size):
                if isinstance(batch_size, int):
                    batch_size = torch.Size([batch_size])
                else:
                    batch_size = torch.Size(batch_size)
            self._batch_size = batch_size

        elif isinstance(source, _TensorDict):
            self._batch_size = source.batch_size
        else:
            raise ValueError(
                "batch size was not specified when creating the TensorDict "
                "instance and it could not be retrieved from source."
            )

        if isinstance(source, _TensorDict) and device is None:
            device = source._device_safe()
        elif device is not None:
            device = torch.device(device)

        map_item_to_device = device is not None
        self._device = device

        if source is not None:
            for key, value in source.items():
                if map_item_to_device:
                    value = value.to(device)
                _meta_val = (
                    None
                    if _meta_source is None or key not in _meta_source
                    else _meta_source[key]
                )
                if (
                    isinstance(value, _TensorDict)
                    and value.batch_size[: self.batch_dims] != self.batch_size
                ):
                    value.batch_size = self.batch_size
                self.set(key, value, _meta_val=_meta_val, _run_checks=False)

        if _run_checks:
            self._check_batch_size()
            self._check_device()

    def _make_meta(self, key: str) -> MetaTensor:
        proc_value = self._tensordict[key]
        is_memmap = (
            self._is_memmap
            if self._is_memmap is not None
            else isinstance(proc_value, MemmapTensor)
        )
        is_shared = (
            self._is_shared if self._is_shared is not None else proc_value.is_shared()
        )
        return MetaTensor(
            proc_value,
            _is_memmap=is_memmap,
            _is_shared=is_shared,
            _is_tensordict=isinstance(proc_value, _TensorDict),
        )

    @property
    def batch_dims(self) -> int:
        return len(self.batch_size)

    @batch_dims.setter
    def batch_dims(self, value: COMPATIBLE_TYPES) -> None:
        raise RuntimeError(
            f"Setting batch dims on {self.__class__.__name__} instances is "
            f"not allowed."
        )

    @property
    def device(self) -> torch.device:
        device = self._device
        if device is None:
            raise RuntimeError(
                "Querying a tensordict device when it has not been set is not "
                "permitted. Either populate the tensordict with a tensor or set "
                "the device upon creation using de `device=dest` keyword."
            )
        return device

    @device.setter
    def device(self, value: DEVICE_TYPING) -> None:
        if self._device is None:
            self._device = torch.device(value)
        else:
            raise RuntimeError(
                "device cannot be set using tensordict.device = device, "
                "because device cannot be updated in-place. To update device, use "
                "tensordict.to(new_device), which will return a new tensordict "
                "on the new device."
            )

    def _device_safe(self) -> Union[None, torch.device]:
        return self._device

    @property
    def batch_size(self) -> torch.Size:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, new_size: torch.Size):
        return self._batch_size_setter(new_size)

    def _change_batch_size(self, new_size: torch.Size):
        if not hasattr(self, "_orig_batch_size"):
            self._orig_batch_size = self.batch_size
        elif self._orig_batch_size == new_size:
            del self._orig_batch_size
        self._batch_size = new_size

    # Checks
    def _check_is_shared(self) -> bool:
        share_list = [value.is_shared() for key, value in self.items_meta()]
        if any(share_list) and not all(share_list):
            shared_str = ", ".join(
                [f"{key}: {value.is_shared()}" for key, value in self.items_meta()]
            )
            raise RuntimeError(
                f"tensors must be either all shared or not, but mixed "
                f"features is not allowed. "
                f"Found: {shared_str}"
            )
        return all(share_list) and len(share_list) > 0

    def _check_is_memmap(self) -> bool:
        memmap_list = [value.is_memmap() for key, value in self.items_meta()]
        if any(memmap_list) and not all(memmap_list):
            memmap_str = ", ".join(
                [f"{key}: {value.is_memmap()}" for key, value in self.items_meta()]
            )
            raise RuntimeError(
                f"tensors must be either all MemmapTensor or not, but mixed "
                f"features is not allowed. "
                f"Found: {memmap_str}"
            )
        return all(memmap_list) and len(memmap_list) > 0

    def _check_device(self) -> None:
        devices = set(value.device for value in self.values_meta())
        if len(devices) > 1:
            raise RuntimeError(f"Found more than one device: {devices}")

    def pin_memory(self) -> _TensorDict:
        if self.device == torch.device("cpu"):
            for key, value in self.items():
                if isinstance(value, _TensorDict) or (
                    value.dtype in (torch.half, torch.float, torch.double)
                ):
                    self.set(key, value.pin_memory(), inplace=False)
        return self

    def expand(self, *shape: int) -> _TensorDict:
        """Expands every tensor with `(*shape, *tensor.shape)` and returns the
        same tensordict with new tensors with expanded shapes.
        """
        _batch_size = torch.Size([*shape, *self.batch_size])
        d = dict()
        for key, value in self.items():
            if isinstance(value, _TensorDict):
                d[key] = value.expand(*shape)
            else:
                d[key] = value.expand(*shape, *value.shape)
        return TensorDict(source=d, batch_size=_batch_size, device=self._device_safe())

    def set(
        self,
        key: str,
        value: COMPATIBLE_TYPES,
        inplace: bool = False,
        _run_checks: bool = True,
        _meta_val: Optional[MetaTensor] = None,
    ) -> _TensorDict:
        """Sets a value in the TensorDict. If inplace=True (default is False),
        and if the key already exists, set will call set_ (in place setting).
        """
        if self.is_locked:
            raise RuntimeError("Cannot modify immutable TensorDict")
        if not isinstance(key, str):
            raise TypeError(f"Expected key to be a string but found {type(key)}")

        if self._is_shared is None:
            try:
                self._is_shared = value.is_shared()
            except NotImplementedError:
                # when running functorch, a NotImplementedError may be raised
                pass
        if self._is_memmap is None:
            self._is_memmap = isinstance(value, MemmapTensor)

        present = key in self._tensordict
        if present and value is self._tensordict[key]:
            return self

        if present and inplace:
            return self.set_(key, value)
        proc_value = self._process_tensor(
            value,
            check_tensor_shape=_run_checks,
            check_shared=False,
            check_device=_run_checks,
        )  # check_tensor_shape=_run_checks
        self._tensordict[key] = proc_value
        if _meta_val:
            self._dict_meta[key] = _meta_val
        elif present and key in self._dict_meta:
            del self._dict_meta[key]
        return self

    def del_(self, key: str) -> _TensorDict:
        del self._tensordict[key]
        if key in self._dict_meta:
            del self._dict_meta[key]
        return self

    def rename_key(self, old_key: str, new_key: str, safe: bool = False) -> _TensorDict:
        if not isinstance(old_key, str):
            raise TypeError(
                f"Expected old_name to be a string but found {type(old_key)}"
            )
        if not isinstance(new_key, str):
            raise TypeError(
                f"Expected new_name to be a string but found {type(new_key)}"
            )

        if safe and (new_key in self.keys()):
            raise KeyError(f"key {new_key} already present in TensorDict.")
        self.set(
            new_key,
            self.get(old_key),
            _meta_val=self._get_meta(old_key) if old_key in self._dict_meta else None,
            _run_checks=False,
        )
        self.del_(old_key)
        return self

    def set_(
        self, key: str, value: COMPATIBLE_TYPES, no_check: bool = False
    ) -> _TensorDict:
        if not no_check:
            if self.is_locked:
                raise RuntimeError("Cannot modify immutable TensorDict")
            if not isinstance(key, str):
                raise TypeError(f"Expected key to be a string but found {type(key)}")

        if no_check or key in self.keys():
            if not no_check:
                proc_value = self._process_tensor(
                    value, check_device=False, check_shared=False
                )
                # copy_ will broadcast one tensor onto another's shape, which we don't want
                target_shape = self._get_meta(key).shape
                if proc_value.shape != target_shape:
                    raise RuntimeError(
                        f'calling set_("{key}", tensor) with tensors of '
                        f"different shape: got tensor.shape={proc_value.shape} "
                        f'and get("{key}").shape={target_shape}'
                    )
            else:
                proc_value = value
            if proc_value is not self._tensordict[key]:
                self._tensordict[key].copy_(proc_value)
                if key in self._dict_meta:
                    self._dict_meta[key].requires_grad = proc_value.requires_grad
        else:
            raise AttributeError(
                f'key "{key}" not found in tensordict, '
                f'call td.set("{key}", value) for populating tensordict with '
                f"new key-value pair"
            )
        return self

    def _stack_onto_(
        self, key: str, list_item: List[COMPATIBLE_TYPES], dim: int
    ) -> TensorDict:
        torch.stack(list_item, dim=dim, out=self._tensordict[key])
        return self

    def _stack_onto_at_(
        self,
        key: str,
        list_item: List[COMPATIBLE_TYPES],
        dim: int,
        idx: INDEX_TYPING,
    ) -> TensorDict:
        if isinstance(idx, tuple) and len(idx) == 1:
            idx = idx[0]
        if isinstance(idx, (int, slice)) or (
            isinstance(idx, tuple)
            and all(isinstance(_idx, (int, slice)) for _idx in idx)
        ):
            torch.stack(list_item, dim=dim, out=self._tensordict[key][idx])
        else:
            raise ValueError(
                f"Cannot stack onto an indexed tensor with index {idx} "
                f"as its storage differs."
            )
        return self

    def set_at_(
        self, key: str, value: COMPATIBLE_TYPES, idx: INDEX_TYPING
    ) -> _TensorDict:
        if self.is_locked:
            raise RuntimeError("Cannot modify immutable TensorDict")
        if not isinstance(key, str):
            raise TypeError(f"Expected key to be a string but found {type(key)}")

        # do we need this?
        # value = self._process_tensor(
        #     value, check_tensor_shape=False, check_device=False
        # )
        if key not in self.keys():
            raise KeyError(f"did not find key {key} in {self.__class__.__name__}")
        tensor_in = self._tensordict[key]
        if isinstance(idx, tuple) and len(idx) and isinstance(idx[0], tuple):
            warn(
                "Multiple indexing can lead to unexpected behaviours when "
                "setting items, for instance `td[idx1][idx2] = other` may "
                "not write to the desired location if idx1 is a list/tensor."
            )
            tensor_in = _sub_index(tensor_in, idx)
            tensor_in.copy_(value)
        else:
            tensor_in[idx] = value
        if key in self._dict_meta:
            # change Meta in case of require_grad coming in value
            self._dict_meta[key].requires_grad = tensor_in.requires_grad
        return self

    def get(
        self, key: str, default: Union[str, COMPATIBLE_TYPES] = "_no_default_"
    ) -> COMPATIBLE_TYPES:
        if not isinstance(key, str):
            raise TypeError(f"Expected key to be a string but found {type(key)}")

        if key in self._tensordict.keys():
            return self._tensordict[key]
        else:
            return self._default_get(key, default)

    def share_memory_(self) -> _TensorDict:
        if self.is_memmap():
            raise RuntimeError(
                "memmap and shared memory are mutually exclusive features."
            )
        if not self._tensordict.keys():
            raise Exception(
                "share_memory_ must be called when the TensorDict is ("
                "partially) populated. Set a tensor first."
            )
        if self.device != torch.device("cpu"):
            # cuda tensors are shared by default
            return self
        for key, value in self.items():
            value.share_memory_()
        for key, value in self.items_meta():
            value.share_memory_()
        self._is_shared = True
        return self

    def detach_(self) -> _TensorDict:
        for key, value in self.items():
            value.detach_()
        return self

    def memmap_(self, prefix=None) -> _TensorDict:
        if self.is_shared() and self.device == torch.device("cpu"):
            raise RuntimeError(
                "memmap and shared memory are mutually exclusive features."
            )
        if not self._tensordict.keys():
            raise Exception(
                "memmap_() must be called when the TensorDict is (partially) "
                "populated. Set a tensor first."
            )
        if any(val.requires_grad for val in self._dict_meta.values()):
            raise Exception(
                "memmap is not compatible with gradients, one of Tensors has requires_grad equals True"
            )
        for key, value in self.items():
            self._tensordict[key] = MemmapTensor(value, prefix=prefix)
        for key, value in self.items_meta():
            value.memmap_()
        self._is_memmap = True
        return self

    def to(self, dest: Union[DEVICE_TYPING, torch.Size, Type], **kwargs) -> _TensorDict:
        if isinstance(dest, type) and issubclass(dest, _TensorDict):
            if isinstance(self, dest):
                return self
            td = dest(
                source=self,
                **kwargs,
            )
            return td
        elif isinstance(dest, (torch.device, str, int)):
            # must be device
            dest = torch.device(dest)
            if self._device_safe() is not None and dest == self.device:
                return self

            self_copy = TensorDict(
                source={key: value.to(dest) for key, value in self.items()},
                batch_size=self.batch_size,
                device=dest,
            )
            if self._safe:
                # sanity check
                self_copy._check_device()
                self_copy._check_is_shared()
            return self_copy
        elif isinstance(dest, torch.Size):
            self.batch_size = dest
            return self
        else:
            raise NotImplementedError(
                f"dest must be a string, torch.device or a TensorDict "
                f"instance, {dest} not allowed"
            )

    def masked_fill_(
        self, mask: torch.Tensor, value: Union[float, int, bool]
    ) -> _TensorDict:
        for key, item in self.items():
            mask_expand = expand_as_right(mask, item)
            item.masked_fill_(mask_expand, value)
        return self

    def masked_fill(self, mask: torch.Tensor, value: Union[float, bool]) -> _TensorDict:
        td_copy = self.clone()
        return td_copy.masked_fill_(mask, value)

    def is_contiguous(self) -> bool:
        return all([value.is_contiguous() for _, value in self.items()])

    def contiguous(self) -> _TensorDict:
        if not self.is_contiguous():
            return self.clone()
        return self

    def select(self, *keys: str, inplace: bool = False) -> _TensorDict:
        d = {key: value for (key, value) in self.items() if key in keys}
        d_meta = {
            key: value
            for (key, value) in self.items_meta(make_unset=False)
            if key in keys
        }
        if inplace:
            self._tensordict = d
            for key in list(self._dict_meta.keys()):
                if key not in keys:
                    del self._dict_meta[key]
            return self
        return TensorDict(
            device=self._device_safe(),
            batch_size=self.batch_size,
            source=d,
            _meta_source=d_meta,
            _run_checks=False,
            _is_memmap=self._is_memmap,
            _is_shared=self._is_shared,
        )

    def keys(self) -> KeysView:
        return self._tensordict.keys()


def implements_for_td(torch_function: Callable) -> Callable:
    """Register a torch function override for ScalarTensor"""

    @functools.wraps(torch_function)
    def decorator(func):
        TD_HANDLED_FUNCTIONS[torch_function] = func
        return func

    return decorator


# @implements_for_td(torch.testing.assert_allclose) TODO
def assert_allclose_td(
    actual: _TensorDict,
    expected: _TensorDict,
    rtol: float = None,
    atol: float = None,
    equal_nan: bool = True,
    msg: str = "",
) -> bool:
    if not isinstance(actual, _TensorDict) or not isinstance(expected, _TensorDict):
        raise TypeError("assert_allclose inputs must be of TensorDict type")
    set1 = set(actual.keys())
    set2 = set(expected.keys())
    if not (len(set1.difference(set2)) == 0 and len(set2) == len(set1)):
        raise KeyError(
            "actual and expected tensordict keys mismatch, "
            f"keys {(set1 - set2).union(set2 - set1)} appear in one but not "
            f"the other."
        )
    keys = sorted(list(actual.keys()))
    for key in keys:
        input1 = actual.get(key)
        input2 = expected.get(key)
        if isinstance(input1, _TensorDict):
            assert_allclose_td(input1, input2, rtol=rtol, atol=atol)
            continue

        mse = (input1.to(torch.float) - input2.to(torch.float)).pow(2).sum()
        mse = mse.div(input1.numel()).sqrt().item()

        default_msg = f"key {key} does not match, got mse = {mse:4.4f}"
        if len(msg):
            msg = "\t".join([default_msg, msg])
        else:
            msg = default_msg
        if isinstance(input1, MemmapTensor):
            input1 = input1._tensor
        if isinstance(input2, MemmapTensor):
            input2 = input2._tensor
        torch.testing.assert_allclose(
            input1, input2, rtol=rtol, atol=atol, equal_nan=equal_nan, msg=msg
        )
    return True


@implements_for_td(torch.unbind)
def unbind(td: _TensorDict, *args, **kwargs) -> Tuple[_TensorDict, ...]:
    return td.unbind(*args, **kwargs)


@implements_for_td(torch.clone)
def clone(td: _TensorDict, *args, **kwargs) -> _TensorDict:
    return td.clone(*args, **kwargs)


@implements_for_td(torch.squeeze)
def squeeze(td: _TensorDict, *args, **kwargs) -> _TensorDict:
    return td.squeeze(*args, **kwargs)


@implements_for_td(torch.unsqueeze)
def unsqueeze(td: _TensorDict, *args, **kwargs) -> _TensorDict:
    return td.unsqueeze(*args, **kwargs)


@implements_for_td(torch.masked_select)
def masked_select(td: _TensorDict, *args, **kwargs) -> _TensorDict:
    return td.masked_select(*args, **kwargs)


@implements_for_td(torch.permute)
def permute(td: _TensorDict, dims) -> _TensorDict:
    return td.permute(*dims)


@implements_for_td(torch.cat)
def cat(
    list_of_tensordicts: Sequence[_TensorDict],
    dim: int = 0,
    device: DEVICE_TYPING = None,
    out: _TensorDict = None,
) -> _TensorDict:
    if not list_of_tensordicts:
        raise RuntimeError("list_of_tensordicts cannot be empty")
    if not device:
        device = list_of_tensordicts[0].device
    if dim < 0:
        raise RuntimeError(
            f"negative dim in torch.dim(list_of_tensordicts, dim=dim) not "
            f"allowed, got dim={dim}"
        )

    batch_size = list(list_of_tensordicts[0].batch_size)
    if dim >= len(batch_size):
        raise RuntimeError(
            f"dim must be in the range 0 <= dim < len(batch_size), got dim"
            f"={dim} and batch_size={batch_size}"
        )
    batch_size[dim] = sum([td.batch_size[dim] for td in list_of_tensordicts])
    batch_size = torch.Size(batch_size)

    # check that all tensordict match
    keys = _check_keys(list_of_tensordicts, strict=True)
    if out is None:
        out = dict()
        for key in keys:
            out[key] = torch.cat([td.get(key) for td in list_of_tensordicts], dim)
        out = TensorDict(out, device=device, batch_size=batch_size, _run_checks=False)
        return out
    else:
        if out.batch_size != batch_size:
            raise RuntimeError(
                "out.batch_size and cat batch size must match, "
                f"got out.batch_size={out.batch_size} and batch_size"
                f"={batch_size}"
            )

        for key in keys:
            out.set_(key, torch.cat([td.get(key) for td in list_of_tensordicts], dim))
        return out


@implements_for_td(torch.stack)
def stack(
    list_of_tensordicts: Sequence[_TensorDict],
    dim: int = 0,
    out: _TensorDict = None,
    strict=False,
    contiguous=False,
) -> _TensorDict:
    if not list_of_tensordicts:
        raise RuntimeError("list_of_tensordicts cannot be empty")
    batch_size = list_of_tensordicts[0].batch_size
    if dim < 0:
        dim = len(batch_size) + dim + 1
    if len(list_of_tensordicts) > 1:
        for td in list_of_tensordicts[1:]:
            if td.batch_size != list_of_tensordicts[0].batch_size:
                raise RuntimeError(
                    "stacking tensordicts requires them to have congruent "
                    "batch sizes, got td1.batch_size={td.batch_size} and "
                    f"td2.batch_size{list_of_tensordicts[0].batch_size}"
                )
    # check that all tensordict match
    keys = _check_keys(list_of_tensordicts)

    if out is None:
        device = list_of_tensordicts[0]._device_safe()
        out = (
            TensorDict(
                {
                    key: torch.stack(
                        [_tensordict.get(key) for _tensordict in list_of_tensordicts],
                        dim,
                    )
                    for key in keys
                },
                batch_size=LazyStackedTensorDict._compute_batch_size(
                    batch_size, dim, len(list_of_tensordicts)
                ),
                device=device,
                _run_checks=False,
            )
            if contiguous
            else LazyStackedTensorDict(
                *list_of_tensordicts,
                stack_dim=dim,
            )
        )
    else:
        batch_size = list(batch_size)
        batch_size.insert(dim, len(list_of_tensordicts))
        batch_size = torch.Size(batch_size)

        if out.batch_size != batch_size:
            raise RuntimeError(
                "out.batch_size and stacked batch size must match, "
                f"got out.batch_size={out.batch_size} and batch_size"
                f"={batch_size}"
            )

        out_keys = set(out.keys())
        if strict:
            in_keys = set(keys)
            if len(out_keys - in_keys) > 0:
                raise RuntimeError(
                    "The output tensordict has keys that are missing in the "
                    "tensordict that has to be written: {out_keys - in_keys}. "
                    "As per the call to `stack(..., strict=True)`, this "
                    "is not permitted."
                )
            elif len(in_keys - out_keys) > 0:
                raise RuntimeError(
                    "The resulting tensordict has keys that are missing in "
                    f"its destination: {in_keys - out_keys}. As per the call "
                    "to `stack(..., strict=True)`, this is not permitted."
                )

        for key in keys:
            if key in out_keys:
                out._stack_onto_(
                    key,
                    [_tensordict.get(key) for _tensordict in list_of_tensordicts],
                    dim,
                )
            else:
                out.set(
                    key,
                    torch.stack(
                        [_tensordict.get(key) for _tensordict in list_of_tensordicts],
                        dim,
                    ),
                    inplace=True,
                )
    return out


def pad(tensordict: _TensorDict, pad_size: Sequence[int], value: float = 0.0):
    """Pads all tensors in a tensordict along the batch dimensions with a constant value,
    returning a new tensordict

    Args:
         tensordict (TensorDict): The tensordict to pad
         pad_size (Sequence[int]): The padding size by which to pad some batch
            dimensions of the tensordict, starting from the first dimension and
            moving forward. [len(pad_size) / 2] dimensions of the batch size will
            be padded. For example to pad only the first dimension, pad has the form
            (padding_left, padding_right). To pad two dimensions,
            (padding_left, padding_right, padding_top, padding_bottom) and so on.
            pad_size must be even and less than or equal to twice the number of batch dimensions.
         value (float, optional): The fill value to pad by, default 0.0

    Returns:
        A new TensorDict padded along the batch dimensions

    Examples:
        >>> from torchrl.data import TensorDict, pad
        >>> import torch
        >>> td = TensorDict({'a': torch.ones(3, 4, 1),
        ...     'b': torch.ones(3, 4, 1, 1)}, batch_size=[3, 4])
        >>> dim0_left, dim0_right, dim1_left, dim1_right = [0, 1, 0, 2]
        >>> padded_td = pad(td, [dim0_left, dim0_right, dim1_left, dim1_right], value=0.0)
        >>> print(padded_td.batch_size)
        torch.Size([4, 6])
        >>> print(padded_td.get("a").shape)
        torch.Size([4, 6, 1])
        >>> print(padded_td.get("b").shape)
        torch.Size([4, 6, 1, 1])
    """

    if len(pad_size) > 2 * len(tensordict.batch_size):
        raise RuntimeError(
            "The length of pad_size must be <= 2 * the number of batch dimensions"
        )

    if len(pad_size) % 2:
        raise RuntimeError("pad_size must have an even number of dimensions")

    new_batch_size = list(tensordict.batch_size)
    for i in range(len(pad_size)):
        new_batch_size[i // 2] += pad_size[i]

    reverse_pad = pad_size[::-1]
    for i in range(0, len(reverse_pad), 2):
        reverse_pad[i], reverse_pad[i + 1] = reverse_pad[i + 1], reverse_pad[i]

    out = TensorDict({}, new_batch_size, device=tensordict.device)
    for key, tensor in tensordict.items():
        cur_pad = reverse_pad
        if len(pad_size) < len(tensor.shape) * 2:
            cur_pad = [0] * (len(tensor.shape) * 2 - len(pad_size)) + reverse_pad

        if isinstance(tensor, _TensorDict):
            padded = pad(tensor, pad_size, value)
        else:
            padded = torch.nn.functional.pad(tensor, cur_pad, value=value)
        out.set(key, padded)

    return out


# @implements_for_td(torch.nn.utils.rnn.pad_sequence)
def pad_sequence_td(
    list_of_tensordicts: Sequence[_TensorDict],
    batch_first: bool = True,
    padding_value: float = 0.0,
    out: _TensorDict = None,
    device: Optional[DEVICE_TYPING] = None,
):
    if not list_of_tensordicts:
        raise RuntimeError("list_of_tensordicts cannot be empty")
    # check that all tensordict match
    keys = _check_keys(list_of_tensordicts)
    if out is None:
        out = TensorDict({}, [], device=device)
        for key in keys:
            out.set(
                key,
                torch.nn.utils.rnn.pad_sequence(
                    [td.get(key) for td in list_of_tensordicts],
                    batch_first=batch_first,
                    padding_value=padding_value,
                ),
            )
        return out
    else:
        for key in keys:
            out.set_(
                key,
                torch.nn.utils.rnn.pad_sequence(
                    [td.get(key) for td in list_of_tensordicts],
                    batch_first=batch_first,
                    padding_value=padding_value,
                ),
            )
        return out


class SubTensorDict(_TensorDict):
    """
    A TensorDict that only sees an index of the stored tensors.

    By default, indexing a tensordict with an iterable will result in a
    SubTensorDict. This is done such that a TensorDict indexed with
    non-contiguous index (e.g. a Tensor) will still point to the original
    memory location (unlike regular indexing of tensors).

    Examples:
        >>> from torchrl.data import TensorDict, SubTensorDict
        >>> source = {'random': torch.randn(3, 4, 5, 6),
        ...    'zeros': torch.zeros(3, 4, 1, dtype=torch.bool)}
        >>> batch_size = torch.Size([3, 4])
        >>> td = TensorDict(source, batch_size)
        >>> td_index = td[:, 2]
        >>> print(type(td_index), td_index.shape)
        <class 'torchrl.data.tensordict.tensordict.TensorDict'> \
torch.Size([3])
        >>> td_index = td[:, slice(None)]
        >>> print(type(td_index), td_index.shape)
        <class 'torchrl.data.tensordict.tensordict.TensorDict'> \
torch.Size([3, 4])
        >>> td_index = td[:, torch.Tensor([0, 2]).to(torch.long)]
        >>> print(type(td_index), td_index.shape)
        <class 'torchrl.data.tensordict.tensordict.SubTensorDict'> \
torch.Size([3, 2])
        >>> _ = td_index.fill_('zeros', 1)
        >>> # the indexed tensors are updated with Trues
        >>> print(td.get('zeros'))
        tensor([[[ True],
                 [False],
                 [ True],
                 [False]],
        <BLANKLINE>
                [[ True],
                 [False],
                 [ True],
                 [False]],
        <BLANKLINE>
                [[ True],
                 [False],
                 [ True],
                 [False]]])

    """

    _safe = False
    _lazy = True

    def __init__(
        self,
        source: _TensorDict,
        idx: INDEX_TYPING,
        batch_size: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        self._is_shared = None
        self._is_memmap = None

        if not isinstance(source, _TensorDict):
            raise TypeError(
                f"Expected source to be a subclass of _TensorDict, "
                f"got {type(source)}"
            )
        self._source = source
        if not isinstance(idx, (tuple, list)):
            idx = (idx,)
        else:
            idx = tuple(idx)
        self.idx = idx
        self._batch_size = _getitem_batch_size(self._source.batch_size, self.idx)
        if batch_size is not None and batch_size != self.batch_size:
            raise RuntimeError("batch_size does not match self.batch_size.")

    def _make_meta(self, key: str) -> MetaTensor:
        out = self._source._get_meta(key)[self.idx]
        return out

    @property
    def batch_size(self) -> torch.Size:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, new_size: torch.Size):
        return self._batch_size_setter(new_size)

    @property
    def device(self) -> torch.device:
        return self._source.device

    @device.setter
    def device(self, value: DEVICE_TYPING) -> None:
        self._source.device = value

    def _device_safe(self) -> Union[None, torch.device]:
        return self._source._device_safe()

    def _preallocate(self, key: str, value: COMPATIBLE_TYPES) -> _TensorDict:
        return self._source.set(key, value)

    def set(
        self,
        key: str,
        tensor: COMPATIBLE_TYPES,
        inplace: bool = False,
        _run_checks: bool = True,
    ) -> _TensorDict:
        if self.is_locked:
            raise RuntimeError("Cannot modify immutable TensorDict")
        keys = set(self.keys())
        if isinstance(tensor, _TensorDict) and tensor.batch_size != self.batch_size:
            tensor.batch_size = self.batch_size
        if inplace and key in keys:
            return self.set_(key, tensor)
        elif key in keys:
            raise RuntimeError(
                "Calling `SubTensorDict.set(key, value, inplace=False)` is prohibited for existing tensors. "
                "Consider calling `SubTensorDict.set_(...)` or cloning your tensordict first."
            )

        tensor = self._process_tensor(
            tensor, check_device=False, check_tensor_shape=False
        )
        parent = self.get_parent_tensordict()

        if isinstance(tensor, _TensorDict):
            tensor_expand = TensorDict(
                {
                    key: _expand_to_match_shape(
                        parent.batch_size, _tensor, self.batch_dims, self.device
                    )
                    for key, _tensor in tensor.items()
                },
                parent.batch_size,
                _run_checks=False,
            )
        else:
            tensor_expand = torch.zeros(
                *parent.batch_size,
                *tensor.shape[self.batch_dims :],
                dtype=tensor.dtype,
                device=self.device,
            )
            if self.is_shared() and self.device == torch.device("cpu"):
                tensor_expand.share_memory_()
            elif self.is_memmap():
                tensor_expand = MemmapTensor(tensor_expand)
        parent.set(key, tensor_expand, _run_checks=_run_checks)
        self.set_(key, tensor)
        if key in self._dict_meta:
            self._dict_meta[key].requires_grad = tensor.requires_grad
        return self

    def keys(self) -> KeysView:
        return self._source.keys()

    def set_(
        self, key: str, tensor: COMPATIBLE_TYPES, no_check: bool = False
    ) -> SubTensorDict:
        if not no_check:
            if self.is_locked:
                raise RuntimeError("Cannot modify immutable TensorDict")
            if key not in self.keys():
                raise KeyError(f"key {key} not found in {self.keys()}")
            if tensor.shape[: self.batch_dims] != self.batch_size:
                raise RuntimeError(
                    f"tensor.shape={tensor.shape[:self.batch_dims]} and "
                    f"self.batch_size={self.batch_size} mismatch"
                )
        self._source.set_at_(key, tensor, self.idx)
        if key in self._dict_meta:
            self._dict_meta[key].requires_grad = tensor.requires_grad

        return self

    def _stack_onto_(
        self, key: str, list_item: List[COMPATIBLE_TYPES], dim: int
    ) -> TensorDict:
        self._source._stack_onto_at_(key, list_item, dim=dim, idx=self.idx)
        return self

    def to(self, dest: Union[DEVICE_TYPING, torch.Size, Type], **kwargs) -> _TensorDict:
        if isinstance(dest, type) and issubclass(dest, _TensorDict):
            if isinstance(self, dest):
                return self
            return dest(
                source=self.clone(),
            )
        elif isinstance(dest, (torch.device, str, int)):
            dest = torch.device(dest)
            try:
                if dest == self.device:
                    return self
            except RuntimeError:
                # if device has not been set, pass
                pass
            td = self.to(TensorDict)
            # must be device
            return td.to(dest, **kwargs)
        elif isinstance(dest, torch.Size):
            self.batch_size = dest
            return self
        else:
            raise NotImplementedError(
                f"dest must be a string, torch.device or a TensorDict "
                f"instance, {dest} not allowed"
            )

    def _change_batch_size(self, new_size: torch.Size):
        if not hasattr(self, "_orig_batch_size"):
            self._orig_batch_size = self.batch_size
        elif self._orig_batch_size == new_size:
            del self._orig_batch_size
        self._batch_size = new_size

    def get(
        self,
        key: str,
        default: Optional[Union[torch.Tensor, str]] = "_no_default_",
    ) -> COMPATIBLE_TYPES:
        return self._source.get_at(key, self.idx, default=default)

    def set_at_(
        self,
        key: str,
        value: COMPATIBLE_TYPES,
        idx: INDEX_TYPING,
        discard_idx_attr: bool = False,
    ) -> SubTensorDict:
        if self.is_locked:
            raise RuntimeError("Cannot modify immutable TensorDict")
        if not isinstance(idx, tuple):
            idx = (idx,)
        if discard_idx_attr:
            self._source.set_at_(key, value, idx)
        else:
            tensor = self._source.get_at(key, self.idx)
            tensor[idx] = value
            self._source.set_at_(key, tensor, self.idx)
        if key in self._dict_meta:
            self._dict_meta[key].requires_grad = value.requires_grad
        return self

    def get_at(
        self,
        key: str,
        idx: INDEX_TYPING,
        discard_idx_attr: bool = False,
        default: Optional[Union[torch.Tensor, str]] = "_no_default_",
    ) -> COMPATIBLE_TYPES:
        if not isinstance(idx, tuple):
            idx = (idx,)
        if discard_idx_attr:
            return self._source.get_at(key, idx, default=default)
        else:
            out = self._source.get_at(key, self.idx, default=default)
            if out is default:
                return out
            return out[idx]

    def update_(
        self,
        input_dict: Union[Dict[str, COMPATIBLE_TYPES], _TensorDict],
        clone: bool = False,
    ) -> SubTensorDict:
        return self.update_at_(
            input_dict, idx=self.idx, discard_idx_attr=True, clone=clone
        )

    def update_at_(
        self,
        input_dict: Union[Dict[str, COMPATIBLE_TYPES], _TensorDict],
        idx: INDEX_TYPING,
        discard_idx_attr: bool = False,
        clone: bool = False,
    ) -> SubTensorDict:
        for key, value in input_dict.items():
            if not isinstance(value, _accepted_classes):
                raise TypeError(
                    f"Expected value to be one of types {_accepted_classes} "
                    f"but got {type(value)}"
                )
            if clone:
                value = value.clone()
            self.set_at_(
                key,
                value,
                idx,
                discard_idx_attr=discard_idx_attr,
            )
        return self

    def get_parent_tensordict(self) -> _TensorDict:
        if not isinstance(self._source, _TensorDict):
            raise TypeError(
                f"SubTensorDict was initialized with a source of type"
                f" {self._source.__class__.__name__}, "
                "parent tensordict not accessible"
            )
        return self._source

    def del_(self, key: str) -> _TensorDict:
        self._source = self._source.del_(key)
        return self

    def clone(self, recursive: bool = True) -> SubTensorDict:
        if not recursive:
            return copy(self)
        return SubTensorDict(
            source=self._source,
            idx=self.idx,
        )

    def is_contiguous(self) -> bool:
        return all([value.is_contiguous() for _, value in self.items()])

    def contiguous(self) -> _TensorDict:
        if self.is_contiguous():
            return self
        return TensorDict(
            batch_size=self.batch_size,
            source={key: value for key, value in self.items()},
            device=self._device_safe(),
        )

    def select(self, *keys: str, inplace: bool = False) -> _TensorDict:
        if inplace:
            self._source = self._source.select(*keys)
            return self
        return self._source.select(*keys)[self.idx]

    def expand(self, *shape: int, inplace: bool = False) -> _TensorDict:
        new_source = self._source.expand(*shape)
        idx = tuple(slice(None) for _ in shape) + tuple(self.idx)
        if inplace:
            self._source = new_source
            self.idx = idx
        return new_source[idx]

    def is_shared(self, no_check: bool = True) -> bool:
        return self._source.is_shared(no_check=no_check)

    def is_memmap(self, no_check: bool = True) -> bool:
        return self._source.is_memmap(no_check=no_check)

    def rename_key(
        self, old_key: str, new_key: str, safe: bool = False
    ) -> SubTensorDict:
        self._source.rename_key(old_key, new_key, safe=safe)
        return self

    def pin_memory(self) -> _TensorDict:
        self._source.pin_memory()
        return self

    def detach_(self) -> _TensorDict:
        raise RuntimeError("Detaching a sub-tensordict in-place cannot be done.")

    def masked_fill_(
        self, mask: torch.Tensor, value: Union[float, bool]
    ) -> _TensorDict:
        for key, item in self.items():
            self.set_(key, torch.full_like(item, value))
        return self

    def masked_fill(self, mask: torch.Tensor, value: Union[float, bool]) -> _TensorDict:
        td_copy = self.clone()
        return td_copy.masked_fill_(mask, value)

    def memmap_(self, prefix=None) -> _TensorDict:
        raise RuntimeError(
            "Converting a sub-tensordict values to memmap cannot be done."
        )

    def share_memory_(self) -> _TensorDict:
        raise RuntimeError(
            "Casting a sub-tensordict values to shared memory cannot be done."
        )


def merge_tensordicts(*tensordicts: _TensorDict) -> _TensorDict:
    if len(tensordicts) < 2:
        raise RuntimeError(
            f"at least 2 tensordicts must be provided, got" f" {len(tensordicts)}"
        )
    d = tensordicts[0].to_dict()
    for td in tensordicts[1:]:
        d.update(td.to_dict())
    return TensorDict({}, [], device=td._device_safe()).update(d)


class LazyStackedTensorDict(_TensorDict):
    """A Lazy stack of TensorDicts.

    When stacking TensorDicts together, the default behaviour is to put them
    in a stack that is not instantiated.
    This allows to seamlessly work with stacks of tensordicts with operations
    that will affect the original tensordicts.

    Args:
         *tensordicts (TensorDict instances): a list of tensordict with
            same batch size.
         stack_dim (int): a dimension (between `-td.ndimension()` and
            `td.ndimension()-1` along which the stack should be performed.

    Examples:
        >>> from torchrl.data import TensorDict
        >>> import torch
        >>> tds = [TensorDict({'a': torch.randn(3, 4)}, batch_size=[3])
        ...     for _ in range(10)]
        >>> td_stack = torch.stack(tds, -1)
        >>> print(td_stack.shape)
        torch.Size([3, 10])
        >>> print(td_stack.get("a").shape)
        torch.Size([3, 10, 4])
        >>> print(td_stack[:, 0] is tds[0])
        True
    """

    _safe = False
    _lazy = True

    def __init__(
        self,
        *tensordicts: _TensorDict,
        stack_dim: int = 0,
        batch_size: Optional[Sequence[int]] = None,  # TODO: remove
    ):
        super().__init__()

        self._is_shared = None
        self._is_memmap = None

        # sanity check
        N = len(tensordicts)
        if not N:
            raise RuntimeError(
                "at least one tensordict must be provided to "
                "StackedTensorDict to be instantiated"
            )
        if not isinstance(tensordicts[0], _TensorDict):
            raise TypeError(
                f"Expected input to be _TensorDict instance"
                f" but got {type(tensordicts[0])} instead."
            )
        if stack_dim < 0:
            raise RuntimeError(
                f"stack_dim must be non negative, got stack_dim={stack_dim}"
            )
        _batch_size = tensordicts[0].batch_size
        device = tensordicts[0]._device_safe()

        for i, td in enumerate(tensordicts[1:]):
            if not isinstance(td, _TensorDict):
                raise TypeError(
                    f"Expected input to be _TensorDict instance"
                    f" but got {type(tensordicts[0])} instead."
                )
            _bs = td.batch_size
            _device = td._device_safe()
            if device != _device:
                raise RuntimeError(f"devices differ, got {device} and {_device}")
            if _bs != _batch_size:
                raise RuntimeError(
                    f"batch sizes in tensordicts differs, StackedTensorDict "
                    f"cannot be created. Got td[0].batch_size={_batch_size} "
                    f"and td[i].batch_size={_bs} "
                )
        self.tensordicts: List[_TensorDict] = list(tensordicts)
        self.stack_dim = stack_dim
        self._batch_size = self._compute_batch_size(_batch_size, stack_dim, N)
        self._update_valid_keys()
        if batch_size is not None and batch_size != self.batch_size:
            raise RuntimeError("batch_size does not match self.batch_size.")

    @property
    def device(self) -> torch.device:
        # devices might have changed so we check that they're all the same
        device_set = {td.device for td in self.tensordicts}
        if len(device_set) != 1:
            raise RuntimeError(
                f"found multiple devices in {self.__class__.__name__}:" f" {device_set}"
            )
        return self.tensordicts[0].device

    @device.setter
    def device(self, value: DEVICE_TYPING) -> None:
        for t in self.tensordicts:
            t.device = value

    def _device_safe(self) -> Union[None, torch.device]:
        return self.tensordicts[0]._device_safe()

    @property
    def batch_size(self) -> torch.Size:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, new_size: torch.Size):
        return self._batch_size_setter(new_size)

    def is_shared(self, no_check: bool = True) -> bool:
        are_shared = [td.is_shared(no_check=no_check) for td in self.tensordicts]
        if any(are_shared) and not all(are_shared):
            raise RuntimeError(
                f"tensordicts shared status mismatch, got {sum(are_shared)} "
                f"shared tensordicts and "
                f"{len(are_shared) - sum(are_shared)} non shared tensordict "
            )
        return all(are_shared)

    def is_memmap(self, no_check: bool = True) -> bool:
        are_memmap = [td.is_memmap() for td in self.tensordicts]
        if any(are_memmap) and not all(are_memmap):
            raise RuntimeError(
                f"tensordicts memmap status mismatch, got {sum(are_memmap)} "
                f"memmap tensordicts and "
                f"{len(are_memmap) - sum(are_memmap)} non memmap tensordict "
            )
        return all(are_memmap)

    def get_valid_keys(self) -> Set[str]:
        if self._valid_keys is None:
            self._update_valid_keys()
        return self._valid_keys

    def set_valid_keys(self, keys: Sequence[str]) -> None:
        raise RuntimeError(
            "setting valid keys is not permitted. valid keys are defined as "
            "the intersection of all the key sets from the TensorDicts in a "
            "stack and cannot be defined explicitely."
        )

    valid_keys = property(get_valid_keys, set_valid_keys)

    @staticmethod
    def _compute_batch_size(
        batch_size: torch.Size, stack_dim: int, N: int
    ) -> torch.Size:
        s = list(batch_size)
        s.insert(stack_dim, N)
        return torch.Size(s)

    def set(self, key: str, tensor: COMPATIBLE_TYPES, **kwargs) -> _TensorDict:
        if self.is_locked:
            raise RuntimeError("Cannot modify immutable TensorDict")
        if isinstance(tensor, _TensorDict):
            if tensor.batch_size[: self.batch_dims] != self.batch_size:
                tensor.batch_size = self.clone(recursive=False).batch_size
        if self.batch_size != tensor.shape[: self.batch_dims]:
            raise RuntimeError(
                "Setting tensor to tensordict failed because the shapes "
                f"mismatch: got tensor.shape = {tensor.shape} and "
                f"tensordict.batch_size={self.batch_size}"
            )
        proc_tensor = self._process_tensor(
            tensor, check_device=False, check_tensor_shape=False
        )
        proc_tensor = proc_tensor.unbind(self.stack_dim)
        for td, _item in zip(self.tensordicts, proc_tensor):
            td.set(key, _item, **kwargs)
        if key not in self._valid_keys:
            self._valid_keys = sorted([*self._valid_keys, key])
        if key in self._dict_meta:
            del self._dict_meta[key]
        return self

    def set_(
        self, key: str, tensor: COMPATIBLE_TYPES, no_check: bool = False
    ) -> _TensorDict:
        if not no_check:
            if self.is_locked:
                raise RuntimeError("Cannot modify immutable TensorDict")
            if isinstance(tensor, _TensorDict):
                if tensor.batch_size[: self.batch_dims] != self.batch_size:
                    tensor.batch_size = self.clone(recursive=False).batch_size
            if self.batch_size != tensor.shape[: self.batch_dims]:
                raise RuntimeError(
                    "Setting tensor to tensordict failed because the shapes "
                    f"mismatch: got tensor.shape = {tensor.shape} and "
                    f"tensordict.batch_size={self.batch_size}"
                )
            if key not in self.valid_keys:
                raise KeyError(
                    "setting a value in-place on a stack of TensorDict is only "
                    "permitted if all members of the stack have this key in "
                    "their register."
                )
            tensor = self._process_tensor(
                tensor,
                check_device=False,
                check_tensor_shape=False,
                check_shared=False,
            )
        if key in self._dict_meta:
            self._dict_meta[key].requires_grad = tensor.requires_grad
        tensors = tensor.unbind(self.stack_dim)
        for td, _item in zip(self.tensordicts, tensors):
            td.set_(key, _item)
        return self

    def set_at_(
        self, key: str, value: COMPATIBLE_TYPES, idx: INDEX_TYPING
    ) -> _TensorDict:
        if self.is_locked:
            raise RuntimeError("Cannot modify immutable TensorDict")
        sub_td = self[idx]
        sub_td.set_(key, value)
        return self

    def _stack_onto_(
        self,
        key: str,
        list_item: List[COMPATIBLE_TYPES],
        dim: int,
    ) -> _TensorDict:
        if dim == self.stack_dim:
            for source, tensordict_dest in zip(list_item, self.tensordicts):
                tensordict_dest.set_(key, source)
        else:
            # we must stack and unbind, there is no way to make it more efficient
            self.set_(key, torch.stack(list_item, dim))
        return self

    def get(
        self,
        key: str,
        default: Union[str, COMPATIBLE_TYPES] = "_no_default_",
    ) -> COMPATIBLE_TYPES:
        if not (key in self.valid_keys):
            # first, let's try to update the valid keys
            self._update_valid_keys()

        if not (key in self.valid_keys):
            return self._default_get(key, default)
        tensors = [td.get(key, default=default) for td in self.tensordicts]
        shapes = set(tensor.shape for tensor in tensors)
        if len(shapes) != 1:
            raise RuntimeError(
                f"found more than one unique shape in the tensors to be "
                f"stacked ({shapes}). This is likely due to a modification "
                f"of one of the stacked TensorDicts, where a key has been "
                f"updated/created with an uncompatible shape."
            )
        return torch.stack(tensors, self.stack_dim)

    def _make_meta(self, key: str) -> MetaTensor:
        return torch.stack(
            [td._get_meta(key) for td in self.tensordicts], self.stack_dim
        )

    def is_contiguous(self) -> bool:
        return False

    def contiguous(self) -> _TensorDict:
        source = {key: value for key, value in self.items()}
        batch_size = self.batch_size
        device = self._device_safe()
        out = TensorDict(
            source=source,
            batch_size=batch_size,
            # we could probably just infer the items_meta by extending them
            # _meta_source=meta_source,
            device=device,
        )
        return out

    def clone(self, recursive: bool = True) -> _TensorDict:
        if recursive:
            return LazyStackedTensorDict(
                *[td.clone() for td in self.tensordicts],
                stack_dim=self.stack_dim,
            )
        return LazyStackedTensorDict(
            *[td for td in self.tensordicts], stack_dim=self.stack_dim
        )

    def pin_memory(self) -> _TensorDict:
        for td in self.tensordicts:
            td.pin_memory()
        return self

    def to(self, dest: Union[DEVICE_TYPING, Type], **kwargs) -> _TensorDict:
        if isinstance(dest, type) and issubclass(dest, _TensorDict):
            if isinstance(self, dest):
                return self
            kwargs.update({"batch_size": self.batch_size})
            return dest(source=self, **kwargs)
        elif isinstance(dest, (torch.device, str, int)):
            dest = torch.device(dest)
            try:
                if dest == self.device:
                    return self
            except RuntimeError:
                pass
            tds = [td.to(dest) for td in self.tensordicts]
            return LazyStackedTensorDict(*tds, stack_dim=self.stack_dim)
        elif isinstance(dest, torch.Size):
            self.batch_size = dest
        else:
            raise NotImplementedError(
                f"dest must be a string, torch.device or a TensorDict "
                f"instance, {dest} not allowed"
            )

    def _check_new_batch_size(self, new_size: torch.Size):
        if len(new_size) <= self.stack_dim:
            raise RuntimeError(
                "Changing the batch_size of a LazyStackedTensorDicts can only "
                "be done with sizes that are at least as long as the "
                "stacking dimension."
            )
        super()._check_new_batch_size(new_size)

    def _change_batch_size(self, new_size: torch.Size):
        if not hasattr(self, "_orig_batch_size"):
            self._orig_batch_size = self.batch_size
        elif self._orig_batch_size == new_size:
            del self._orig_batch_size
        self._batch_size = new_size

    def keys(self) -> Iterator[str]:
        for key in self.valid_keys:
            yield key

    def _update_valid_keys(self) -> None:
        valid_keys = set(self.tensordicts[0].keys())
        for td in self.tensordicts[1:]:
            valid_keys = valid_keys.intersection(td.keys())
        self._valid_keys = sorted(list(valid_keys))

    def select(self, *keys: str, inplace: bool = False) -> _TensorDict:
        # if len(set(self.valid_keys).intersection(keys)) != len(keys):
        #     raise KeyError(
        #         f"Selected and existing keys mismatch, got self.valid_keys"
        #         f"={self.valid_keys} and keys={keys}"
        #     )
        tensordicts = [td.select(*keys, inplace=inplace) for td in self.tensordicts]
        if inplace:
            return self
        return LazyStackedTensorDict(
            *tensordicts,
            stack_dim=self.stack_dim,
        )

    def __setitem__(self, item: INDEX_TYPING, value: _TensorDict) -> _TensorDict:
        if isinstance(item, list):
            item = torch.tensor(item, device=self.device)
        if isinstance(item, tuple) and any(
            isinstance(sub_index, list) for sub_index in item
        ):
            item = tuple(
                torch.tensor(sub_index, device=self.device)
                if isinstance(sub_index, list)
                else sub_index
                for sub_index in item
            )
        if (isinstance(item, torch.Tensor) and item.dtype is torch.bool) or (
            isinstance(item, tuple)
            and any(
                isinstance(_item, torch.Tensor) and _item.dtype is torch.bool
                for _item in item
            )
        ):
            raise RuntimeError(
                "setting values to a LazyStackTensorDict using boolean values is not supported yet."
                "If this feature is needed, feel free to raise an issue on github."
            )
        return super().__setitem__(item, value)

    def __getitem__(self, item: INDEX_TYPING) -> _TensorDict:
        if item is Ellipsis or (isinstance(item, tuple) and Ellipsis in item):
            item = convert_ellipsis_to_idx(item, self.batch_size)
        if isinstance(item, tuple) and sum(
            isinstance(_item, str) for _item in item
        ) not in [len(item), 0]:
            raise IndexError(_STR_MIXED_INDEX_ERROR)
        if isinstance(item, list):
            item = torch.tensor(item, device=self.device)
        if isinstance(item, tuple) and any(
            isinstance(sub_index, list) for sub_index in item
        ):
            item = tuple(
                torch.tensor(sub_index, device=self.device)
                if isinstance(sub_index, list)
                else sub_index
                for sub_index in item
            )
        if isinstance(item, str):
            return self.get(item)
        elif isinstance(item, tuple) and all(
            isinstance(sub_item, str) for sub_item in item
        ):
            out = self.get(item[0])
            if len(item) > 1:
                return out[item[1:]]
            else:
                return out
        elif isinstance(item, torch.Tensor) and item.dtype == torch.bool:
            return self.masked_select(item)
        elif (
            isinstance(item, (Number,))
            or (isinstance(item, torch.Tensor) and item.ndimension() == 0)
        ) and self.stack_dim == 0:
            return self.tensordicts[item]
        elif isinstance(item, (torch.Tensor, list)) and self.stack_dim == 0:
            out = LazyStackedTensorDict(
                *[self.tensordicts[_item] for _item in item],
                stack_dim=self.stack_dim,
            )
            return out
        elif isinstance(item, (torch.Tensor, list)) and self.stack_dim != 0:
            out = LazyStackedTensorDict(
                *[tensordict[item] for tensordict in self.tensordicts],
                stack_dim=self.stack_dim,
            )
            return out
        elif isinstance(item, slice) and self.stack_dim == 0:
            return LazyStackedTensorDict(
                *self.tensordicts[item], stack_dim=self.stack_dim
            )
        elif isinstance(item, slice) and self.stack_dim != 0:
            return LazyStackedTensorDict(
                *[tensordict[item] for tensordict in self.tensordicts],
                stack_dim=self.stack_dim,
            )
        elif isinstance(item, (slice, Number)):
            new_stack_dim = (
                self.stack_dim - 1 if isinstance(item, Number) else self.stack_dim
            )
            return LazyStackedTensorDict(
                *[td[item] for td in self.tensordicts],
                stack_dim=new_stack_dim,
            )
        elif isinstance(item, tuple):
            _sub_item = tuple(
                _item for i, _item in enumerate(item) if i == self.stack_dim
            )
            if len(_sub_item):
                tensordicts = self.tensordicts[_sub_item[0]]
                if isinstance(tensordicts, _TensorDict):
                    return tensordicts
            else:
                tensordicts = self.tensordicts
            # select sub tensordicts
            _sub_item = tuple(
                _item for i, _item in enumerate(item) if i != self.stack_dim
            )
            if len(_sub_item):
                tensordicts = [td[_sub_item] for td in tensordicts]
            new_stack_dim = self.stack_dim - sum(
                [isinstance(_item, Number) for _item in item[: self.stack_dim]]
            )
            return torch.stack(list(tensordicts), dim=new_stack_dim)
        else:
            raise NotImplementedError(
                f"selecting StackedTensorDicts with type "
                f"{item.__class__.__name__} is not supported yet"
            )

    def del_(self, key: str, **kwargs) -> _TensorDict:
        for td in self.tensordicts:
            td.del_(key, **kwargs)
        self._valid_keys.remove(key)
        return self

    def share_memory_(self) -> _TensorDict:
        for td in self.tensordicts:
            td.share_memory_()
        self._is_shared = True
        return self

    def detach_(self) -> _TensorDict:
        for td in self.tensordicts:
            td.detach_()
        return self

    def memmap_(self, prefix=None) -> _TensorDict:
        for td in self.tensordicts:
            td.memmap_(prefix=prefix)
        self._is_memmap = True
        return self

    def expand(self, *shape: int, inplace: bool = False) -> _TensorDict:
        stack_dim = self.stack_dim + len(shape)
        tensordicts = [td.expand(*shape) for td in self.tensordicts]
        if inplace:
            self.tensordicts = tensordicts
            self.stack_dim = stack_dim
            return self
        return torch.stack(tensordicts, stack_dim)

    def update(
        self, input_dict_or_td: _TensorDict, clone: bool = False, **kwargs
    ) -> _TensorDict:
        if input_dict_or_td is self:
            # no op
            return self
        for key, value in input_dict_or_td.items():
            if not isinstance(value, _accepted_classes):
                raise TypeError(
                    f"Expected value to be one of types {_accepted_classes} "
                    f"but got {type(value)}"
                )
            if clone:
                value = value.clone()
            self.set(key, value, **kwargs)
        return self

    def update_(
        self,
        input_dict_or_td: Union[Dict[str, COMPATIBLE_TYPES], _TensorDict],
        clone: bool = False,
        **kwargs,
    ) -> _TensorDict:
        if input_dict_or_td is self:
            # no op
            return self
        for key, value in input_dict_or_td.items():
            if not isinstance(value, _accepted_classes):
                raise TypeError(
                    f"Expected value to be one of types {_accepted_classes} "
                    f"but got {type(value)}"
                )
            if clone:
                value = value.clone()
            self.set_(key, value, **kwargs)
        return self

    def rename_key(self, old_key: str, new_key: str, safe: bool = False) -> _TensorDict:
        for td in self.tensordicts:
            td.rename_key(old_key, new_key, safe=safe)
        self._valid_keys = sorted(
            [key if key != old_key else new_key for key in self._valid_keys]
        )
        return self

    def masked_fill_(
        self, mask: torch.Tensor, value: Union[float, bool]
    ) -> _TensorDict:
        mask_unbind = mask.unbind(dim=self.stack_dim)
        for _mask, td in zip(mask_unbind, self.tensordicts):
            td.masked_fill_(_mask, value)
        return self

    def masked_fill(self, mask: torch.Tensor, value: Union[float, bool]) -> _TensorDict:
        td_copy = self.clone()
        return td_copy.masked_fill_(mask, value)


class SavedTensorDict(_TensorDict):
    _safe = False
    _lazy = False

    def __init__(
        self,
        source: _TensorDict,
        device: Optional[torch.device] = None,
        batch_size: Optional[Sequence[int]] = None,
    ):
        if not isinstance(source, _TensorDict):
            raise TypeError(
                f"Expected source to be a _TensorDict instance, but got {type(source)} instead."
            )
        elif isinstance(source, SavedTensorDict):
            source = source._load()
        if any(val.requires_grad for val in source.values_meta()):
            raise Exception(
                "SavedTensorDicts is not compatible with gradients, one of Tensors has requires_grad equals True"
            )
        self.file = tempfile.NamedTemporaryFile()
        self.filename = self.file.name
        # if source.is_memmap():
        #     source = source.clone()
        self._device = (
            torch.device(device)
            if device is not None
            else source._device_safe()
            if (hasattr(source, "_device_safe") and source._device_safe() is not None)
            else source[list(source.keys())[0]].device
            if source.keys()
            else None
        )
        self._save(source)
        if batch_size is not None and batch_size != self.batch_size:
            raise RuntimeError("batch_size does not match self.batch_size.")

    def _save(self, tensordict: _TensorDict) -> None:
        self._version = uuid.uuid1()
        self._keys = list(tensordict.keys())
        self._batch_size = tensordict.batch_size
        self._td_fields = _td_fields(tensordict)
        self._dict_meta = {key: value for key, value in tensordict.items_meta()}
        torch.save(tensordict, self.filename)

    def _make_meta(self, key: str) -> MetaTensor:
        if key not in self._dict_meta:
            raise RuntimeError(
                f'the key "{key}" was not found in SavedTensorDict._dict_meta (keys: {self._dict_meta.keys()}.'
            )
        return self._dict_meta["key"]

    def _load(self) -> _TensorDict:
        return torch.load(self.filename, map_location=self._device_safe())

    @property
    def batch_size(self) -> torch.Size:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, new_size: torch.Size):
        return self._batch_size_setter(new_size)

    @property
    def device(self) -> torch.device:
        device = self._device
        if device is None and not self.is_empty():
            for _, item in self.items_meta():
                device = item.device
                break
        elif device is None:
            raise RuntimeError(
                "querying device from an empty tensordict is not permitted, "
                "unless this device has been specified upon creation."
            )
        return device

    @device.setter
    def device(self, value: DEVICE_TYPING) -> None:
        if self._device is None:
            self._device = torch.device(value)
        else:
            raise RuntimeError(
                "device cannot be set using tensordict.device = device, "
                "because device cannot be updated in-place. To update device, use "
                "tensordict.to(new_device), which will return a new tensordict "
                "on the new device."
            )

    def _device_safe(self) -> Union[None, torch.device]:
        return self._device

    def keys(self) -> Sequence[str]:
        for k in self._keys:
            yield k

    def get(
        self, key: str, default: Union[str, COMPATIBLE_TYPES] = "_no_default_"
    ) -> COMPATIBLE_TYPES:
        td = self._load()
        return td.get(key, default=default)

    def set(self, key: str, value: COMPATIBLE_TYPES, **kwargs) -> _TensorDict:
        if self.is_locked:
            raise RuntimeError("Cannot modify immutable TensorDict")
        td = self._load()
        td.set(key, value, **kwargs)
        self._save(td)
        return self

    def expand(self, *shape: int, inplace: bool = False) -> _TensorDict:
        td = self._load()
        td = td.expand(*shape)
        if inplace:
            self._save(td)
            return self
        return td.to(SavedTensorDict)

    def _stack_onto_(
        self,
        key: str,
        list_item: List[COMPATIBLE_TYPES],
        dim: int,
    ) -> _TensorDict:
        if self.is_locked:
            raise RuntimeError("Cannot modify immutable TensorDict")
        td = self._load()
        td._stack_onto_(key, list_item, dim)
        self._save(td)
        return self

    def set_(
        self, key: str, value: COMPATIBLE_TYPES, no_check: bool = False
    ) -> _TensorDict:
        if not no_check and self.is_locked:
            raise RuntimeError("Cannot modify immutable TensorDict")
        self.set(key, value)
        return self

    def set_at_(
        self, key: str, value: COMPATIBLE_TYPES, idx: INDEX_TYPING
    ) -> _TensorDict:
        if self.is_locked:
            raise RuntimeError("Cannot modify immutable TensorDict")
        td = self._load()
        td.set_at_(key, value, idx)
        self._save(td)
        return self

    def update(
        self,
        input_dict_or_td: Union[Dict[str, COMPATIBLE_TYPES], _TensorDict],
        clone: bool = False,
        **kwargs,
    ) -> _TensorDict:
        if input_dict_or_td is self:
            # no op
            return self
        td = self._load()
        for key, value in input_dict_or_td.items():
            if not isinstance(value, _accepted_classes):
                raise TypeError(
                    f"Expected value to be one of types {_accepted_classes} "
                    f"but got {type(value)}"
                )
            if clone:
                value = value.clone()
            td.set(key, value, **kwargs)
        self._save(td)
        return self

    def update_(
        self,
        input_dict_or_td: Union[Dict[str, COMPATIBLE_TYPES], _TensorDict],
        clone: bool = False,
    ) -> _TensorDict:
        if input_dict_or_td is self:
            return self
        return self.update(input_dict_or_td, clone=clone)

    def __del__(self) -> None:
        if hasattr(self, "file"):
            self.file.close()

    def is_shared(self, no_check: bool = False) -> bool:
        return False

    def is_memmap(self, no_check: bool = False) -> bool:
        return False

    def share_memory_(self) -> _TensorDict:
        raise RuntimeError("SavedTensorDict cannot be put in shared memory.")

    def memmap_(self, prefix=None) -> _TensorDict:
        raise RuntimeError(
            "SavedTensorDict and memmap are mutually exclusive features."
        )

    def detach_(self) -> _TensorDict:
        raise RuntimeError("SavedTensorDict cannot be put detached.")

    def items(self) -> Iterator[Tuple[str, COMPATIBLE_TYPES]]:
        version = self._version
        for v in self._load().items():
            if version != self._version:
                raise RuntimeError("The SavedTensorDict changed while querying items.")
            yield v

    def values(self) -> Iterator[COMPATIBLE_TYPES]:
        version = self._version
        for v in self._load().values():
            if version != self._version:
                raise RuntimeError("The SavedTensorDict changed while querying values.")
            yield v

    def is_contiguous(self) -> bool:
        return False

    def contiguous(self) -> _TensorDict:
        return self._load().contiguous()

    def clone(self, recursive: bool = True) -> _TensorDict:
        return SavedTensorDict(self, device=self.device)

    def select(self, *keys: str, inplace: bool = False) -> _TensorDict:
        _source = self.contiguous().select(*keys)
        if inplace:
            self._save(_source)
            return self
        return SavedTensorDict(source=_source)

    def rename_key(self, old_key: str, new_key: str, safe: bool = False) -> _TensorDict:
        td = self._load()
        td.rename_key(old_key, new_key, safe=safe)
        self._save(td)
        return self

    def __repr__(self) -> str:
        return (
            f"SavedTensorDict(\n\tfields={{{self._td_fields}}}, \n\t"
            f"batch_size={self.batch_size}, \n\tfile={self.filename})"
        )

    def to(self, dest: Union[DEVICE_TYPING, Type], **kwargs):
        if isinstance(dest, type) and issubclass(dest, _TensorDict):
            if isinstance(self, dest):
                return self
            kwargs.update({"batch_size": self.batch_size})
            td = dest(
                source=self.to_dict(),
                **kwargs,
            )
            return td
        elif isinstance(dest, (torch.device, str, int)):
            # must be device
            dest = torch.device(dest)
            try:
                if dest == self.device:
                    return self
            except RuntimeError:
                pass
            self_copy = copy(self)
            self_copy._device = dest
            self_copy._dict_meta = deepcopy(self._dict_meta)
            for k, item in self.items_meta():
                self_copy._dict_meta[k].device = dest
            return self_copy
        if isinstance(dest, torch.Size):
            self.batch_size = dest
        else:
            raise NotImplementedError(
                f"dest must be a string, torch.device or a TensorDict "
                f"instance, {dest} not allowed"
            )

    def _change_batch_size(self, new_size: torch.Size):
        if not hasattr(self, "_orig_batch_size"):
            self._orig_batch_size = self.batch_size
        elif self._orig_batch_size == new_size:
            del self._orig_batch_size
        self._batch_size = new_size

    def del_(self, key: str) -> _TensorDict:
        td = self._load()
        td = td.del_(key)
        self._save(td)
        return self

    def pin_memory(self) -> _TensorDict:
        raise RuntimeError("pin_memory requires tensordicts that live in memory.")

    def __reduce__(self, *args, **kwargs):
        if hasattr(self, "file"):
            file = self.file
            del self.file
            self_copy = copy(self)
            self.file = file
            return super(SavedTensorDict, self_copy).__reduce__(*args, **kwargs)
        return super().__reduce__(*args, **kwargs)

    def __getitem__(self, idx: INDEX_TYPING) -> _TensorDict:
        if isinstance(idx, list):
            idx = torch.tensor(idx, device=self.device)
        if isinstance(idx, tuple) and any(
            isinstance(sub_index, list) for sub_index in idx
        ):
            idx = tuple(
                torch.tensor(sub_index, device=self.device)
                if isinstance(sub_index, list)
                else sub_index
                for sub_index in idx
            )
        if idx is Ellipsis or (isinstance(idx, tuple) and Ellipsis in idx):
            idx = convert_ellipsis_to_idx(idx, self.batch_size)

        if isinstance(idx, tuple) and sum(
            isinstance(_idx, str) for _idx in idx
        ) not in [len(idx), 0]:
            raise IndexError(_STR_MIXED_INDEX_ERROR)

        if isinstance(idx, str):
            return self.get(idx)
        elif isinstance(idx, tuple) and all(
            isinstance(sub_idx, str) for sub_idx in idx
        ):
            out = self.get(idx[0])
            if len(idx) > 1:
                return out[idx[1:]]
            else:
                return out
        elif isinstance(idx, Number):
            idx = (idx,)
        elif isinstance(idx, torch.Tensor) and idx.dtype == torch.bool:
            return self.masked_select(idx)
        if not self.batch_size:
            raise IndexError(
                "indexing a tensordict with td.batch_dims==0 is not permitted"
            )
        return self.get_sub_tensordict(idx)

    def masked_fill_(
        self, mask: torch.Tensor, value: Union[float, bool]
    ) -> _TensorDict:
        td = self._load()
        td.masked_fill_(mask, value)
        self._save(td)
        return self

    def masked_fill(self, mask: torch.Tensor, value: Union[float, bool]) -> _TensorDict:
        td_copy = self.clone()
        return td_copy.masked_fill_(mask, value)


class _CustomOpTensorDict(_TensorDict):
    """Encodes lazy operations on tensors contained in a TensorDict."""

    _lazy = True

    def __init__(
        self,
        source: _TensorDict,
        custom_op: str,
        inv_op: Optional[str] = None,
        custom_op_kwargs: Optional[dict] = None,
        inv_op_kwargs: Optional[dict] = None,
        batch_size: Optional[Sequence[int]] = None,
    ):
        super().__init__()

        self._is_shared = None
        self._is_memmap = None

        if not isinstance(source, _TensorDict):
            raise TypeError(
                f"Expected source to be a _TensorDict isntance, "
                f"but got {type(source)} instead."
            )
        self._source = source
        self.custom_op = custom_op
        self.inv_op = inv_op
        self.custom_op_kwargs = custom_op_kwargs if custom_op_kwargs is not None else {}
        self.inv_op_kwargs = inv_op_kwargs if inv_op_kwargs is not None else {}
        self._batch_size = None
        if batch_size is not None and batch_size != self.batch_size:
            raise RuntimeError("batch_size does not match self.batch_size.")

    def _update_custom_op_kwargs(self, source_meta_tensor: MetaTensor) -> dict:
        """Allows for a transformation to be customized for a certain shape,
        device or dtype. By default, this is a no-op on self.custom_op_kwargs

        Args:
            source_meta_tensor: corresponding MetaTensor

        Returns:
            a dictionary with the kwargs of the operation to execute
            for the tensor

        """
        return self.custom_op_kwargs

    def _update_inv_op_kwargs(self, source_tensor: torch.Tensor) -> dict:
        """Allows for an inverse transformation to be customized for a
        certain shape, device or dtype.

        By default, this is a no-op on self.inv_op_kwargs

        Args:
            source_tensor: corresponding tensor

        Returns:
            a dictionary with the kwargs of the operation to execute for
            the tensor

        """
        return self.inv_op_kwargs

    @property
    def device(self) -> torch.device:
        return self._source.device

    @device.setter
    def device(self, value: DEVICE_TYPING) -> None:
        self._source.device = value

    def _device_safe(self) -> Union[None, torch.device]:
        return self._source._device_safe()

    def _make_meta(self, key: str) -> MetaTensor:
        item = self._source._get_meta(key)
        return getattr(item, self.custom_op)(**self._update_custom_op_kwargs(item))

    def _get_meta(self, key) -> MetaTensor:
        return self._make_meta(key)

    @property
    def batch_size(self) -> torch.Size:
        if self._batch_size is None:
            self._batch_size = getattr(
                MetaTensor(*self._source.batch_size), self.custom_op
            )(**self.custom_op_kwargs).shape
        return self._batch_size

    @batch_size.setter
    def batch_size(self, new_size: torch.Size):
        return self._batch_size_setter(new_size)

    def _change_batch_size(self, new_size: torch.Size):
        if not hasattr(self, "_orig_batch_size"):
            self._orig_batch_size = self.batch_size
        elif self._orig_batch_size == new_size:
            del self._orig_batch_size
        self._batch_size = new_size

    def get(
        self,
        key: str,
        default: Union[str, COMPATIBLE_TYPES] = "_no_default_",
        _return_original_tensor: bool = False,
    ) -> COMPATIBLE_TYPES:
        if key in self._source.keys():
            source_meta_tensor = self._source._get_meta(key)
            item = self._source.get(key)
            transformed_tensor = getattr(item, self.custom_op)(
                **self._update_custom_op_kwargs(source_meta_tensor)
            )
            if not _return_original_tensor:
                return transformed_tensor
            return transformed_tensor, item
        else:
            if _return_original_tensor:
                raise RuntimeError(
                    "_return_original_tensor not compatible with get(..., "
                    "default=smth)"
                )
            return self._default_get(key, default)

    def set(self, key: str, value: COMPATIBLE_TYPES, **kwargs) -> _TensorDict:
        if self.inv_op is None:
            raise Exception(
                f"{self.__class__.__name__} does not support setting values. "
                f"Consider calling .contiguous() before calling this method."
            )
        if self.is_locked:
            raise RuntimeError("Cannot modify immutable TensorDict")
        proc_value = self._process_tensor(
            value,
            check_device=False,
            check_tensor_shape=True,
        )
        proc_value = getattr(proc_value, self.inv_op)(
            **self._update_inv_op_kwargs(proc_value)
        )
        self._source.set(key, proc_value, **kwargs)
        return self

    def set_(
        self, key: str, value: COMPATIBLE_TYPES, no_check: bool = False
    ) -> _CustomOpTensorDict:
        if not no_check:
            if self.is_locked:
                raise RuntimeError("Cannot modify immutable TensorDict")
            if self.inv_op is None:
                raise Exception(
                    f"{self.__class__.__name__} does not support setting values. "
                    f"Consider calling .contiguous() before calling this method."
                )
        value = getattr(value, self.inv_op)(**self._update_inv_op_kwargs(value))
        self._source.set_(key, value)
        return self

    def set_at_(
        self, key: str, value: COMPATIBLE_TYPES, idx: INDEX_TYPING
    ) -> _CustomOpTensorDict:
        if self.is_locked:
            raise RuntimeError("Cannot modify immutable TensorDict")
        transformed_tensor, original_tensor = self.get(
            key, _return_original_tensor=True
        )
        if transformed_tensor.data_ptr() != original_tensor.data_ptr():
            raise RuntimeError(
                f"{self} original tensor and transformed_in do not point to the "
                f"same storage. Setting values in place is not currently "
                f"supported in this setting, consider calling "
                f"`td.clone()` before `td.set_at_(...)`"
            )
        transformed_tensor[idx] = value
        return self

    def _stack_onto_(
        self,
        key: str,
        list_item: List[COMPATIBLE_TYPES],
        dim: int,
    ) -> _TensorDict:
        raise RuntimeError(
            f"stacking tensordicts is not allowed for type {type(self)}"
            f"consider calling 'to_tensordict()` first"
        )

    def __repr__(self) -> str:
        custom_op_kwargs_str = ", ".join(
            [f"{key}={value}" for key, value in self.custom_op_kwargs.items()]
        )
        indented_source = textwrap.indent(f"source={self._source}", "\t")
        return (
            f"{self.__class__.__name__}(\n{indented_source}, "
            f"\n\top={self.custom_op}({custom_op_kwargs_str}))"
        )

    def keys(self) -> KeysView:
        return self._source.keys()

    def select(self, *keys: str, inplace: bool = False) -> _CustomOpTensorDict:
        if inplace:
            self._source.select(*keys, inplace=inplace)
            return self
        self_copy = copy(self)
        self_copy._source = self_copy._source.select(*keys)
        return self_copy

    def clone(self, recursive: bool = True) -> _TensorDict:
        if not recursive:
            return copy(self)
        return TensorDict(
            source=self.to_dict(),
            batch_size=self.batch_size,
            device=self._device_safe(),
        )

    def is_contiguous(self) -> bool:
        return all([value.is_contiguous() for _, value in self.items()])

    def contiguous(self) -> _TensorDict:
        if self.is_contiguous():
            return self
        return self.to(TensorDict)

    def rename_key(
        self, old_key: str, new_key: str, safe: bool = False
    ) -> _CustomOpTensorDict:
        self._source.rename_key(old_key, new_key, safe=safe)
        return self

    def del_(self, key: str) -> _CustomOpTensorDict:
        self._source = self._source.del_(key)
        return self

    def to(self, dest: Union[DEVICE_TYPING, Type], **kwargs) -> _TensorDict:
        if isinstance(dest, type) and issubclass(dest, _TensorDict):
            if isinstance(self, dest):
                return self
            return dest(source=self)
        elif isinstance(dest, (torch.device, str, int)):
            if self._device_safe() is not None and torch.device(dest) == self.device:
                return self
            td = self._source.to(dest)
            self_copy = copy(self)
            self_copy._source = td
            return self_copy
        else:
            raise NotImplementedError(
                f"dest must be a string, torch.device or a TensorDict "
                f"instance, {dest} not allowed"
            )

    def pin_memory(self) -> _TensorDict:
        self._source.pin_memory()
        return self

    def detach_(self):
        self._source.detach_()

    def masked_fill_(
        self, mask: torch.Tensor, value: Union[float, bool]
    ) -> _TensorDict:
        for key, item in self.items():
            # source_meta_tensor = self._get_meta(key)
            val = self._source.get(key)
            mask_exp = expand_right(
                mask, list(mask.shape) + list(val.shape[self._source.batch_dims :])
            )
            mask_proc_inv = getattr(mask_exp, self.inv_op)(
                **self._update_inv_op_kwargs(item)
            )
            val[mask_proc_inv] = value
            self._source.set(key, val)
        return self

    def masked_fill(self, mask: torch.Tensor, value: Union[float, bool]) -> _TensorDict:
        td_copy = self.clone()
        return td_copy.masked_fill_(mask, value)

    def memmap_(self, prefix=None):
        self._source.memmap_(prefix=prefix)
        self._is_memmap = True
        return self

    def share_memory_(self):
        self._source.share_memory_()
        self._is_shared = True


class UnsqueezedTensorDict(_CustomOpTensorDict):
    """A lazy view on an unsqueezed TensorDict.

    When calling `tensordict.unsqueeze(dim)`, a lazy view of this operation is
    returned such that the following code snippet works without raising an
    exception:

        >>> assert tensordict.unsqueeze(dim).squeeze(dim) is tensordict

    Examples:
        >>> from torchrl.data import TensorDict
        >>> import torch
        >>> td = TensorDict({'a': torch.randn(3, 4)}, batch_size=[3])
        >>> td_unsqueeze = td.unsqueeze(-1)
        >>> print(td_unsqueeze.shape)
        torch.Size([3, 1])
        >>> print(td_unsqueeze.squeeze(-1) is td)
        True
    """

    def squeeze(self, dim: int) -> _TensorDict:
        if dim < 0:
            dim = self.batch_dims + dim
        if dim == self.custom_op_kwargs.get("dim"):
            return self._source
        return super().squeeze(dim)

    def _stack_onto_(
        self,
        key: str,
        list_item: List[COMPATIBLE_TYPES],
        dim: int,
    ) -> _TensorDict:
        unsqueezed_dim = self.custom_op_kwargs["dim"]
        diff_to_apply = 1 if dim < unsqueezed_dim else 0
        list_item_unsqueeze = [
            item.squeeze(unsqueezed_dim - diff_to_apply) for item in list_item
        ]
        return self._source._stack_onto_(key, list_item_unsqueeze, dim)


class SqueezedTensorDict(_CustomOpTensorDict):
    """
    A lazy view on a squeezed TensorDict.
    See the `UnsqueezedTensorDict` class documentation for more information.
    """

    def unsqueeze(self, dim: int) -> _TensorDict:
        if dim < 0:
            dim = self.batch_dims + dim + 1
        inv_op_dim = self.inv_op_kwargs.get("dim")
        if inv_op_dim < 0:
            inv_op_dim = self.batch_dims + inv_op_dim + 1
        if dim == inv_op_dim:
            return self._source
        return super().unsqueeze(dim)

    def _stack_onto_(
        self,
        key: str,
        list_item: List[COMPATIBLE_TYPES],
        dim: int,
    ) -> _TensorDict:
        squeezed_dim = self.custom_op_kwargs["dim"]
        # dim=0, squeezed_dim=2, [3, 4, 5] [3, 4, 1, 5] [[4, 5], [4, 5], [4, 5]] => unsq 1
        # dim=1, squeezed_dim=2, [3, 4, 5] [3, 4, 1, 5] [[3, 5], [3, 5], [3, 5], [3, 4]] => unsq 1
        # dim=2, squeezed_dim=2, [3, 4, 5] [3, 4, 1, 5] [[3, 4], [3, 4], ...] => unsq 2
        diff_to_apply = 1 if dim < squeezed_dim else 0
        list_item_unsqueeze = [
            item.unsqueeze(squeezed_dim - diff_to_apply) for item in list_item
        ]
        return self._source._stack_onto_(key, list_item_unsqueeze, dim)


class ViewedTensorDict(_CustomOpTensorDict):
    def _update_custom_op_kwargs(self, source_meta_tensor: MetaTensor) -> dict:
        new_dim_list = list(self.custom_op_kwargs.get("size"))
        new_dim_list += list(source_meta_tensor.shape[self._source.batch_dims :])
        new_dim = torch.Size(new_dim_list)
        new_dict = deepcopy(self.custom_op_kwargs)
        new_dict.update({"size": new_dim})
        return new_dict

    def _update_inv_op_kwargs(self, tensor: torch.Tensor) -> Dict:
        size = list(self.inv_op_kwargs.get("size"))
        size += list(tensor.shape[self.batch_dims :])
        new_dim = torch.Size(size)
        new_dict = deepcopy(self.inv_op_kwargs)
        new_dict.update({"size": new_dim})
        return new_dict

    def view(
        self, *shape, size: Optional[Union[List, Tuple, torch.Size]] = None
    ) -> _TensorDict:
        if len(shape) == 0 and size is not None:
            return self.view(*size)
        elif len(shape) == 1 and isinstance(shape[0], (list, tuple, torch.Size)):
            return self.view(*shape[0])
        elif not isinstance(shape, torch.Size):
            shape = torch.Size(shape)
        if shape == self._source.batch_size:
            return self._source
        return super().view(*shape)


class PermutedTensorDict(_CustomOpTensorDict):
    """
    A lazy view on a TensorDict with the batch dimensions permuted.
    """

    def add_missing_dims(self, num_dims: int, batch_dims: tuple) -> tuple:
        dim_diff = num_dims - len(batch_dims)
        all_dims = [i for i in range(num_dims)]
        for i, x in enumerate(batch_dims):
            if x < 0:
                x = x - dim_diff
            all_dims[i] = x
        return tuple(all_dims)

    def _update_custom_op_kwargs(self, source_meta_tensor: MetaTensor) -> Dict:
        new_dims = self.add_missing_dims(
            len(source_meta_tensor.shape), self.custom_op_kwargs["dims"]
        )
        kwargs = deepcopy(self.custom_op_kwargs)
        kwargs.update({"dims": new_dims})
        return kwargs

    def _update_inv_op_kwargs(self, tensor: torch.Tensor) -> dict:
        new_dims = self.add_missing_dims(
            self._source.batch_dims + len(tensor.shape[self.batch_dims :]),
            self.custom_op_kwargs["dims"],
        )
        kwargs = deepcopy(self.custom_op_kwargs)
        kwargs.update({"dims": tuple(np.argsort(new_dims))})
        return kwargs

    def _stack_onto_(
        self,
        key: str,
        list_item: List[COMPATIBLE_TYPES],
        dim: int,
    ) -> _TensorDict:

        permute_dims = self.custom_op_kwargs["dims"]
        inv_permute_dims = np.argsort(permute_dims)
        new_dim = [i for i, v in enumerate(inv_permute_dims) if v == dim][0]
        inv_permute_dims = [p for p in inv_permute_dims if p != dim]
        inv_permute_dims = np.argsort(np.argsort(inv_permute_dims))

        list_permuted_items = []
        for item in list_item:
            perm = list(inv_permute_dims) + list(
                range(self.batch_dims - 1, item.ndimension())
            )
            list_permuted_items.append(item.permute(*perm))
        self._source._stack_onto_(key, list_permuted_items, new_dim)
        return self


def _td_fields(td: _TensorDict) -> str:
    return indent(
        "\n"
        + ",\n".join(
            sorted([f"{key}: {item.get_repr()}" for key, item in td.items_meta()])
        ),
        4 * " ",
    )


def _check_keys(
    list_of_tensordicts: Sequence[_TensorDict], strict: bool = False
) -> Set[str]:
    keys: Set[str] = set()
    for td in list_of_tensordicts:
        if not len(keys):
            keys = set(td.keys())
        else:
            if not strict:
                keys = keys.intersection(set(td.keys()))
            else:
                if len(set(td.keys()).difference(keys)) or len(set(td.keys())) != len(
                    keys
                ):
                    raise KeyError(
                        f"got keys {keys} and {set(td.keys())} which are "
                        f"incompatible"
                    )
    return keys


_accepted_classes = (torch.Tensor, MemmapTensor, _TensorDict)


def _expand_to_match_shape(parent_batch_size, tensor, self_batch_dims, self_device):
    if hasattr(tensor, "dtype"):
        return torch.zeros(
            *parent_batch_size,
            *tensor.shape[self_batch_dims:],
            dtype=tensor.dtype,
            device=self_device,
        )
    else:
        # tensordict
        out = TensorDict(
            {},
            [*parent_batch_size, *tensor.shape[self_batch_dims:]],
            device=self_device,
        )
        return out
