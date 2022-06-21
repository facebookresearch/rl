# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import collections
import concurrent.futures
import functools
import threading
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from torchrl._torchrl import (
    MinSegmentTreeFp32,
    MinSegmentTreeFp64,
    SumSegmentTreeFp32,
    SumSegmentTreeFp64,
)
from torchrl.data.replay_buffers.utils import (
    cat_fields_to_device,
    to_numpy,
    to_torch,
)

__all__ = [
    "ReplayBuffer",
    "PrioritizedReplayBuffer",
    "TensorDictReplayBuffer",
    "TensorDictPrioritizedReplayBuffer",
    "create_replay_buffer",
    "create_prioritized_replay_buffer",
]

from torchrl.data.tensordict.tensordict import (
    _TensorDict,
    stack as stack_td,
    LazyStackedTensorDict,
)
from torchrl.data.utils import DEVICE_TYPING


def stack_tensors(list_of_tensor_iterators: List) -> Tuple[torch.Tensor]:
    """Zips a list of iterables containing tensor-like objects and stacks the
    resulting lists of tensors together.

    Args:
        list_of_tensor_iterators (list): Sequence containing similar iterators,
            where each element of the nested iterator is a tensor whose
            shape match the tensor of other iterators that have the same index.

    Returns:
         Tuple of stacked tensors.

    Examples:
         >>> list_of_tensor_iterators = [[torch.ones(3), torch.zeros(1,2)]
         ...     for _ in range(4)]
         >>> stack_tensors(list_of_tensor_iterators)
         (tensor([[1., 1., 1.],
                 [1., 1., 1.],
                 [1., 1., 1.],
                 [1., 1., 1.]]), tensor([[[0., 0.]],
         <BLANKLINE>
                 [[0., 0.]],
         <BLANKLINE>
                 [[0., 0.]],
         <BLANKLINE>
                 [[0., 0.]]]))

    """
    return tuple(torch.stack(tensors, 0) for tensors in zip(*list_of_tensor_iterators))


def _pin_memory(output: Any) -> Any:
    if hasattr(output, "pin_memory") and output.device == torch.device("cpu"):
        return output.pin_memory()
    else:
        return output


def pin_memory_output(fun) -> Callable:
    """Calls pin_memory on outputs of decorated function if they have such
    method."""

    def decorated_fun(self, *args, **kwargs):
        output = fun(self, *args, **kwargs)
        if self._pin_memory:
            _tuple_out = True
            if not isinstance(output, tuple):
                _tuple_out = False
                output = (output,)
            output = tuple(_pin_memory(_output) for _output in output)
            if _tuple_out:
                return output
            return output[0]
        return output

    return decorated_fun


class ReplayBuffer:
    """
    Circular replay buffer.

    Args:
        size (int): integer indicating the maximum size of the replay buffer.
        collate_fn (callable, optional): merges a list of samples to form a
            mini-batch of Tensor(s)/outputs.  Used when using batched
            loading from a map-style dataset.
        pin_memory (bool): whether pin_memory() should be called on the rb
            samples.
        prefetch (int, optional): number of next batches to be prefetched
            using multithreading.
    """

    def __init__(
        self,
        size: int,
        collate_fn: Optional[Callable] = None,
        pin_memory: bool = False,
        prefetch: Optional[int] = None,
    ):
        self._storage = []
        self._capacity = size
        self._cursor = 0
        if collate_fn is not None:
            self._collate_fn = collate_fn
        else:
            self._collate_fn = stack_tensors
        self._pin_memory = pin_memory

        self._prefetch = prefetch is not None and prefetch > 0
        self._prefetch_cap = prefetch if prefetch is not None else 0
        self._prefetch_fut = collections.deque()
        if self._prefetch_cap > 0:
            self._prefetch_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self._prefetch_cap
            )

        self._replay_lock = threading.RLock()
        self._future_lock = threading.RLock()

    def __len__(self) -> int:
        with self._replay_lock:
            return len(self._storage)

    @pin_memory_output
    def __getitem__(self, index: Union[int, Tensor]) -> Any:
        index = to_numpy(index)

        with self._replay_lock:
            if isinstance(index, int):
                data = self._storage[index]
            else:
                data = [self._storage[i] for i in index]

        if isinstance(data, list):
            data = self._collate_fn(data)
        return data

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def cursor(self) -> int:
        with self._replay_lock:
            return self._cursor

    def add(self, data: Any) -> int:
        """Add a single element to the replay buffer.

        Args:
            data (Any): data to be added to the replay buffer

        Returns:
            index where the data lives in the replay buffer.
        """
        with self._replay_lock:
            ret = self._cursor
            if self._cursor >= len(self._storage):
                self._storage.append(data)
            else:
                self._storage[self._cursor] = data
            self._cursor = (self._cursor + 1) % self._capacity
            return ret

    def extend(self, data: Sequence[Any]):
        """Extends the replay buffer with one or more elements contained in
        an iterable.

        Args:
            data (iterable): collection of data to be added to the replay
                buffer.

        Returns:
            Indices of the data aded to the replay buffer.

        """
        if not len(data):
            raise Exception("extending with empty data is not supported")
        if not isinstance(data, list):
            data = list(data)
        with self._replay_lock:
            cur_size = len(self._storage)
            batch_size = len(data)
            storage = self._storage
            cursor = self._cursor
            if cur_size + batch_size <= self._capacity:
                index = np.arange(cur_size, cur_size + batch_size)
                self._storage += data
                self._cursor = (self._cursor + batch_size) % self._capacity
            elif cur_size < self._capacity:
                d = self._capacity - cur_size
                index = np.empty(batch_size, dtype=np.int64)
                index[:d] = np.arange(cur_size, self._capacity)
                index[d:] = np.arange(batch_size - d)
                storage += data[:d]
                for i, v in enumerate(data[d:]):
                    storage[i] = v
                self._cursor = batch_size - d
            elif self._cursor + batch_size <= self._capacity:
                index = np.arange(self._cursor, self._cursor + batch_size)
                for i, v in enumerate(data):
                    storage[cursor + i] = v
                self._cursor = (self._cursor + batch_size) % self._capacity
            else:
                d = self._capacity - self._cursor
                index = np.empty(batch_size, dtype=np.int64)
                index[:d] = np.arange(self._cursor, self._capacity)
                index[d:] = np.arange(batch_size - d)
                for i, v in enumerate(data[:d]):
                    storage[cursor + i] = v
                for i, v in enumerate(data[d:]):
                    storage[i] = v
                self._cursor = batch_size - d

            return index

    @pin_memory_output
    def _sample(self, batch_size: int) -> Any:
        index = np.random.randint(0, len(self._storage), size=batch_size)

        with self._replay_lock:
            data = [self._storage[i] for i in index]

        data = self._collate_fn(data)
        return data

    def sample(self, batch_size: int) -> Any:
        """Samples a batch of data from the replay buffer.

        Args:
            batch_size (int): float of data to be collected.

        Returns:
            A batch of data randomly selected in the replay buffer.

        """
        if not self._prefetch:
            return self._sample(batch_size)

        with self._future_lock:
            if len(self._prefetch_fut) == 0:
                ret = self._sample(batch_size)
            else:
                ret = self._prefetch_fut.popleft().result()

            while len(self._prefetch_fut) < self._prefetch_cap:
                fut = self._prefetch_executor.submit(self._sample, batch_size)
                self._prefetch_fut.append(fut)

            return ret

    def __repr__(self) -> str:
        string = (
            f"{self.__class__.__name__}(size={len(self)}, "
            f"pin_memory={self._pin_memory})"
        )
        return string


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized replay buffer as presented in
        "Schaul, T.; Quan, J.; Antonoglou, I.; and Silver, D. 2015.
        Prioritized experience replay."
        (https://arxiv.org/abs/1511.05952)

    Args:
        size (int): integer indicating the maximum size of the replay buffer.
        alpha (float): exponent α determines how much prioritization is used,
            with α = 0 corresponding to the uniform case.
        beta (float): importance sampling negative exponent.
        eps (float): delta added to the priorities to ensure that the buffer
            does not contain null priorities.
        collate_fn (callable, optional): merges a list of samples to form a
            mini-batch of Tensor(s)/outputs.  Used when using batched
            loading from a map-style dataset.
        pin_memory (bool): whether pin_memory() should be called on the rb
            samples.
        prefetch (int, optional): number of next batches to be prefetched
            using multithreading.
    """

    def __init__(
        self,
        size: int,
        alpha: float,
        beta: float,
        eps: float = 1e-8,
        dtype: torch.dtype = torch.float,
        collate_fn=None,
        pin_memory: bool = False,
        prefetch: Optional[int] = None,
    ) -> None:
        super(PrioritizedReplayBuffer, self).__init__(
            size, collate_fn, pin_memory, prefetch
        )
        if alpha <= 0:
            raise ValueError(
                f"alpha must be strictly greater than 0, got alpha={alpha}"
            )
        if beta < 0:
            raise ValueError(f"beta must be greater or equal to 0, got beta={beta}")

        self._alpha = alpha
        self._beta = beta
        self._eps = eps
        if dtype in (torch.float, torch.FloatType, torch.float32):
            self._sum_tree = SumSegmentTreeFp32(size)
            self._min_tree = MinSegmentTreeFp32(size)
        elif dtype in (torch.double, torch.DoubleTensor, torch.float64):
            self._sum_tree = SumSegmentTreeFp64(size)
            self._min_tree = MinSegmentTreeFp64(size)
        else:
            raise NotImplementedError(
                f"dtype {dtype} not supported by PrioritizedReplayBuffer"
            )
        self._max_priority = 1.0

    @pin_memory_output
    def __getitem__(self, index: Union[int, Tensor]) -> Any:
        index = to_numpy(index)

        with self._replay_lock:
            p_min = self._min_tree.query(0, self._capacity)
            if p_min <= 0:
                raise ValueError(f"p_min must be greater than 0, got p_min={p_min}")
            if isinstance(index, int):
                data = self._storage[index]
                weight = np.array(self._sum_tree[index])
            else:
                data = [self._storage[i] for i in index]
                weight = self._sum_tree[index]

        if isinstance(data, list):
            data = self._collate_fn(data)
        # weight = np.power(weight / (p_min + self._eps), -self._beta)
        weight = np.power(weight / p_min, -self._beta)
        # x = first_field(data)
        # if isinstance(x, torch.Tensor):
        device = data.device if hasattr(data, "device") else torch.device("cpu")
        weight = to_torch(weight, device, self._pin_memory)
        return data, weight

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def beta(self) -> float:
        return self._beta

    @property
    def eps(self) -> float:
        return self._eps

    @property
    def max_priority(self) -> float:
        with self._replay_lock:
            return self._max_priority

    @property
    def _default_priority(self) -> float:
        return (self._max_priority + self._eps) ** self._alpha

    def _add_or_extend(
        self,
        data: Any,
        priority: Optional[torch.Tensor] = None,
        do_add: bool = True,
    ) -> torch.Tensor:
        if priority is not None:
            priority = to_numpy(priority)
            max_priority = np.max(priority)
            with self._replay_lock:
                self._max_priority = max(self._max_priority, max_priority)
            priority = np.power(priority + self._eps, self._alpha)
        else:
            with self._replay_lock:
                priority = self._default_priority

        if do_add:
            index = super(PrioritizedReplayBuffer, self).add(data)
        else:
            index = super(PrioritizedReplayBuffer, self).extend(data)

        if not (
            isinstance(priority, float)
            or len(priority) == 1
            or len(priority) == len(index)
        ):
            raise RuntimeError(
                "priority should be a scalar or an iterable of the same "
                "length as index"
            )

        with self._replay_lock:
            self._sum_tree[index] = priority
            self._min_tree[index] = priority

        return index

    def add(self, data: Any, priority: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self._add_or_extend(data, priority, True)

    def extend(
        self, data: Sequence, priority: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self._add_or_extend(data, priority, False)

    @pin_memory_output
    def _sample(self, batch_size: int) -> Tuple[Any, torch.Tensor, torch.Tensor]:
        with self._replay_lock:
            p_sum = self._sum_tree.query(0, self._capacity)
            p_min = self._min_tree.query(0, self._capacity)
            if p_sum <= 0:
                raise RuntimeError("negative p_sum")
            if p_min <= 0:
                raise RuntimeError("negative p_min")
            mass = np.random.uniform(0.0, p_sum, size=batch_size)
            index = self._sum_tree.scan_lower_bound(mass)
            if isinstance(index, torch.Tensor):
                index.clamp_max_(len(self._storage) - 1)
            else:
                index = np.clip(index, None, len(self._storage) - 1)
            data = [self._storage[i] for i in index]
            weight = self._sum_tree[index]

        data = self._collate_fn(data)

        # Importance sampling weight formula:
        #   w_i = (p_i / sum(p) * N) ^ (-beta)
        #   weight_i = w_i / max(w)
        #   weight_i = (p_i / sum(p) * N) ^ (-beta) /
        #       ((min(p) / sum(p) * N) ^ (-beta))
        #   weight_i = ((p_i / sum(p) * N) / (min(p) / sum(p) * N)) ^ (-beta)
        #   weight_i = (p_i / min(p)) ^ (-beta)
        # weight = np.power(weight / (p_min + self._eps), -self._beta)
        weight = np.power(weight / p_min, -self._beta)

        # x = first_field(data)  # avoid calling tree.flatten
        # if isinstance(x, torch.Tensor):
        device = data.device if hasattr(data, "device") else torch.device("cpu")
        weight = to_torch(weight, device, self._pin_memory)
        return data, weight, index

    def sample(self, batch_size: int) -> Tuple[Any, np.ndarray, torch.Tensor]:
        """Gather a batch of data according to the non-uniform multinomial
        distribution with weights computed with the provided priorities of
        each input.

        Args:
            batch_size (int): float of data to be collected.

        Returns:

        """
        if not self._prefetch:
            return self._sample(batch_size)

        with self._future_lock:
            if len(self._prefetch_fut) == 0:
                ret = self._sample(batch_size)
            else:
                ret = self._prefetch_fut.popleft().result()

            while len(self._prefetch_fut) < self._prefetch_cap:
                fut = self._prefetch_executor.submit(self._sample, batch_size)
                self._prefetch_fut.append(fut)

            return ret

    def update_priority(
        self, index: Union[int, Tensor], priority: Union[float, Tensor]
    ) -> None:
        """Updates the priority of the data pointed by the index.

        Args:
            index (int or torch.Tensor): indexes of the priorities to be
                updated.
            priority (Number or torch.Tensor): new priorities of the
                indexed elements


        """
        if isinstance(index, int):
            if not isinstance(priority, float):
                if len(priority) != 1:
                    raise RuntimeError(
                        f"priority length should be 1, got {len(priority)}"
                    )
                priority = priority.item()
        else:
            if not (
                isinstance(priority, float)
                or len(priority) == 1
                or len(index) == len(priority)
            ):
                raise RuntimeError(
                    "priority should be a number or an iterable of the same "
                    "length as index"
                )
            index = to_numpy(index)
            priority = to_numpy(priority)

        with self._replay_lock:
            self._max_priority = max(self._max_priority, np.max(priority))
            priority = np.power(priority + self._eps, self._alpha)
            self._sum_tree[index] = priority
            self._min_tree[index] = priority


class TensorDictReplayBuffer(ReplayBuffer):
    """
    TensorDict-specific wrapper around the ReplayBuffer class.
    """

    def __init__(
        self,
        size: int,
        collate_fn: Optional[Callable] = None,
        pin_memory: bool = False,
        prefetch: Optional[int] = None,
    ):
        if collate_fn is None:

            def collate_fn(x):
                return stack_td(x, 0, contiguous=True)

        super().__init__(size, collate_fn, pin_memory, prefetch)

    def sample(self, size: int) -> Any:
        return super(TensorDictReplayBuffer, self).sample(size)


class TensorDictPrioritizedReplayBuffer(PrioritizedReplayBuffer):
    """
    TensorDict-specific wrapper around the PrioritizedReplayBuffer class.
    This class returns tensordicts with a new key "index" that represents
    the index of each element in the replay buffer. It also facilitates the
    call to the 'update_priority' method, as it only requires for the
    tensordict to be passed to it with its new priority value.

    Args:
        size (int): integer indicating the maximum size of the replay buffer.
        alpha (flaot): exponent α determines how much prioritization is
            used, with α = 0 corresponding to the uniform case.
        beta (float): importance sampling negative exponent.
        priority_key (str, optional): key where the priority value can be
            found in the stored tensordicts. Default is `"td_error"`
        eps (float, optional): delta added to the priorities to ensure that the
            buffer does not contain null priorities.
        collate_fn (callable, optional): merges a list of samples to form a
            mini-batch of Tensor(s)/outputs.  Used when using batched loading
            from a map-style dataset.
        pin_memory (bool, optional): whether pin_memory() should be called on
            the rb samples. Default is `False`.
        prefetch (int, optional): number of next batches to be prefetched
            using multithreading.
    """

    def __init__(
        self,
        size: int,
        alpha: float,
        beta: float,
        priority_key: str = "td_error",
        eps: float = 1e-8,
        collate_fn=None,
        pin_memory: bool = False,
        prefetch: Optional[int] = None,
    ) -> None:
        if collate_fn is None:

            def collate_fn(x):
                return stack_td(x, 0, contiguous=True)

        super(TensorDictPrioritizedReplayBuffer, self).__init__(
            size=size,
            alpha=alpha,
            beta=beta,
            eps=eps,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            prefetch=prefetch,
        )
        self.priority_key = priority_key

    def _get_priority(self, tensordict: _TensorDict) -> torch.Tensor:
        if tensordict.batch_dims:
            tensordict = tensordict.clone(recursive=False)
            tensordict.batch_size = []
        if self.priority_key in tensordict.keys():
            try:
                priority = tensordict.get(self.priority_key).item()
            except ValueError:
                raise ValueError(
                    f"Found a priority key of size"
                    f" {tensordict.get(self.priority_key).shape} but expected "
                    f"scalar value"
                )
        else:
            priority = self._default_priority
        return priority

    def add(self, tensordict: _TensorDict) -> torch.Tensor:
        priority = self._get_priority(tensordict)
        index = super().add(tensordict, priority)
        tensordict.set("index", index)
        return index

    def extend(
        self, tensordicts: Union[_TensorDict, List[_TensorDict]]
    ) -> torch.Tensor:
        if isinstance(tensordicts, _TensorDict):
            if self.priority_key in tensordicts.keys():
                priorities = tensordicts.get(self.priority_key)
            else:
                priorities = None

            if tensordicts.batch_dims > 1:
                # we want the tensordict to have one dimension only. The batch size
                # of the sampled tensordicts can be changed thereafter
                if not isinstance(tensordicts, LazyStackedTensorDict):
                    tensordicts = tensordicts.clone(recursive=False)
                else:
                    tensordicts = tensordicts.contiguous()
                tensordicts.batch_size = tensordicts.batch_size[:1]
            # we split the tensordict such that the setting of the "index" key herebelow results in a change in
            # the tensordicts stored in the buffer
            tensordicts = tensordicts.unbind(0)
        else:
            priorities = [self._get_priority(td) for td in tensordicts]

        if not isinstance(tensordicts, _TensorDict):
            stacked_td = torch.stack(tensordicts, 0)
        else:
            stacked_td = tensordicts
        idx = super().extend(tensordicts, priorities)
        stacked_td.set("index", idx, inplace=True)
        return idx

    def update_priority(self, tensordict: _TensorDict) -> None:
        """Updates the priorities of the tensordicts stored in the replay
        buffer.

        Args:
            tensordict: tensordict with key-value pairs 'self.priority_key'
                and 'index'.


        """
        priority = tensordict.get(self.priority_key)
        if (priority < 0).any():
            raise RuntimeError(
                f"Priority must be a positive value, got "
                f"{(priority < 0).sum()} negative priority values."
            )
        return super().update_priority(tensordict.get("index"), priority=priority)

    def sample(self, size: int, return_weight: bool = False) -> _TensorDict:
        """
        Gather a batch of tensordicts according to the non-uniform multinomial
        distribution with weights computed with the priority_key of each
        input tensordict.

        Args:
            size (int): size of the batch to be returned
            return_weight (bool, optional): if True, a '_weight' key will be
                written in the output tensordict that indicates the weight
                of the selected items

        Returns:
            Stack of tensordicts

        """
        td, weight, _ = super(TensorDictPrioritizedReplayBuffer, self).sample(size)
        if return_weight:
            td.set("_weight", weight)
        return td


def create_replay_buffer(
    size: int,
    device: Optional[DEVICE_TYPING] = None,
    collate_fn: Callable = None,
    pin_memory: bool = False,
    prefetch: Optional[int] = None,
) -> ReplayBuffer:
    """
    Helper function to create a Replay buffer.

    Args:
        size (int): integer indicating the maximum size of the replay buffer.
        device (str, int or torch.device, optional): device where to cast the
            samples.
        collate_fn (callable, optional): merges a list of samples to form a
            mini-batch of Tensor(s)/outputs.  Used when using batched loading
            from a map-style dataset.
        pin_memory (bool): whether pin_memory() should be called on the rb
            samples.
        prefetch (int, optional): number of next batches to be prefetched
            using multithreading.

    Returns:
         a ReplayBuffer instance

    """
    if isinstance(device, str):
        device = torch.device(device)

    if device.type == "cuda" and collate_fn is None:
        # Postman will add batch_dim for uploaded data, so using cat instead of
        # stack here.
        collate_fn = functools.partial(cat_fields_to_device, device=device)

    return ReplayBuffer(size, collate_fn, pin_memory, prefetch)


def create_prioritized_replay_buffer(
    size: int,
    alpha: float,
    beta: float,
    eps: float = 1e-8,
    device: Optional[DEVICE_TYPING] = "cpu",
    collate_fn: Callable = None,
    pin_memory: bool = False,
    prefetch: Optional[int] = None,
) -> PrioritizedReplayBuffer:
    """
    Helper function to create a Prioritized Replay buffer.

    Args:
        size (int): integer indicating the maximum size of the replay buffer.
        alpha (float): exponent α determines how much prioritization is used,
            with α = 0 corresponding to the uniform case.
        beta (float): importance sampling negative exponent.
        eps (float): delta added to the priorities to ensure that the buffer
            does not contain null priorities.
        device (str, int or torch.device, optional): device where to cast the
            samples.
        collate_fn (callable, optional): merges a list of samples to form a
            mini-batch of Tensor(s)/outputs.  Used when using batched loading
            from a map-style dataset.
        pin_memory (bool): whether pin_memory() should be called on the rb
            samples.
        prefetch (int, optional): number of next batches to be prefetched
            using multithreading.

    Returns:
         a ReplayBuffer instance

    """
    if isinstance(device, str):
        device = torch.device(device)

    if device.type == "cuda" and collate_fn is None:
        # Postman will add batch_dim for uploaded data, so using cat instead of
        # stack here.
        collate_fn = functools.partial(cat_fields_to_device, device=device)

    return PrioritizedReplayBuffer(
        size, alpha, beta, eps, collate_fn, pin_memory, prefetch
    )


class InPlaceSampler:
    def __init__(self, device: Optional[DEVICE_TYPING] = None):
        self.out = None
        if device is None:
            device = "cpu"
        self.device = torch.device(device)

    def __call__(self, list_of_tds):
        if self.out is None:
            self.out = torch.stack(list_of_tds, 0).contiguous()
            if self.device is not None:
                self.out = self.out.to(self.device)
        else:
            torch.stack(list_of_tds, 0, out=self.out)
        return self.out
