# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from copy import copy, deepcopy
from typing import List, Iterable, Union, Tuple

import functorch
import torch
from torch import nn, Tensor

from torchrl.data import (
    TensorSpec,
    CompositeSpec,
)
from torchrl.data.tensordict.tensordict import _TensorDict
from torchrl.modules.tensordict_module.common import TensorDictModule
from torchrl.modules.tensordict_module.probabilistic import (
    ProbabilisticTensorDictModule,
)

__all__ = ["TensorDictSequence"]


class TensorDictSequence(TensorDictModule):
    """
    A sequence of TDModules.
    Similarly to `nn.Sequence` which passes a tensor through a chain of mappings that read and write a single tensor
    each, this module will read and write over a tensordict by querying each of the input modules.
    When calling a `TDSequence` instance with a functional module, it is expected that the parameter lists (and
    buffers) will be concatenated in a single list.

    Args:
         modules (iterable of TDModules): ordered sequence of TDModule instances to be run sequentially.

    TDSequence supportse functional, modular and vmap coding:
    Examples:
        >>> from torchrl.modules.tensordict_module import ProbabilisticTensorDictModule
        >>> from torchrl.data import TensorDict, NdUnboundedContinuousTensorSpec
        >>> from torchrl.modules import  TanhNormal, TensorDictSequence, NormalParamWrapper
        >>> import torch, functorch
        >>> td = TensorDict({"input": torch.randn(3, 4)}, [3,])
        >>> spec1 = NdUnboundedContinuousTensorSpec(4)
        >>> net1 = NormalParamWrapper(torch.nn.Linear(4, 8))
        >>> fnet1, params1, buffers1 = functorch.make_functional_with_buffers(net1)
        >>> fmodule1 = TensorDictModule(fnet1, in_keys=["input"], out_keys=["loc", "scale"])
        >>> td_module1 = ProbabilisticTensorDictModule(
        ...    module=fmodule1,
        ...    spec=spec1,
        ...    dist_param_keys=["loc", "scale"],
        ...    out_key_sample=["hidden"],
        ...    distribution_class=TanhNormal,
        ...    return_log_prob=True,
        ...    )
        >>> spec2 = NdUnboundedContinuousTensorSpec(8)
        >>> module2 = torch.nn.Linear(4, 8)
        >>> fmodule2, params2, buffers2 = functorch.make_functional_with_buffers(module2)
        >>> td_module2 = TensorDictModule(
        ...    module=fmodule2,
        ...    spec=spec2,
        ...    in_keys=["hidden"],
        ...    out_keys=["output"],
        ...    )
        >>> td_module = TensorDictSequence(td_module1, td_module2)
        >>> params = params1 + params2
        >>> buffers = buffers1 + buffers2
        >>> _ = td_module(td, params=params, buffers=buffers)
        >>> print(td)
        TensorDict(
            fields={
                input: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                loc: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                scale: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                hidden: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                sample_log_prob: Tensor(torch.Size([3, 1]), dtype=torch.float32),
                output: Tensor(torch.Size([3, 8]), dtype=torch.float32)},
            batch_size=torch.Size([3]),
            device=cpu,
            is_shared=False)

        >>> # The module spec aggregates all the input specs:
        >>> print(td_module.spec)
        CompositeSpec(
            hidden: NdUnboundedContinuousTensorSpec(
                 shape=torch.Size([4]),space=None,device=cpu,dtype=torch.float32,domain=continuous),
            output: NdUnboundedContinuousTensorSpec(
                 shape=torch.Size([8]),space=None,device=cpu,dtype=torch.float32,domain=continuous))

    In the vmap case:
        >>> params = tuple(p.expand(4, *p.shape).contiguous().normal_() for p in params)
        >>> buffers = tuple(b.expand(4, *b.shape).contiguous().normal_() for p in buffers)
        >>> td_vmap = td_module(td, params=params, buffers=buffers, vmap=True)
        >>> print(td_vmap)
        TensorDict(
            fields={
                input: Tensor(torch.Size([4, 3, 4]), dtype=torch.float32),
                loc: Tensor(torch.Size([4, 3, 4]), dtype=torch.float32),
                scale: Tensor(torch.Size([4, 3, 4]), dtype=torch.float32),
                hidden: Tensor(torch.Size([4, 3, 4]), dtype=torch.float32),
                sample_log_prob: Tensor(torch.Size([4, 3, 1]), dtype=torch.float32),
                output: Tensor(torch.Size([4, 3, 8]), dtype=torch.float32)},
            batch_size=torch.Size([4, 3]),
            device=cpu,
            is_shared=False)


    """

    module: nn.ModuleList

    def __init__(
        self,
        *modules: TensorDictModule,
    ):
        in_keys = []
        out_keys = []
        for module in modules:
            # we sometimes use in_keys to select keys of a tensordict that are
            # necessary to run a TensorDictModule. If a key is an intermediary in
            # the chain, there is no reason why it should belong to the input
            # TensorDict.
            in_keys += [key for key in module.in_keys if key not in out_keys + in_keys]
            out_keys += module.out_keys

        out_keys = [
            out_key
            for i, out_key in enumerate(out_keys)
            if out_key not in out_keys[i + 1 :]
        ]

        super().__init__(
            spec=None,
            module=nn.ModuleList(list(modules)),
            in_keys=in_keys,
            out_keys=out_keys,
        )

    @staticmethod
    def _find_functional_module(module: TensorDictModule) -> nn.Module:
        fmodule = module
        while not isinstance(
            fmodule, (functorch.FunctionalModule, functorch.FunctionalModuleWithBuffers)
        ):
            try:
                fmodule = fmodule.module
            except AttributeError:
                raise AttributeError(
                    f"couldn't find a functional module in module of type {type(module)}"
                )
        return fmodule

    @property
    def num_params(self):
        return self.param_len[-1]

    @property
    def num_buffers(self):
        return self.buffer_len[-1]

    @property
    def param_len(self) -> List[int]:
        param_list = []
        prev = 0
        for module in self.module:
            param_list.append(module.num_params + prev)
            prev = param_list[-1]
        return param_list

    @property
    def buffer_len(self) -> List[int]:
        buffer_list = []
        prev = 0
        for module in self.module:
            buffer_list.append(module.num_buffers + prev)
            prev = buffer_list[-1]
        return buffer_list

    def _split_param(
        self, param_list: Iterable[Tensor], params_or_buffers: str
    ) -> Iterable[Iterable[Tensor]]:
        if params_or_buffers == "params":
            list_out = self.param_len
        elif params_or_buffers == "buffers":
            list_out = self.buffer_len
        list_in = [0] + list_out[:-1]
        out = []
        for a, b in zip(list_in, list_out):
            out.append(param_list[a:b])
        return out

    def forward(
        self, tensordict: _TensorDict, tensordict_out=None, **kwargs
    ) -> _TensorDict:
        if "params" in kwargs and "buffers" in kwargs:
            param_splits = self._split_param(kwargs["params"], "params")
            buffer_splits = self._split_param(kwargs["buffers"], "buffers")
            kwargs_pruned = {
                key: item
                for key, item in kwargs.items()
                if key not in ("params", "buffers")
            }
            for i, (module, param, buffer) in enumerate(
                zip(self.module, param_splits, buffer_splits)
            ):
                if "vmap" in kwargs_pruned and i > 0:
                    # the tensordict is already expended
                    kwargs_pruned["vmap"] = (0, 0, *(0,) * len(module.in_keys))
                tensordict = module(
                    tensordict, params=param, buffers=buffer, **kwargs_pruned
                )

        elif "params" in kwargs:
            param_splits = self._split_param(kwargs["params"], "params")
            kwargs_pruned = {
                key: item for key, item in kwargs.items() if key not in ("params",)
            }
            for i, (module, param) in enumerate(zip(self.module, param_splits)):
                if "vmap" in kwargs_pruned and i > 0:
                    # the tensordict is already expended
                    kwargs_pruned["vmap"] = (0, *(0,) * len(module.in_keys))
                tensordict = module(tensordict, params=param, **kwargs_pruned)

        elif not len(kwargs):
            for module in self.module:
                tensordict = module(tensordict)
        else:
            if any(
                str(parentclass) == "<class 'higher.patch._MonkeyPatchBase'>"
                for parentclass in self.__class__.__bases__
            ):
                return self.forward(tensordict)
            raise RuntimeError(
                "TensorDictSequence does not support keyword arguments other than 'tensordict_out', 'params', 'buffers' and 'vmap'"
            )
        if tensordict_out is not None:
            tensordict_out.update(tensordict, inplace=True)
            return tensordict_out
        return tensordict

    def __len__(self):
        return len(self.module)

    def __getitem__(self, index: Union[int, slice]) -> TensorDictModule:
        return self.module.__getitem__(index)

    def __setitem__(self, index: int, tensordict_module: TensorDictModule) -> None:
        return self.module.__setitem__(idx=index, module=tensordict_module)

    def __delitem__(self, index: Union[int, slice]) -> None:
        self.module.__delitem__(idx=index)

    @property
    def spec(self):
        kwargs = {}
        for layer in self.module:
            out_key = layer.out_keys[0]
            spec = layer.spec
            if spec is not None and not isinstance(spec, TensorSpec):
                raise RuntimeError(
                    f"TensorDictSequence.spec requires all specs to be valid TensorSpec objects. Got "
                    f"{type(layer.spec)}"
                )
            if isinstance(spec, CompositeSpec):
                kwargs.update(spec._specs)
            else:
                kwargs[out_key] = spec
        return CompositeSpec(**kwargs)

    def make_functional_with_buffers(self, clone: bool = True):
        """
        Transforms a stateful module in a functional module and returns its parameters and buffers.
        Unlike functorch.make_functional_with_buffers, this method supports lazy modules.

        Returns:
            A tuple of parameter and buffer tuples

        Examples:
            >>> from torchrl.data import NdUnboundedContinuousTensorSpec, TensorDict
            >>> lazy_module1 = nn.LazyLinear(4)
            >>> lazy_module2 = nn.LazyLinear(3)
            >>> spec1 = NdUnboundedContinuousTensorSpec(18)
            >>> spec2 = NdUnboundedContinuousTensorSpec(4)
            >>> td_module1 = TensorDictModule(spec=spec1, module=lazy_module1, in_keys=["some_input"], out_keys=["hidden"])
            >>> td_module2 = TensorDictModule(spec=spec2, module=lazy_module2, in_keys=["hidden"], out_keys=["some_output"])
            >>> td_module = TensorDictSequence(td_module1, td_module2)
            >>> _, (params, buffers) = td_module.make_functional_with_buffers()
            >>> print(params[0].shape) # the lazy module has been initialized
            torch.Size([4, 18])
            >>> print(td_module(
            ...    TensorDict({'some_input': torch.randn(18)}, batch_size=[]),
            ...    params=params,
            ...    buffers=buffers))
            TensorDict(
                fields={
                    some_input: Tensor(torch.Size([18]), dtype=torch.float32),
                    hidden: Tensor(torch.Size([4]), dtype=torch.float32),
                    some_output: Tensor(torch.Size([3]), dtype=torch.float32)},
                batch_size=torch.Size([]),
                device=cpu,
                is_shared=False)

        """
        if clone:
            self_copy = deepcopy(self)
            self_copy.module = copy(self_copy.module)
        else:
            self_copy = self
        params = []
        buffers = []
        for i, module in enumerate(self.module):
            self_copy.module[i], (
                _params,
                _buffers,
            ) = module.make_functional_with_buffers(clone=True)
            params.extend(_params)
            buffers.extend(_buffers)
        return self_copy, (params, buffers)

    def get_dist(
        self,
        tensordict: _TensorDict,
        **kwargs,
    ) -> Tuple[torch.distributions.Distribution, ...]:
        L = len(self.module)

        if isinstance(self.module[-1], ProbabilisticTensorDictModule):
            if (
                "params" in kwargs
                and "buffers" in kwargs
                and kwargs["params"] is not None
            ):
                param_splits = self._split_param(kwargs["params"], "params")
                buffer_splits = self._split_param(kwargs["buffers"], "buffers")
                kwargs_pruned = {
                    key: item
                    for key, item in kwargs.items()
                    if key not in ("params", "buffers")
                }
                for i, (module, param, buffer) in enumerate(
                    zip(self.module, param_splits, buffer_splits)
                ):
                    if "vmap" in kwargs_pruned and i > 0:
                        # the tensordict is already expended
                        kwargs_pruned["vmap"] = (0, 0, *(0,) * len(module.in_keys))
                    if i < L - 1:
                        tensordict = module(
                            tensordict, params=param, buffers=buffer, **kwargs_pruned
                        )
                    else:
                        out = module.get_dist(
                            tensordict, params=param, buffers=buffer, **kwargs_pruned
                        )

            elif "params" in kwargs and kwargs["params"] is not None:
                param_splits = self._split_param(kwargs["params"], "params")
                kwargs_pruned = {
                    key: item for key, item in kwargs.items() if key not in ("params",)
                }
                for i, (module, param) in enumerate(zip(self.module, param_splits)):
                    if "vmap" in kwargs_pruned and i > 0:
                        # the tensordict is already expended
                        kwargs_pruned["vmap"] = (0, *(0,) * len(module.in_keys))
                    if i < L - 1:
                        tensordict = module(tensordict, params=param, **kwargs_pruned)
                    else:
                        out = module.get_dist(tensordict, params=param, **kwargs_pruned)

            elif not len(kwargs) or kwargs["params"] is None:
                for i, module in enumerate(self.module):
                    if i < L - 1:
                        tensordict = module(tensordict)
                    else:
                        out = module.get_dist(tensordict)
            else:
                raise RuntimeError(
                    "TensorDictSequence does not support keyword arguments other than 'params', 'buffers' and 'vmap'"
                )

            return out
        else:
            raise RuntimeError(
                "Cannot call get_dist on a sequence of tensordicts that does not end with a probabilistic TensorDict"
            )
