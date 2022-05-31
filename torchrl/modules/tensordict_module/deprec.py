# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# implements a function that takes a sequence of returns and multiply its by the policy log_prob to get a differentiable objective

from __future__ import annotations

from _warnings import warn
from copy import deepcopy
from typing import Union, Callable, Sequence, Type, Optional, Tuple

import torch
from torch import Tensor, nn, distributions as d

from torchrl.data import TensorSpec, DEVICE_TYPING
from torchrl.data.tensordict.tensordict import _TensorDict
from torchrl.envs.utils import exploration_mode
from torchrl.modules import TensorDictModule, Delta, distributions_maps


class ProbabilisticTDModule(TensorDictModule):
    """
    DEPRECATED

    A probabilistic TD Module.
    ProbabilisticTDModule is a special case of a TensorDictModule where the output is sampled given some rule, specified by
    the input `default_interaction_mode` argument and the `exploration_mode()` global function.

    A ProbabilisticTDModule instance has two main features:
    - It reads and writes TensorDict objects
    - It uses a real mapping R^n -> R^m to create a distribution in R^d from which values can be sampled or computed.
    When the __call__ / forward method is called, a distribution is created, and a value computed (using the 'mean',
    'mode', 'median' attribute or the 'rsample', 'sample' method).

    By default, ProbabilisticTDModule distribution class is a Delta distribution, making ProbabilisticTDModule a
    simple wrapper around a deterministic mapping function (i.e. it can be used interchangeably with its parent
    TensorDictModule).

    Args:
        module (nn.Module): a nn.Module used to map the input to the output parameter space. Can be a functional
            module (FunctionalModule or FunctionalModuleWithBuffers), in which case the `forward` method will expect
            the params (and possibly) buffers keyword arguments.
        spec (TensorSpec): specs of the first output tensor. Used when calling td_module.random() to generate random
            values in the target space.
        in_keys (iterable of str): keys to be read from input tensordict and passed to the module. If it
            contains more than one element, the values will be passed in the order given by the in_keys iterable.
        out_keys (iterable of str): keys to be written to the input tensordict. The length of out_keys must match the
            number of tensors returned by the distribution sampling method plus the extra tensors returned by the
            module.
        distribution_class (Type, optional): a torch.distributions.Distribution class to be used for sampling.
            Default is Delta.
        distribution_kwargs (dict, optional): kwargs to be passed to the distribution.
        default_interaction_mode (str, optional): default method to be used to retrieve the output value. Should be one of:
            'mode', 'median', 'mean' or 'random' (in which case the value is sampled randomly from the distribution).
            Default is 'mode'.
            Note: When a sample is drawn, the `ProbabilisticTDModule` instance will fist look for the interaction mode
            dictated by the `exploration_mode()` global function. If this returns `None` (its default value),
            then the `default_interaction_mode` of the `ProbabilisticTDModule` instance will be used.
            Note that DataCollector instances will use `set_exploration_mode` to `"random"` by default.
        return_log_prob (bool, optional): if True, the log-probability of the distribution sample will be written in the
            tensordict with the key `f'{in_keys[0]}_log_prob'`. Default is `False`.
        safe (bool, optional): if True, the value of the sample is checked against the input spec. Out-of-domain sampling can
            occur because of exploration policies or numerical under/overflow issues. As for the `spec` argument,
            this check will only occur for the distribution sample, but not the other tensors returned by the input
            module. If the sample is out of bounds, it is projected back onto the desired space using the
            `TensorSpec.project`
            method.
            Default is `False`.
        save_dist_params (bool, optional): if True, the parameters of the distribution (i.e. the output of the module)
            will be written to the tensordict along with the sample. Those parameters can be used to
            re-compute the original distribution later on (e.g. to compute the divergence between the distribution
            used to sample the action and the updated distribution in PPO).
            Default is `False`.
        cache_dist (bool, optional): if True, the parameters of the distribution (i.e. the output of the module)
            will be written to the tensordict along with the sample. Those parameters can be used to
            re-compute the original distribution later on (e.g. to compute the divergence between the distribution
            used to sample the action and the updated distribution in PPO).
            Default is `False`.

    Examples:
        >>> from torchrl.data import TensorDict, NdUnboundedContinuousTensorSpec
        >>> from torchrl.modules import TanhNormal
        >>> import functorch, torch
        >>> td = TensorDict({"input": torch.randn(3, 4), "hidden": torch.randn(3, 8)}, [3,])
        >>> spec = NdUnboundedContinuousTensorSpec(4)
        >>> module = torch.nn.GRUCell(4, 8)
        >>> module_func, params, buffers = functorch.make_functional_with_buffers(module)
        >>> td_module = ProbabilisticTDModule(
        ...    module=module_func,
        ...    spec=spec,
        ...    in_keys=["input"],
        ...    out_keys=["output"],
        ...    distribution_class=TanhNormal,
        ...    return_log_prob=True,
        ...    )
        >>> _ = td_module(td, params=params, buffers=buffers)
        >>> print(td)
        TensorDict(
            fields={
                input: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                hidden: Tensor(torch.Size([3, 8]), dtype=torch.float32),
                output: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                output_log_prob: Tensor(torch.Size([3, 1]), dtype=torch.float32)},
            batch_size=torch.Size([3]),
            device=cpu,
            is_shared=False)

        >>> # In the vmap case, the tensordict is again expended to match the batch:
        >>> params = tuple(p.expand(4, *p.shape).contiguous().normal_() for p in params)
        >>> buffers = tuple(b.expand(4, *b.shape).contiguous().normal_() for p in buffers)
        >>> td_vmap = td_module(td, params=params, buffers=buffers, vmap=True)
        >>> print(td_vmap)
        TensorDict(
            fields={
                input: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                hidden: Tensor(torch.Size([3, 8]), dtype=torch.float32),
                output: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                output_log_prob: Tensor(torch.Size([3, 1]), dtype=torch.float32)},
            batch_size=torch.Size([3]),
            device=cpu,
            is_shared=False)

    """

    def __init__(
        self,
        module: Union[Callable[[Tensor], Tensor], nn.Module],
        spec: TensorSpec,
        in_keys: Sequence[str],
        out_keys: Sequence[str],
        distribution_class: Type = Delta,
        distribution_kwargs: Optional[dict] = None,
        default_interaction_mode: str = "mode",
        _n_empirical_est: int = 1000,
        return_log_prob: bool = False,
        safe: bool = False,
        save_dist_params: bool = False,
        cache_dist: bool = False,
    ):
        warn(
            "ProbabilisticTDModule will be deprecated soon, consider using ProbabilisticTensorDictModule instead."
        )
        super().__init__(
            spec=spec,
            module=module,
            out_keys=out_keys,
            in_keys=in_keys,
            safe=safe,
        )

        self.save_dist_params = save_dist_params
        self._n_empirical_est = _n_empirical_est
        self.cache_dist = cache_dist if hasattr(distribution_class, "update") else False
        self._dist = None

        if isinstance(distribution_class, str):
            distribution_class = distributions_maps.get(distribution_class.lower())
        self.distribution_class = distribution_class
        self.distribution_kwargs = (
            distribution_kwargs if distribution_kwargs is not None else dict()
        )
        self.return_log_prob = return_log_prob

        self.default_interaction_mode = default_interaction_mode
        self.interact = False

    def get_dist(
        self,
        tensordict: _TensorDict,
        **kwargs,
    ) -> Tuple[torch.distributions.Distribution, ...]:
        """Calls the module using the tensors retrieved from the 'in_keys' attribute and returns a distribution
        using its output.

        Args:
            tensordict (_TensorDict): tensordict with the input values for the creation of the distribution.

        Returns:
            a distribution along with other tensors returned by the module.

        """
        tensors = [tensordict.get(key, None) for key in self.in_keys]
        out_tensors = self._call_module(tensors, **kwargs)
        if isinstance(out_tensors, Tensor):
            out_tensors = (out_tensors,)
        if self.save_dist_params:
            for i, _tensor in enumerate(out_tensors):
                tensordict.set(f"{self.out_keys[0]}_dist_param_{i}", _tensor)
        dist, num_params = self.build_dist_from_params(out_tensors)
        tensors = out_tensors[num_params:]

        return (dist, *tensors)

    def build_dist_from_params(
        self, params: Tuple[Tensor, ...]
    ) -> Tuple[d.Distribution, int]:
        """Given a tuple of temsors, returns a distribution object and the number of parameters used for it.

        Args:
            params (Tuple[Tensor, ...]): tensors to be used for the distribution construction.

        Returns:
            a distribution object and the number of parameters used for its construction.

        """
        num_params = (
            getattr(self.distribution_class, "num_params")
            if hasattr(self.distribution_class, "num_params")
            else 1
        )
        if self.cache_dist and self._dist is not None:
            self._dist.update(*params[:num_params])
            dist = self._dist
        else:
            dist = self.distribution_class(
                *params[:num_params], **self.distribution_kwargs
            )
            if self.cache_dist:
                self._dist = dist
        return dist, num_params

    def forward(
        self,
        tensordict: _TensorDict,
        tensordict_out: Optional[_TensorDict] = None,
        **kwargs,
    ) -> _TensorDict:

        dist, *tensors = self.get_dist(tensordict, **kwargs)
        out_tensor = self._dist_sample(
            dist, *tensors, interaction_mode=exploration_mode()
        )
        tensordict_out = self._write_to_tensordict(
            tensordict,
            [out_tensor] + list(tensors),
            tensordict_out,
            vmap=kwargs.get("vmap", 0),
        )
        if self.return_log_prob:
            log_prob = dist.log_prob(out_tensor)
            tensordict_out.set("_".join([self.out_keys[0], "log_prob"]), log_prob)
        return tensordict_out

    def log_prob(self, tensordict: _TensorDict, **kwargs) -> _TensorDict:
        """
        Samples/computes an action using the module and writes this value onto the input tensordict along
        with its log-probability.

        Args:
            tensordict (_TensorDict): tensordict containing the in_keys specified in the initializer.

        Returns:
            the same tensordict with the out_keys values added/updated as well as a
                f"{out_keys[0]}_log_prob" key containing the log-probability of the first output.

        """
        dist, *_ = self.get_dist(tensordict, **kwargs)
        lp = dist.log_prob(tensordict.get(self.out_keys[0]))
        tensordict.set(self.out_keys[0] + "_log_prob", lp)
        return tensordict

    def _dist_sample(
        self,
        dist: d.Distribution,
        *tensors: Tensor,
        interaction_mode: bool = None,
        eps: float = None,
    ) -> Tensor:
        if interaction_mode is None:
            interaction_mode = self.default_interaction_mode
        if not isinstance(dist, d.Distribution):
            raise TypeError(f"type {type(dist)} not recognised by _dist_sample")

        if interaction_mode == "mode":
            if hasattr(dist, "mode"):
                return dist.mode
            else:
                raise NotImplementedError(
                    f"method {type(dist)}.mode is not implemented"
                )

        elif interaction_mode == "median":
            if hasattr(dist, "median"):
                return dist.median
            else:
                raise NotImplementedError(
                    f"method {type(dist)}.median is not implemented"
                )

        elif interaction_mode == "mean":
            try:
                return dist.mean
            except AttributeError:
                if dist.has_rsample:
                    return dist.rsample((self._n_empirical_est,)).mean(0)
                else:
                    return dist.sample((self._n_empirical_est,)).mean(0)

        elif interaction_mode == "random":
            if dist.has_rsample:
                return dist.rsample()
            else:
                return dist.sample()
        elif interaction_mode == "net_output":
            if len(tensors) > 1:
                raise RuntimeError(
                    "Multiple values passed to _dist_sample when trying to return a single action "
                    "tensor."
                )
            return tensors[0]
        else:
            raise NotImplementedError(f"unknown interaction_mode {interaction_mode}")

    @property
    def device(self):
        for p in self.parameters():
            return p.device
        return torch.device("cpu")

    def to(self, dest: Union[torch.dtype, DEVICE_TYPING]) -> ProbabilisticTDModule:
        if self.spec is not None:
            self.spec = self.spec.to(dest)
        out = super().to(dest)
        return out

    def __deepcopy__(self, memodict={}):
        self._dist = None
        cls = self.__class__
        result = cls.__new__(cls)
        memodict[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memodict))
        return result

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(module={self.module}, distribution_class={self.distribution_class}, device={self.device})"


class ProbabilisticActor_deprecated(ProbabilisticTDModule):
    """
    General class for probabilistic actors in RL.
    The Actor class comes with default values for the in_keys and out_keys
    arguments (["observation"] and ["action"], respectively).

    Examples:
        >>> from torchrl.data import TensorDict, NdBoundedTensorSpec
        >>> from torchrl.modules import Actor, TanhNormal
        >>> import torch, functorch
        >>> td = TensorDict({"observation": torch.randn(3, 4)}, [3,])
        >>> action_spec = NdBoundedTensorSpec(shape=torch.Size([4]),
        ...    minimum=-1, maximum=1)
        >>> module = torch.nn.Linear(4, 8)
        >>> fmodule, params, buffers = functorch.make_functional_with_buffers(
        ...     module)
        >>> td_module = ProbabilisticActor_deprecated(
        ...    module=fmodule,
        ...    spec=action_spec,
        ...    distribution_class=TanhNormal,
        ...    )
        >>> td_module(td, params=params, buffers=buffers)
        >>> print(td.get("action"))

    """

    def __init__(
        self,
        *args,
        in_keys: Optional[Sequence[str]] = None,
        out_keys: Optional[Sequence[str]] = None,
        **kwargs,
    ):
        if in_keys is None:
            in_keys = ["observation"]
        if out_keys is None:
            out_keys = ["action"]

        super().__init__(
            *args,
            in_keys=in_keys,
            out_keys=out_keys,
            **kwargs,
        )
