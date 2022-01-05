from numbers import Number
from typing import Union, Iterable

import numpy as np
import torch
from torch import distributions as D
from torch.distributions import constraints

from torchrl.modules.utils import mappings
from .utils import UNIFORM

__all__ = ["TanhNormal", "Delta", "TanhDelta"]


class SafeTanhTransform(D.TanhTransform):
    def _call(self, x: torch.Tensor) -> torch.Tensor:
        y = super()._call(x)
        y = y.clamp(-1 + 1e-4, 1 - 1e-4)
        return y


class TanhNormal(D.TransformedDistribution):
    """
    Implements a TanhNormal distribution with location scaling
    """

    arg_constraints = {
        "loc": constraints.real,
        "scale": constraints.greater_than(1e-6),
    }

    def __init__(
            self,
            net_output: torch.Tensor,
            upscale: Union[torch.Tensor, Number] = 5.0,
            min: Union[torch.Tensor, Number] = -1.0,
            max: Union[torch.Tensor, Number] = 1.0,
            scale_mapping: str = "biased_softplus_1.0",
            event_dims: int = 1,
            tanh_loc: bool = True,
            tanh_scale: bool = True,
    ):
        err_msg = "TanhNormal max values must be strictly greater than min values"
        if isinstance(max, torch.Tensor) or isinstance(min, torch.Tensor):
            if not (max > min).all():
                raise RuntimeError(err_msg)
        elif isinstance(max, Number) and isinstance(min, Number):
            if not max > min:
                raise RuntimeError(err_msg)
        else:
            if not all(max > min):
                raise RuntimeError(err_msg)

        loc, scale = net_output.chunk(chunks=2, dim=-1)
        self.tanh_loc = tanh_loc
        if tanh_loc:
            if upscale != 1.0:
                loc = loc / upscale
            loc = loc.tanh() * upscale
        if tanh_scale:
            if upscale != 1.0:
                scale = scale / upscale
            scale = scale.tanh() * upscale

        loc = loc + (max - min) / 2 + min

        self.loc = loc
        self.scale = mappings(scale_mapping)(scale)
        self.upscale = upscale

        t = SafeTanhTransform()
        non_trivial_min = (isinstance(min, torch.Tensor) and (min != 1.0).any()) or (
                not isinstance(min, torch.Tensor) and min != 1.0
        )
        non_trivial_max = (isinstance(max, torch.Tensor) and (max != 1.0).any()) or (
                not isinstance(max, torch.Tensor) and max != 1.0
        )
        if non_trivial_max or non_trivial_min:
            t = D.ComposeTransform(
                [t, D.AffineTransform(loc=(max + min) / 2, scale=(max - min) / 2)]
            )
        try:
            base = D.Independent(D.Normal(self.loc, self.scale), event_dims)
        except:
            self.loc
            self.scale

        super().__init__(base, t)


def uniform_sample_tanhnormal(dist: TanhNormal, size=torch.Size([])) -> torch.Tensor:
    return torch.rand_like(dist.sample(size)) * (dist.max - dist.min) + dist.min


UNIFORM[TanhNormal] = uniform_sample_tanhnormal


class Delta(D.Distribution):
    arg_constraints = {}

    def __init__(
            self,
            param: torch.Tensor,
            atol: Number = 1e-6,
            rtol: Number = 1e-6,
            batch_shape: Union[torch.Size, Iterable] = torch.Size([]),
            event_shape: Union[torch.Size, Iterable] = torch.Size([]),
    ):
        self.param = param
        self.atol = atol
        self.rtol = rtol
        if not len(batch_shape) and not len(event_shape):
            batch_shape = param.shape[:-1]
            event_shape = param.shape[-1:]
        super().__init__(batch_shape=batch_shape, event_shape=event_shape)

    def _is_equal(self, value: torch.Tensor) -> torch.Tensor:
        param = self.param.expand_as(value)
        is_equal = abs(value - param) < self.atol + self.rtol * abs(param)
        for i in range(-1, -len(self.event_shape) - 1, -1):
            is_equal = is_equal.all(i)
        return is_equal

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        is_equal = self._is_equal(value)
        out = torch.zeros_like(is_equal, dtype=value.dtype)
        out.masked_fill_(is_equal, np.inf)
        out.masked_fill_(~is_equal, -np.inf)
        return out

    @torch.no_grad()
    def sample(self, size=torch.Size([])) -> torch.Tensor:
        return self.param.expand(*size, *self.param.shape)

    def rsample(self, size=torch.Size([])) -> torch.Tensor:
        return self.param.expand(*size, *self.param.shape)

    @property
    def mode(self) -> torch.Tensor:
        return self.param

    @property
    def mean(self) -> torch.Tensor:
        return self.param


class TanhDelta(D.TransformedDistribution):
    """
    Implements a Tanh transformed Delta distribution with location scaling
    """

    arg_constraints = {
        "loc": constraints.real,
    }

    def __init__(
            self,
            net_output: torch.Tensor,
            min: Union[torch.Tensor, Number] = -1.0,
            max: Union[torch.Tensor, Number] = 1.0,
            event_dims: int = 1,
            atol: Number = 1e-4,
            rtol: Number = 1e-4,
            **kwargs,
    ):
        minmax_msg = "max value has been found to be equal or less than min value"
        if isinstance(max, torch.Tensor) or isinstance(min, torch.Tensor):
            if not (max > min).all():
                raise ValueError(minmax_msg)
        elif isinstance(max, Number) and isinstance(min, Number):
            if max <= min:
                raise ValueError(minmax_msg)
        else:
            if not all(max > min):
                raise ValueError(minmax_msg)

        loc = net_output
        loc = loc + (max - min) / 2 + min

        self.loc = loc

        t = D.TanhTransform()
        non_trivial_min = (isinstance(min, torch.Tensor) and (min != 1.0).any()) or (
                not isinstance(min, torch.Tensor) and min != 1.0
        )
        non_trivial_max = (isinstance(max, torch.Tensor) and (max != 1.0).any()) or (
                not isinstance(max, torch.Tensor) and max != 1.0
        )
        if non_trivial_max or non_trivial_min:
            t = D.ComposeTransform(
                [t, D.AffineTransform(loc=(max + min) / 2, scale=(max - min) / 2)]
            )
        event_shape = net_output.shape[-event_dims:]
        batch_shape = net_output.shape[:-event_dims]
        base = Delta(loc, atol=atol, rtol=rtol, batch_shape=batch_shape, event_shape=event_shape, **kwargs)

        super().__init__(base, t)

    @property
    def mode(self) -> torch.Tensor:
        mode = self.base_dist.param
        for t in self.transforms:
            mode = t(mode)
        return mode

    @property
    def mean(self) -> torch.Tensor:
        raise AttributeError("TanhDelta mean has not analytical form.")


def uniform_sample_delta(dist: Delta, size=torch.Size([])) -> torch.Tensor:
    return torch.randn_like(dist.sample(size))


UNIFORM[Delta] = uniform_sample_delta
