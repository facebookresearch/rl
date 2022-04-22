# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from numbers import Number

import functorch
import pytest
import torch
from _utils_internal import get_available_devices
from torch import nn, distributions as D
from torchrl.data import TensorDict
from torchrl.data.tensor_specs import (
    OneHotDiscreteTensorSpec,
    CustomNdOneHotDiscreteTensorSpec,
)
from torchrl.modules import (
    QValueActor,
    ActorValueOperator,
    TDModule,
    ValueOperator,
    ProbabilisticActor,
)
from torchrl.modules.distributions import OneHotCategorical
from torchrl.modules.models import NoisyLinear, MLP, NoisyLazyLinear, MaskedLogitPolicy


@pytest.mark.parametrize("in_features", [3, 10, None])
@pytest.mark.parametrize("out_features", [3, (3, 10)])
@pytest.mark.parametrize("depth, num_cells", [(3, 32), (None, (32, 32, 32))])
@pytest.mark.parametrize("activation_kwargs", [{"inplace": True}, {}])
@pytest.mark.parametrize(
    "norm_class, norm_kwargs",
    [(nn.LazyBatchNorm1d, {}), (nn.BatchNorm1d, {"num_features": 32})],
)
@pytest.mark.parametrize("bias_last_layer", [True, False])
@pytest.mark.parametrize("single_bias_last_layer", [True, False])
@pytest.mark.parametrize("layer_class", [nn.Linear, NoisyLinear])
@pytest.mark.parametrize("device", get_available_devices())
def test_mlp(
    in_features,
    out_features,
    depth,
    num_cells,
    activation_kwargs,
    bias_last_layer,
    norm_class,
    norm_kwargs,
    single_bias_last_layer,
    layer_class,
    device,
    seed=0,
):
    torch.manual_seed(seed)
    batch = 2
    mlp = MLP(
        in_features=in_features,
        out_features=out_features,
        depth=depth,
        num_cells=num_cells,
        activation_class=nn.ReLU,
        activation_kwargs=activation_kwargs,
        norm_class=norm_class,
        norm_kwargs=norm_kwargs,
        bias_last_layer=bias_last_layer,
        single_bias_last_layer=False,
        layer_class=layer_class,
    ).to(device)
    if in_features is None:
        in_features = 5
    x = torch.randn(batch, in_features, device=device)
    y = mlp(x)
    out_features = [out_features] if isinstance(out_features, Number) else out_features
    assert y.shape == torch.Size([batch, *out_features])


@pytest.mark.parametrize(
    "layer_class",
    [
        NoisyLinear,
        NoisyLazyLinear,
    ],
)
@pytest.mark.parametrize("device", get_available_devices())
def test_noisy(layer_class, device, seed=0):
    torch.manual_seed(seed)
    layer = layer_class(3, 4).to(device)
    x = torch.randn(10, 3, device=device)
    y1 = layer(x)
    layer.reset_noise()
    y2 = layer(x)
    y3 = layer(x)
    torch.testing.assert_allclose(y2, y3)
    with pytest.raises(AssertionError):
        torch.testing.assert_allclose(y1, y2)


@pytest.mark.parametrize("device", get_available_devices())
def test_value_based_policy(device):
    torch.manual_seed(0)
    obs_dim = 4
    action_dim = 5
    action_spec = OneHotDiscreteTensorSpec(action_dim)

    def make_net():
        net = MLP(in_features=obs_dim, out_features=action_dim, depth=2)
        for mod in net.modules():
            if hasattr(mod, "bias") and mod.bias is not None:
                mod.bias.data.zero_()
        return net

    actor = QValueActor(spec=action_spec, module=make_net(), safe=True).to(device)
    obs = torch.zeros(2, obs_dim, device=device)
    td = TensorDict(batch_size=[2], source={"observation": obs})
    action = actor(td).get("action")
    assert (action.sum(-1) == 1).all()

    actor = QValueActor(spec=action_spec, module=make_net(), safe=False).to(device)
    obs = torch.randn(2, obs_dim, device=device)
    td = TensorDict(batch_size=[2], source={"observation": obs})
    action = actor(td).get("action")
    assert (action.sum(-1) == 1).all()

    actor = QValueActor(spec=action_spec, module=make_net(), safe=False).to(device)
    obs = torch.zeros(2, obs_dim, device=device)
    td = TensorDict(batch_size=[2], source={"observation": obs})
    action = actor(td).get("action")
    with pytest.raises(AssertionError):
        assert (action.sum(-1) == 1).all()


@pytest.mark.parametrize("device", get_available_devices())
def test_actorcritic(device):
    common_module = TDModule(
        spec=None, module=nn.Linear(3, 4), in_keys=["obs"], out_keys=["hidden"]
    ).to(device)
    policy_operator = ProbabilisticActor(
        spec=None, module=nn.Linear(4, 5), in_keys=["hidden"], return_log_prob=True
    ).to(device)
    value_operator = ValueOperator(nn.Linear(4, 1), in_keys=["hidden"]).to(device)
    op = ActorValueOperator(
        common_operator=common_module,
        policy_operator=policy_operator,
        value_operator=value_operator,
    ).to(device)
    td = TensorDict(
        source={"obs": torch.randn(4, 3)},
        batch_size=[
            4,
        ],
    ).to(device)
    td_total = op(td.clone())
    policy_op = op.get_policy_operator()
    td_policy = policy_op(td.clone())
    value_op = op.get_value_operator()
    td_value = value_op(td)
    torch.testing.assert_allclose(td_total.get("action"), td_policy.get("action"))
    torch.testing.assert_allclose(
        td_total.get("action_log_prob"), td_policy.get("action_log_prob")
    )
    torch.testing.assert_allclose(
        td_total.get("state_value"), td_value.get("state_value")
    )

    value_params = set(
        list(op.get_value_operator().parameters()) + list(op.module[0].parameters())
    )
    value_params2 = set(value_op.parameters())
    assert len(value_params.difference(value_params2)) == 0 and len(
        value_params.intersection(value_params2)
    ) == len(value_params)

    policy_params = set(
        list(op.get_policy_operator().parameters()) + list(op.module[0].parameters())
    )
    policy_params2 = set(policy_op.parameters())
    assert len(policy_params.difference(policy_params2)) == 0 and len(
        policy_params.intersection(policy_params2)
    ) == len(policy_params)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("functional", [True, False])
@pytest.mark.parametrize(
    "mask", [torch.tensor([False, True, False, True, False]), "random"]
)
def test_maskedlogit(device, functional, mask):
    batch = 10
    torch.manual_seed(0)
    if isinstance(mask, str):
        random_mask = True
        mask = torch.zeros(batch, 5, dtype=torch.bool, device=device).bernoulli_()
    else:
        random_mask = False
        mask = mask.to(device)
    policy_net = nn.Linear(3, 5, bias=False).to(device)  # model that returns logits

    policy_net_wrapped = MaskedLogitPolicy(policy_net)
    if functional:
        policy_net_wrapped, params = functorch.make_functional(policy_net_wrapped)
    observation = torch.randn(batch, 3, device=device)
    if functional:
        logits_masked = policy_net_wrapped(params, observation, mask)
    else:
        logits_masked = policy_net_wrapped(observation, mask)
    c = D.Categorical(logits=logits_masked)
    samples = c.sample((1000,))
    samples_uniques = samples.unique()
    if random_mask:
        mask_expand = mask.expand(1000, batch, 5)
        assert mask_expand.gather(-1, samples.unsqueeze(-1)).all()
    else:
        assert ((samples_uniques == 1) | (samples_uniques == 3)).all()

    # test synergy with CustomNdOneHotDiscreteTensorSpec
    spec = CustomNdOneHotDiscreteTensorSpec(mask=mask)
    c = OneHotCategorical(logits=logits_masked)
    assert spec.is_in(c.sample((1000,)))
    c = OneHotCategorical(logits=torch.randn_like(logits_masked))
    with pytest.raises(AssertionError):
        assert spec.is_in(c.sample((1000,)))


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
