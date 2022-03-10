import argparse

import pytest
import torch
from scipy.stats import ttest_1samp
from torch import nn

from torchrl.data import NdBoundedTensorSpec
from torchrl.data.tensordict.tensordict import TensorDict
from torchrl.modules import ProbabilisticActor
from torchrl.modules.distributions import TanhNormal
from torchrl.modules.td_module import Actor
from torchrl.modules.td_module.exploration import (
    _OrnsteinUhlenbeckProcess,
    OrnsteinUhlenbeckProcessWrapper,
)


def test_ou(seed=0):
    torch.manual_seed(seed)
    td = TensorDict({"action": torch.randn(3) / 10}, batch_size=[])
    ou = _OrnsteinUhlenbeckProcess(10.0, mu=2.0, x0=-4, sigma=0.1, sigma_min=0.01)

    tds = []
    for i in range(2000):
        td = ou.add_sample(td)
        tds.append(td.clone())
        td.set_("action", torch.randn(3) / 10)
        if i % 1000 == 0:
            td.zero_()

    tds = torch.stack(tds, 0)

    tset, pval_acc = ttest_1samp(tds.get("action")[950:1000, 0].numpy(), 2.0)
    tset, pval_reg = ttest_1samp(tds.get("action")[:50, 0].numpy(), 2.0)
    assert pval_acc > 0.05
    assert pval_reg < 0.1

    tset, pval_acc = ttest_1samp(tds.get("action")[1950:2000, 0].numpy(), 2.0)
    tset, pval_reg = ttest_1samp(tds.get("action")[1000:1050, 0].numpy(), 2.0)
    assert pval_acc > 0.05
    assert pval_reg < 0.1


def test_ou_wrapper(device="cpu", d_obs=4, d_act=6, batch=32, n_steps=100, seed=0):
    torch.manual_seed(seed)
    module = nn.Linear(d_obs, 2*d_act).to(device)
    action_spec = NdBoundedTensorSpec(
        -torch.ones(d_act ), torch.ones(d_act ), (d_act ,)
    )
    policy = ProbabilisticActor(
        spec=action_spec,
        module=module,
        distribution_class=TanhNormal,
        default_interaction_mode="random",
    ).to(device)
    exploratory_policy = OrnsteinUhlenbeckProcessWrapper(policy)

    tensor_dict = TensorDict(
        batch_size=[batch],
        source={"observation": torch.randn(batch, d_obs, device=device)},
        device=device,
    )
    out_noexp = []
    out = []
    for i in range(n_steps):
        tensor_dict_noexp = policy(tensor_dict.select("observation"))
        tensor_dict = exploratory_policy(tensor_dict)
        out.append(tensor_dict.clone())
        out_noexp.append(tensor_dict_noexp.clone())
        tensor_dict.set_("observation", torch.randn(batch, d_obs, device=device))
    out = torch.stack(out, 0)
    out_noexp = torch.stack(out_noexp, 0)
    assert (out_noexp.get("action") != out.get("action")).all()
    assert (out.get("action") <= 1.0).all(), out.get("action").min()
    assert (out.get("action") >= -1.0).all(), out.get("action").max()


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)