# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from copy import deepcopy

import numpy as np
import pytest
import torch
from _utils_internal import get_available_devices
from torch import nn, autograd
from torchrl.data import (
    TensorDict,
    NdBoundedTensorSpec,
    MultOneHotDiscreteTensorSpec,
    NdUnboundedContinuousTensorSpec,
    CompositeSpec,
)
from torchrl.data.postprocs.postprocs import MultiStep

# from torchrl.data.postprocs.utils import expand_as_right
from torchrl.data.tensordict.tensordict import assert_allclose_td
from torchrl.data.utils import expand_as_right
from torchrl.modules import DistributionalQValueActor, QValueActor, TensorDictModule
from torchrl.modules.distributions.continuous import TanhNormal, NormalParamWrapper
from torchrl.modules.models.models import MLP
from torchrl.modules.tensordict_module.actors import (
    ValueOperator,
    Actor,
    ProbabilisticActor,
)
from torchrl.objectives import (
    DQNLoss,
    DistributionalDQNLoss,
    DDPGLoss,
    SACLoss,
    PPOLoss,
    ClipPPOLoss,
    KLPENPPOLoss,
)
from torchrl.objectives.common import _LossModule
from torchrl.objectives.redq import (
    REDQLoss,
    REDQLoss_deprecated,
    DoubleREDQLoss_deprecated,
)
from torchrl.objectives.reinforce import ReinforceLoss
from torchrl.objectives.utils import hold_out_net, HardUpdate, SoftUpdate
from torchrl.objectives.value.advantages import TDEstimate, GAE, TDLambdaEstimate
from torchrl.objectives.value.functional import (
    vec_td_lambda_advantage_estimate,
    td_lambda_advantage_estimate,
)


class _check_td_steady:
    def __init__(self, td):
        self.td_clone = td.clone()
        self.td = td

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert (self.td.select(*self.td_clone.keys()) == self.td_clone).all()


def get_devices():
    devices = [torch.device("cpu")]
    for i in range(torch.cuda.device_count()):
        devices += [torch.device(f"cuda:{i}")]
    return devices


class TestDQN:
    seed = 0

    def _create_mock_actor(self, batch=2, obs_dim=3, action_dim=4, device="cpu"):
        # Actor
        action_spec = NdBoundedTensorSpec(
            -torch.ones(action_dim), torch.ones(action_dim), (action_dim,)
        )
        module = nn.Linear(obs_dim, action_dim)
        actor = QValueActor(
            spec=CompositeSpec(
                action=action_spec, action_value=None, chosen_action_value=None
            ),
            module=module,
        ).to(device)
        return actor

    def _create_mock_distributional_actor(
        self, batch=2, obs_dim=3, action_dim=4, atoms=5, vmin=1, vmax=5
    ):
        # Actor
        action_spec = MultOneHotDiscreteTensorSpec([atoms] * action_dim)
        support = torch.linspace(vmin, vmax, atoms, dtype=torch.float)
        module = MLP(obs_dim, (atoms, action_dim))
        actor = DistributionalQValueActor(
            spec=CompositeSpec(action=action_spec, action_value=None),
            module=module,
            support=support,
        )
        return actor

    def _create_mock_data_dqn(
        self, batch=2, obs_dim=3, action_dim=4, atoms=None, device="cpu"
    ):
        # create a tensordict
        obs = torch.randn(batch, obs_dim)
        next_obs = torch.randn(batch, obs_dim)
        if atoms:
            action_value = torch.randn(batch, atoms, action_dim).softmax(-2)
            action = (
                action_value[..., 0, :] == action_value[..., 0, :].max(-1, True)[0]
            ).to(torch.long)
        else:
            action_value = torch.randn(batch, action_dim)
            action = (action_value == action_value.max(-1, True)[0]).to(torch.long)
        reward = torch.randn(batch, 1)
        done = torch.zeros(batch, 1, dtype=torch.bool)
        td = TensorDict(
            batch_size=(batch,),
            source={
                "observation": obs,
                "next_observation": next_obs,
                "done": done,
                "reward": reward,
                "action": action,
                "action_value": action_value,
            },
        ).to(device)
        return td

    def _create_seq_mock_data_dqn(
        self, batch=2, T=4, obs_dim=3, action_dim=4, atoms=None, device="cpu"
    ):
        # create a tensordict
        total_obs = torch.randn(batch, T + 1, obs_dim, device=device)
        obs = total_obs[:, :T]
        next_obs = total_obs[:, 1:]
        if atoms:
            action_value = torch.randn(
                batch, T, atoms, action_dim, device=device
            ).softmax(-2)
            action = (
                action_value[..., 0, :] == action_value[..., 0, :].max(-1, True)[0]
            ).to(torch.long)
        else:
            action_value = torch.randn(batch, T, action_dim, device=device)
            action = (action_value == action_value.max(-1, True)[0]).to(torch.long)
        reward = torch.randn(batch, T, 1, device=device)
        done = torch.zeros(batch, T, 1, dtype=torch.bool, device=device)
        mask = ~torch.zeros(batch, T, 1, dtype=torch.bool, device=device)
        td = TensorDict(
            batch_size=(batch, T),
            source={
                "observation": obs * mask.to(obs.dtype),
                "next_observation": next_obs * mask.to(obs.dtype),
                "done": done,
                "mask": mask,
                "reward": reward * mask.to(obs.dtype),
                "action": action * mask.to(obs.dtype),
                "action_value": action_value
                * expand_as_right(mask.to(obs.dtype).squeeze(-1), action_value),
            },
        )
        return td

    @pytest.mark.parametrize("delay_value", (False, True))
    @pytest.mark.parametrize("device", get_available_devices())
    def test_dqn(self, delay_value, device):
        torch.manual_seed(self.seed)
        actor = self._create_mock_actor(device=device)
        td = self._create_mock_data_dqn(device=device)
        loss_fn = DQNLoss(actor, gamma=0.9, loss_function="l2", delay_value=delay_value)
        with _check_td_steady(td):
            loss = loss_fn(td)
        assert loss_fn.priority_key in td.keys()

        sum([item for _, item in loss.items()]).backward()
        assert torch.nn.utils.clip_grad.clip_grad_norm_(actor.parameters(), 1.0) > 0.0

        # Check param update effect on targets
        target_value = [p.clone() for p in loss_fn.target_value_network_params]
        for p in loss_fn.parameters():
            p.data += torch.randn_like(p)
        target_value2 = [p.clone() for p in loss_fn.target_value_network_params]
        if loss_fn.delay_value:
            assert all((p1 == p2).all() for p1, p2 in zip(target_value, target_value2))
        else:
            assert not any(
                (p1 == p2).any() for p1, p2 in zip(target_value, target_value2)
            )

        # check that policy is updated after parameter update
        parameters = [p.clone() for p in actor.parameters()]
        for p in loss_fn.parameters():
            p.data += torch.randn_like(p)
        assert all((p1 != p2).all() for p1, p2 in zip(parameters, actor.parameters()))

    @pytest.mark.parametrize("n", range(4))
    @pytest.mark.parametrize("delay_value", (False, True))
    @pytest.mark.parametrize("device", get_available_devices())
    def test_dqn_batcher(self, n, delay_value, device, gamma=0.9):
        torch.manual_seed(self.seed)
        actor = self._create_mock_actor(device=device)

        td = self._create_seq_mock_data_dqn(device=device)
        loss_fn = DQNLoss(
            actor, gamma=gamma, loss_function="l2", delay_value=delay_value
        )

        ms = MultiStep(gamma=gamma, n_steps_max=n).to(device)
        ms_td = ms(td.clone())

        with _check_td_steady(ms_td):
            loss_ms = loss_fn(ms_td)
        assert loss_fn.priority_key in ms_td.keys()

        with torch.no_grad():
            loss = loss_fn(td)
        if n == 0:
            assert_allclose_td(td, ms_td.select(*list(td.keys())))
            _loss = sum([item for _, item in loss.items()])
            _loss_ms = sum([item for _, item in loss_ms.items()])
            assert (
                abs(_loss - _loss_ms) < 1e-3
            ), f"found abs(loss-loss_ms) = {abs(loss - loss_ms):4.5f} for n=0"
        else:
            with pytest.raises(AssertionError):
                assert_allclose_td(loss, loss_ms)
        sum([item for _, item in loss_ms.items()]).backward()
        assert torch.nn.utils.clip_grad.clip_grad_norm_(actor.parameters(), 1.0) > 0.0

        # Check param update effect on targets
        target_value = [p.clone() for p in loss_fn.target_value_network_params]
        for p in loss_fn.parameters():
            p.data += torch.randn_like(p)
        target_value2 = [p.clone() for p in loss_fn.target_value_network_params]
        if loss_fn.delay_value:
            assert all((p1 == p2).all() for p1, p2 in zip(target_value, target_value2))
        else:
            assert not any(
                (p1 == p2).any() for p1, p2 in zip(target_value, target_value2)
            )

        # check that policy is updated after parameter update
        parameters = [p.clone() for p in actor.parameters()]
        for p in loss_fn.parameters():
            p.data += torch.randn_like(p)
        assert all((p1 != p2).all() for p1, p2 in zip(parameters, actor.parameters()))

    @pytest.mark.parametrize("atoms", range(4, 10))
    @pytest.mark.parametrize("delay_value", (False, True))
    @pytest.mark.parametrize("device", get_devices())
    def test_distributional_dqn(self, atoms, delay_value, device, gamma=0.9):
        torch.manual_seed(self.seed)
        actor = self._create_mock_distributional_actor(atoms=atoms).to(device)

        td = self._create_mock_data_dqn(atoms=atoms).to(device)
        loss_fn = DistributionalDQNLoss(actor, gamma=gamma, delay_value=delay_value)

        with _check_td_steady(td):
            loss = loss_fn(td)
        assert loss_fn.priority_key in td.keys()

        sum([item for _, item in loss.items()]).backward()
        assert torch.nn.utils.clip_grad.clip_grad_norm_(actor.parameters(), 1.0) > 0.0

        # Check param update effect on targets
        target_value = [p.clone() for p in loss_fn.target_value_network_params]
        for p in loss_fn.parameters():
            p.data += torch.randn_like(p)
        target_value2 = [p.clone() for p in loss_fn.target_value_network_params]
        if loss_fn.delay_value:
            assert all((p1 == p2).all() for p1, p2 in zip(target_value, target_value2))
        else:
            assert not any(
                (p1 == p2).any() for p1, p2 in zip(target_value, target_value2)
            )

        # check that policy is updated after parameter update
        parameters = [p.clone() for p in actor.parameters()]
        for p in loss_fn.parameters():
            p.data += torch.randn_like(p)
        assert all((p1 != p2).all() for p1, p2 in zip(parameters, actor.parameters()))


class TestDDPG:
    seed = 0

    def _create_mock_actor(self, batch=2, obs_dim=3, action_dim=4, device="cpu"):
        # Actor
        action_spec = NdBoundedTensorSpec(
            -torch.ones(action_dim), torch.ones(action_dim), (action_dim,)
        )
        module = nn.Linear(obs_dim, action_dim)
        actor = Actor(
            spec=action_spec,
            module=module,
        )
        return actor.to(device)

    def _create_mock_value(self, batch=2, obs_dim=3, action_dim=4, device="cpu"):
        # Actor
        class ValueClass(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(obs_dim + action_dim, 1)

            def forward(self, obs, act):
                return self.linear(torch.cat([obs, act], -1))

        module = ValueClass()
        value = ValueOperator(
            module=module,
            in_keys=["observation", "action"],
        )
        return value.to(device)

    def _create_mock_distributional_actor(
        self, batch=2, obs_dim=3, action_dim=4, atoms=5, vmin=1, vmax=5
    ):
        raise NotImplementedError

    def _create_mock_data_ddpg(
        self, batch=8, obs_dim=3, action_dim=4, atoms=None, device="cpu"
    ):
        # create a tensordict
        obs = torch.randn(batch, obs_dim, device=device)
        next_obs = torch.randn(batch, obs_dim, device=device)
        if atoms:
            raise NotImplementedError
        else:
            action = torch.randn(batch, action_dim, device=device).clamp(-1, 1)
        reward = torch.randn(batch, 1, device=device)
        done = torch.zeros(batch, 1, dtype=torch.bool, device=device)
        td = TensorDict(
            batch_size=(batch,),
            source={
                "observation": obs,
                "next_observation": next_obs,
                "done": done,
                "reward": reward,
                "action": action,
            },
        )
        return td

    def _create_seq_mock_data_ddpg(
        self, batch=8, T=4, obs_dim=3, action_dim=4, atoms=None, device="cpu"
    ):
        # create a tensordict
        total_obs = torch.randn(batch, T + 1, obs_dim, device=device)
        obs = total_obs[:, :T]
        next_obs = total_obs[:, 1:]
        if atoms:
            action = torch.randn(batch, T, atoms, action_dim, device=device).clamp(
                -1, 1
            )
        else:
            action = torch.randn(batch, T, action_dim, device=device).clamp(-1, 1)
        reward = torch.randn(batch, T, 1, device=device)
        done = torch.zeros(batch, T, 1, dtype=torch.bool, device=device)
        mask = ~torch.zeros(batch, T, 1, dtype=torch.bool, device=device)
        td = TensorDict(
            batch_size=(batch, T),
            source={
                "observation": obs * mask.to(obs.dtype),
                "next_observation": next_obs * mask.to(obs.dtype),
                "done": done,
                "mask": mask,
                "reward": reward * mask.to(obs.dtype),
                "action": action * mask.to(obs.dtype),
            },
        )
        return td

    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize("delay_actor,delay_value", [(False, False), (True, True)])
    def test_ddpg(self, delay_actor, delay_value, device):
        torch.manual_seed(self.seed)
        actor = self._create_mock_actor(device=device)
        value = self._create_mock_value(device=device)
        td = self._create_mock_data_ddpg(device=device)
        loss_fn = DDPGLoss(
            actor,
            value,
            gamma=0.9,
            loss_function="l2",
            delay_actor=delay_actor,
            delay_value=delay_value,
        )
        with _check_td_steady(td):
            loss = loss_fn(td)

        # check that loss are independent
        for k in loss.keys():
            if not k.startswith("loss"):
                continue
            loss[k].sum().backward(retain_graph=True)
            if k == "loss_actor":
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.value_network_params
                )
                assert not any(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.actor_network_params
                )
            elif k == "loss_value":
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.actor_network_params
                )
                assert not any(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.value_network_params
                )
            else:
                raise NotImplementedError(k)
            loss_fn.zero_grad()

        # check overall grad
        sum([item for _, item in loss.items()]).backward()
        parameters = list(actor.parameters()) + list(value.parameters())
        for p in parameters:
            assert p.grad.norm() > 0.0

        # Check param update effect on targets
        target_actor = [p.clone() for p in loss_fn.target_actor_network_params]
        target_value = [p.clone() for p in loss_fn.target_value_network_params]
        for p in loss_fn.parameters():
            p.data += torch.randn_like(p)
        target_actor2 = [p.clone() for p in loss_fn.target_actor_network_params]
        target_value2 = [p.clone() for p in loss_fn.target_value_network_params]
        if loss_fn.delay_actor:
            assert all((p1 == p2).all() for p1, p2 in zip(target_actor, target_actor2))
        else:
            assert not any(
                (p1 == p2).any() for p1, p2 in zip(target_actor, target_actor2)
            )
        if loss_fn.delay_value:
            assert all((p1 == p2).all() for p1, p2 in zip(target_value, target_value2))
        else:
            assert not any(
                (p1 == p2).any() for p1, p2 in zip(target_value, target_value2)
            )

        # check that policy is updated after parameter update
        parameters = [p.clone() for p in actor.parameters()]
        for p in loss_fn.parameters():
            p.data += torch.randn_like(p)
        assert all((p1 != p2).all() for p1, p2 in zip(parameters, actor.parameters()))

    @pytest.mark.parametrize("n", list(range(4)))
    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize("delay_actor,delay_value", [(False, False), (True, True)])
    def test_ddpg_batcher(self, n, delay_actor, delay_value, device, gamma=0.9):
        torch.manual_seed(self.seed)
        actor = self._create_mock_actor(device=device)
        value = self._create_mock_value(device=device)
        td = self._create_seq_mock_data_ddpg(device=device)
        loss_fn = DDPGLoss(
            actor,
            value,
            gamma=gamma,
            loss_function="l2",
            delay_actor=delay_actor,
            delay_value=delay_value,
        )

        ms = MultiStep(gamma=gamma, n_steps_max=n).to(device)
        ms_td = ms(td.clone())
        with _check_td_steady(ms_td):
            loss_ms = loss_fn(ms_td)
        with torch.no_grad():
            loss = loss_fn(td)
        if n == 0:
            assert_allclose_td(td, ms_td.select(*list(td.keys())))
            _loss = sum([item for _, item in loss.items()])
            _loss_ms = sum([item for _, item in loss_ms.items()])
            assert (
                abs(_loss - _loss_ms) < 1e-3
            ), f"found abs(loss-loss_ms) = {abs(loss - loss_ms):4.5f} for n=0"
        else:
            with pytest.raises(AssertionError):
                assert_allclose_td(loss, loss_ms)
        sum([item for _, item in loss_ms.items()]).backward()
        parameters = list(actor.parameters()) + list(value.parameters())
        for p in parameters:
            assert p.grad.norm() > 0.0


class TestSAC:
    seed = 0

    def _create_mock_actor(self, batch=2, obs_dim=3, action_dim=4, device="cpu"):
        # Actor
        action_spec = NdBoundedTensorSpec(
            -torch.ones(action_dim), torch.ones(action_dim), (action_dim,)
        )
        net = NormalParamWrapper(nn.Linear(obs_dim, 2 * action_dim))
        module = TensorDictModule(
            net, in_keys=["observation"], out_keys=["loc", "scale"]
        )
        actor = ProbabilisticActor(
            spec=CompositeSpec(action=action_spec, loc=None, scale=None),
            module=module,
            distribution_class=TanhNormal,
            dist_param_keys=["loc", "scale"],
        )
        return actor.to(device)

    def _create_mock_qvalue(self, batch=2, obs_dim=3, action_dim=4, device="cpu"):
        class ValueClass(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(obs_dim + action_dim, 1)

            def forward(self, obs, act):
                return self.linear(torch.cat([obs, act], -1))

        module = ValueClass()
        qvalue = ValueOperator(
            module=module,
            in_keys=["observation", "action"],
        )
        return qvalue.to(device)

    def _create_mock_value(self, batch=2, obs_dim=3, action_dim=4, device="cpu"):
        module = nn.Linear(obs_dim, 1)
        value = ValueOperator(
            module=module,
            in_keys=["observation"],
        )
        return value.to(device)

    def _create_mock_distributional_actor(
        self, batch=2, obs_dim=3, action_dim=4, atoms=5, vmin=1, vmax=5
    ):
        raise NotImplementedError

    def _create_mock_data_sac(
        self, batch=16, obs_dim=3, action_dim=4, atoms=None, device="cpu"
    ):
        # create a tensordict
        obs = torch.randn(batch, obs_dim, device=device)
        next_obs = torch.randn(batch, obs_dim, device=device)
        if atoms:
            raise NotImplementedError
        else:
            action = torch.randn(batch, action_dim, device=device).clamp(-1, 1)
        reward = torch.randn(batch, 1, device=device)
        done = torch.zeros(batch, 1, dtype=torch.bool, device=device)
        td = TensorDict(
            batch_size=(batch,),
            source={
                "observation": obs,
                "next_observation": next_obs,
                "done": done,
                "reward": reward,
                "action": action,
            },
        )
        return td

    def _create_seq_mock_data_sac(
        self, batch=8, T=4, obs_dim=3, action_dim=4, atoms=None, device="cpu"
    ):
        # create a tensordict
        total_obs = torch.randn(batch, T + 1, obs_dim, device=device)
        obs = total_obs[:, :T]
        next_obs = total_obs[:, 1:]
        if atoms:
            action = torch.randn(batch, T, atoms, action_dim, device=device).clamp(
                -1, 1
            )
        else:
            action = torch.randn(batch, T, action_dim, device=device).clamp(-1, 1)
        reward = torch.randn(batch, T, 1, device=device)
        done = torch.zeros(batch, T, 1, dtype=torch.bool, device=device)
        mask = ~torch.zeros(batch, T, 1, dtype=torch.bool, device=device)
        td = TensorDict(
            batch_size=(batch, T),
            source={
                "observation": obs * mask.to(obs.dtype),
                "next_observation": next_obs * mask.to(obs.dtype),
                "done": done,
                "mask": mask,
                "reward": reward * mask.to(obs.dtype),
                "action": action * mask.to(obs.dtype),
            },
        )
        return td

    @pytest.mark.parametrize("delay_value", (True, False))
    @pytest.mark.parametrize("delay_actor", (True, False))
    @pytest.mark.parametrize("delay_qvalue", (True, False))
    @pytest.mark.parametrize("num_qvalue", [1, 2, 4, 8])
    @pytest.mark.parametrize("device", get_available_devices())
    def test_sac(self, delay_value, delay_actor, delay_qvalue, num_qvalue, device):
        if (delay_actor or delay_qvalue) and not delay_value:
            pytest.skip("incompatible config")

        torch.manual_seed(self.seed)
        td = self._create_mock_data_sac(device=device)

        actor = self._create_mock_actor(device=device)
        qvalue = self._create_mock_qvalue(device=device)
        value = self._create_mock_value(device=device)

        kwargs = {}
        if delay_actor:
            kwargs["delay_actor"] = True
        if delay_qvalue:
            kwargs["delay_qvalue"] = True
        if delay_value:
            kwargs["delay_value"] = True

        loss_fn = SACLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            value_network=value,
            num_qvalue_nets=num_qvalue,
            gamma=0.9,
            loss_function="l2",
            **kwargs,
        )

        with _check_td_steady(td):
            loss = loss_fn(td)
        assert loss_fn.priority_key in td.keys()

        # check that loss are independent
        for k in loss.keys():
            if not k.startswith("loss"):
                continue
            loss[k].sum().backward(retain_graph=True)
            if k == "loss_actor":
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.value_network_params
                )
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.qvalue_network_params
                )
                assert not any(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.actor_network_params
                )
            elif k == "loss_value":
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.actor_network_params
                )
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.qvalue_network_params
                )
                assert not any(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.value_network_params
                )
            elif k == "loss_qvalue":
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.actor_network_params
                )
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.value_network_params
                )
                assert not any(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.qvalue_network_params
                )
            elif k == "loss_alpha":
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.actor_network_params
                )
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.value_network_params
                )
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.qvalue_network_params
                )
            else:
                raise NotImplementedError(k)
            loss_fn.zero_grad()

        sum([item for _, item in loss.items()]).backward()
        named_parameters = list(loss_fn.named_parameters())
        named_buffers = list(loss_fn.named_buffers())

        assert len(set(p for n, p in named_parameters)) == len(list(named_parameters))
        assert len(set(p for n, p in named_buffers)) == len(list(named_buffers))

        for name, p in named_parameters:
            assert p.grad.norm() > 0.0, f"parameter {name} has a null gradient"

    @pytest.mark.parametrize("n", list(range(4)))
    @pytest.mark.parametrize("delay_value", (True, False))
    @pytest.mark.parametrize("delay_actor", (True, False))
    @pytest.mark.parametrize("delay_qvalue", (True, False))
    @pytest.mark.parametrize("num_qvalue", [1, 2, 4, 8])
    @pytest.mark.parametrize("device", get_available_devices())
    def test_sac_batcher(
        self, n, delay_value, delay_actor, delay_qvalue, num_qvalue, device, gamma=0.9
    ):
        if (delay_actor or delay_qvalue) and not delay_value:
            pytest.skip("incompatible config")
        torch.manual_seed(self.seed)
        td = self._create_seq_mock_data_sac(device=device)

        actor = self._create_mock_actor(device=device)
        qvalue = self._create_mock_qvalue(device=device)
        value = self._create_mock_value(device=device)

        kwargs = {}
        if delay_actor:
            kwargs["delay_actor"] = True
        if delay_qvalue:
            kwargs["delay_qvalue"] = True
        if delay_value:
            kwargs["delay_value"] = True

        loss_fn = SACLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            value_network=value,
            num_qvalue_nets=num_qvalue,
            gamma=0.9,
            loss_function="l2",
            **kwargs,
        )

        ms = MultiStep(gamma=gamma, n_steps_max=n).to(device)

        td_clone = td.clone()
        ms_td = ms(td_clone)

        torch.manual_seed(0)
        np.random.seed(0)
        with _check_td_steady(ms_td):
            loss_ms = loss_fn(ms_td)
        assert loss_fn.priority_key in ms_td.keys()

        with torch.no_grad():
            torch.manual_seed(0)  # log-prob is computed with a random action
            np.random.seed(0)
            loss = loss_fn(td)
        if n == 0:
            assert_allclose_td(td, ms_td.select(*list(td.keys())))
            _loss = sum([item for _, item in loss.items()])
            _loss_ms = sum([item for _, item in loss_ms.items()])
            assert (
                abs(_loss - _loss_ms) < 1e-3
            ), f"found abs(loss-loss_ms) = {abs(loss - loss_ms):4.5f} for n=0"
        else:
            with pytest.raises(AssertionError):
                assert_allclose_td(loss, loss_ms)
        sum([item for _, item in loss_ms.items()]).backward()
        named_parameters = loss_fn.named_parameters()
        for name, p in named_parameters:
            assert p.grad.norm() > 0.0, f"parameter {name} has null gradient"

        # Check param update effect on targets
        target_actor = [p.clone() for p in loss_fn.target_actor_network_params]
        target_qvalue = [p.clone() for p in loss_fn.target_qvalue_network_params]
        target_value = [p.clone() for p in loss_fn.target_value_network_params]
        for p in loss_fn.parameters():
            p.data += torch.randn_like(p)
        target_actor2 = [p.clone() for p in loss_fn.target_actor_network_params]
        target_qvalue2 = [p.clone() for p in loss_fn.target_qvalue_network_params]
        target_value2 = [p.clone() for p in loss_fn.target_value_network_params]
        if loss_fn.delay_actor:
            assert all((p1 == p2).all() for p1, p2 in zip(target_actor, target_actor2))
        else:
            assert not any(
                (p1 == p2).any() for p1, p2 in zip(target_actor, target_actor2)
            )
        if loss_fn.delay_qvalue:
            assert all(
                (p1 == p2).all() for p1, p2 in zip(target_qvalue, target_qvalue2)
            )
        else:
            assert not any(
                (p1 == p2).any() for p1, p2 in zip(target_qvalue, target_qvalue2)
            )
        if loss_fn.delay_value:
            assert all((p1 == p2).all() for p1, p2 in zip(target_value, target_value2))
        else:
            assert not any(
                (p1 == p2).any() for p1, p2 in zip(target_value, target_value2)
            )

        # check that policy is updated after parameter update
        parameters = [p.clone() for p in actor.parameters()]
        for p in loss_fn.parameters():
            p.data += torch.randn_like(p)
        assert all((p1 != p2).all() for p1, p2 in zip(parameters, actor.parameters()))


class TestREDQ:
    seed = 0

    def _create_mock_actor(self, batch=2, obs_dim=3, action_dim=4, device="cpu"):
        # Actor
        action_spec = NdBoundedTensorSpec(
            -torch.ones(action_dim), torch.ones(action_dim), (action_dim,)
        )
        net = NormalParamWrapper(nn.Linear(obs_dim, 2 * action_dim))
        module = TensorDictModule(
            net, in_keys=["observation"], out_keys=["loc", "scale"]
        )
        actor = ProbabilisticActor(
            module=module,
            distribution_class=TanhNormal,
            return_log_prob=True,
            dist_param_keys=["loc", "scale"],
            spec=CompositeSpec(action=action_spec, loc=None, scale=None),
        )
        return actor.to(device)

    def _create_mock_qvalue(self, batch=2, obs_dim=3, action_dim=4, device="cpu"):
        class ValueClass(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(obs_dim + action_dim, 1)

            def forward(self, obs, act):
                return self.linear(torch.cat([obs, act], -1))

        module = ValueClass()
        qvalue = ValueOperator(
            module=module,
            in_keys=["observation", "action"],
        )
        return qvalue.to(device)

    def _create_mock_data_redq(
        self, batch=16, obs_dim=3, action_dim=4, atoms=None, device="cpu"
    ):
        # create a tensordict
        obs = torch.randn(batch, obs_dim, device=device)
        next_obs = torch.randn(batch, obs_dim, device=device)
        if atoms:
            raise NotImplementedError
        else:
            action = torch.randn(batch, action_dim, device=device).clamp(-1, 1)
        reward = torch.randn(batch, 1, device=device)
        done = torch.zeros(batch, 1, dtype=torch.bool, device=device)
        td = TensorDict(
            batch_size=(batch,),
            source={
                "observation": obs,
                "next_observation": next_obs,
                "done": done,
                "reward": reward,
                "action": action,
            },
        )
        return td

    def _create_seq_mock_data_redq(
        self, batch=8, T=4, obs_dim=3, action_dim=4, atoms=None, device="cpu"
    ):
        # create a tensordict
        total_obs = torch.randn(batch, T + 1, obs_dim, device=device)
        obs = total_obs[:, :T]
        next_obs = total_obs[:, 1:]
        if atoms:
            action = torch.randn(batch, T, atoms, action_dim, device=device).clamp(
                -1, 1
            )
        else:
            action = torch.randn(batch, T, action_dim, device=device).clamp(-1, 1)
        reward = torch.randn(batch, T, 1, device=device)
        done = torch.zeros(batch, T, 1, dtype=torch.bool, device=device)
        mask = ~torch.zeros(batch, T, 1, dtype=torch.bool, device=device)
        td = TensorDict(
            batch_size=(batch, T),
            source={
                "observation": obs * mask.to(obs.dtype),
                "next_observation": next_obs * mask.to(obs.dtype),
                "done": done,
                "mask": mask,
                "reward": reward * mask.to(obs.dtype),
                "action": action * mask.to(obs.dtype),
            },
        )
        return td

    @pytest.mark.parametrize("delay_qvalue", (True, False))
    @pytest.mark.parametrize("num_qvalue", [1, 2, 4, 8])
    @pytest.mark.parametrize("device", get_available_devices())
    def test_redq(self, delay_qvalue, num_qvalue, device):

        torch.manual_seed(self.seed)
        td = self._create_mock_data_redq(device=device)

        actor = self._create_mock_actor(device=device)
        qvalue = self._create_mock_qvalue(device=device)

        loss_fn = REDQLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            num_qvalue_nets=num_qvalue,
            gamma=0.9,
            loss_function="l2",
            delay_qvalue=delay_qvalue,
        )

        with _check_td_steady(td):
            loss = loss_fn(td)

        # check td is left untouched
        assert loss_fn.priority_key in td.keys()

        # check that loss are independent
        for k in loss.keys():
            if not k.startswith("loss"):
                continue
            loss[k].sum().backward(retain_graph=True)
            if k == "loss_actor":
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.qvalue_network_params
                )
                assert not any(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.actor_network_params
                )
            elif k == "loss_qvalue":
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.actor_network_params
                )
                assert not any(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.qvalue_network_params
                )
            elif k == "loss_alpha":
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.actor_network_params
                )
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.qvalue_network_params
                )
            else:
                raise NotImplementedError(k)
            loss_fn.zero_grad()

        sum([item for _, item in loss.items()]).backward()
        named_parameters = list(loss_fn.named_parameters())
        named_buffers = list(loss_fn.named_buffers())

        assert len(set(p for n, p in named_parameters)) == len(list(named_parameters))
        assert len(set(p for n, p in named_buffers)) == len(list(named_buffers))

        for name, p in named_parameters:
            assert p.grad.norm() > 0.0, f"parameter {name} has a null gradient"

    @pytest.mark.parametrize("delay_qvalue", (True, False))
    @pytest.mark.parametrize("num_qvalue", [1, 2, 4, 8])
    @pytest.mark.parametrize("device", get_available_devices())
    def test_redq_batched(self, delay_qvalue, num_qvalue, device):

        torch.manual_seed(self.seed)
        td = self._create_mock_data_redq(device=device)

        actor = self._create_mock_actor(device=device)
        qvalue = self._create_mock_qvalue(device=device)

        loss_fn = REDQLoss(
            actor_network=deepcopy(actor),
            qvalue_network=deepcopy(qvalue),
            num_qvalue_nets=num_qvalue,
            gamma=0.9,
            loss_function="l2",
            delay_qvalue=delay_qvalue,
        )

        loss_class_deprec = (
            REDQLoss_deprecated if not delay_qvalue else DoubleREDQLoss_deprecated
        )
        loss_fn_deprec = loss_class_deprec(
            actor_network=deepcopy(actor),
            qvalue_network=deepcopy(qvalue),
            num_qvalue_nets=num_qvalue,
            gamma=0.9,
            loss_function="l2",
        )

        td_clone1 = td.clone()
        td_clone2 = td.clone()
        torch.manual_seed(0)
        with _check_td_steady(td_clone1):
            loss1 = loss_fn(td_clone1)

        torch.manual_seed(0)
        with _check_td_steady(td_clone2):
            loss2 = loss_fn_deprec(td_clone2)

        # TODO: find a way to compare the losses: problem is that we sample actions either sequentially or in batch,
        #  so setting seed has little impact

    @pytest.mark.parametrize("n", list(range(4)))
    @pytest.mark.parametrize("delay_qvalue", (True, False))
    @pytest.mark.parametrize("num_qvalue", [1, 2, 4, 8])
    @pytest.mark.parametrize("device", get_available_devices())
    def test_redq_batcher(self, n, delay_qvalue, num_qvalue, device, gamma=0.9):
        torch.manual_seed(self.seed)
        td = self._create_seq_mock_data_redq(device=device)

        actor = self._create_mock_actor(device=device)
        qvalue = self._create_mock_qvalue(device=device)

        loss_fn = REDQLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            num_qvalue_nets=num_qvalue,
            gamma=0.9,
            loss_function="l2",
            delay_qvalue=delay_qvalue,
        )

        ms = MultiStep(gamma=gamma, n_steps_max=n).to(device)

        td_clone = td.clone()
        ms_td = ms(td_clone)

        torch.manual_seed(0)
        np.random.seed(0)

        with _check_td_steady(ms_td):
            loss_ms = loss_fn(ms_td)
        assert loss_fn.priority_key in ms_td.keys()

        with torch.no_grad():
            torch.manual_seed(0)  # log-prob is computed with a random action
            np.random.seed(0)
            loss = loss_fn(td)
        if n == 0:
            assert_allclose_td(td, ms_td.select(*list(td.keys())))
            _loss = sum([item for _, item in loss.items()])
            _loss_ms = sum([item for _, item in loss_ms.items()])
            assert (
                abs(_loss - _loss_ms) < 1e-3
            ), f"found abs(loss-loss_ms) = {abs(loss - loss_ms):4.5f} for n=0"
        else:
            with pytest.raises(AssertionError):
                assert_allclose_td(loss, loss_ms)
        sum([item for _, item in loss_ms.items()]).backward()
        named_parameters = loss_fn.named_parameters()
        for name, p in named_parameters:
            assert p.grad.norm() > 0.0, f"parameter {name} has null gradient"

        # Check param update effect on targets
        target_actor = [p.clone() for p in loss_fn.target_actor_network_params]
        target_qvalue = [p.clone() for p in loss_fn.target_qvalue_network_params]
        for p in loss_fn.parameters():
            p.data += torch.randn_like(p)
        target_actor2 = [p.clone() for p in loss_fn.target_actor_network_params]
        target_qvalue2 = [p.clone() for p in loss_fn.target_qvalue_network_params]
        if loss_fn.delay_actor:
            assert all((p1 == p2).all() for p1, p2 in zip(target_actor, target_actor2))
        else:
            assert not any(
                (p1 == p2).any() for p1, p2 in zip(target_actor, target_actor2)
            )
        if loss_fn.delay_qvalue:
            assert all(
                (p1 == p2).all() for p1, p2 in zip(target_qvalue, target_qvalue2)
            )
        else:
            assert not any(
                (p1 == p2).any() for p1, p2 in zip(target_qvalue, target_qvalue2)
            )

        # check that policy is updated after parameter update
        actorp_set = set(actor.parameters())
        loss_fnp_set = set(loss_fn.parameters())
        assert len(actorp_set.intersection(loss_fnp_set)) == len(actorp_set)
        parameters = [p.clone() for p in actor.parameters()]
        for p in loss_fn.parameters():
            p.data += torch.randn_like(p)
        assert all((p1 != p2).all() for p1, p2 in zip(parameters, actor.parameters()))


class TestPPO:
    seed = 0

    def _create_mock_actor(self, batch=2, obs_dim=3, action_dim=4, device="cpu"):
        # Actor
        action_spec = NdBoundedTensorSpec(
            -torch.ones(action_dim), torch.ones(action_dim), (action_dim,)
        )
        net = NormalParamWrapper(nn.Linear(obs_dim, 2 * action_dim))
        module = TensorDictModule(
            net, in_keys=["observation"], out_keys=["loc", "scale"]
        )
        actor = ProbabilisticActor(
            module=module,
            distribution_class=TanhNormal,
            dist_param_keys=["loc", "scale"],
            spec=CompositeSpec(action=action_spec, loc=None, scale=None),
        )
        return actor.to(device)

    def _create_mock_value(self, batch=2, obs_dim=3, action_dim=4, device="cpu"):
        module = nn.Linear(obs_dim, 1)
        value = ValueOperator(
            module=module,
            in_keys=["observation"],
        )
        return value.to(device)

    def _create_mock_distributional_actor(
        self, batch=2, obs_dim=3, action_dim=4, atoms=0, vmin=1, vmax=5
    ):
        raise NotImplementedError

    def _create_mock_data_ppo(
        self, batch=2, obs_dim=3, action_dim=4, atoms=None, device="cpu"
    ):
        # create a tensordict
        obs = torch.randn(batch, obs_dim, device=device)
        next_obs = torch.randn(batch, obs_dim, device=device)
        if atoms:
            raise NotImplementedError
        else:
            action = torch.randn(batch, action_dim, device=device).clamp(-1, 1)
        reward = torch.randn(batch, 1, device=device)
        done = torch.zeros(batch, 1, dtype=torch.bool, device=device)
        td = TensorDict(
            batch_size=(batch,),
            source={
                "observation": obs,
                "next_observation": next_obs,
                "done": done,
                "reward": reward,
                "action": action,
                "sample_log_prob": torch.randn_like(action[..., :1]) / 10,
            },
        )
        return td

    def _create_seq_mock_data_ppo(
        self, batch=2, T=4, obs_dim=3, action_dim=4, atoms=None, device="cpu"
    ):
        # create a tensordict
        total_obs = torch.randn(batch, T + 1, obs_dim, device=device)
        obs = total_obs[:, :T]
        next_obs = total_obs[:, 1:]
        if atoms:
            action = torch.randn(batch, T, atoms, action_dim, device=device).clamp(
                -1, 1
            )
        else:
            action = torch.randn(batch, T, action_dim, device=device).clamp(-1, 1)
        reward = torch.randn(batch, T, 1, device=device)
        done = torch.zeros(batch, T, 1, dtype=torch.bool, device=device)
        mask = ~torch.zeros(batch, T, 1, dtype=torch.bool, device=device)
        params_mean = torch.randn_like(action) / 10
        params_scale = torch.rand_like(action) / 10
        td = TensorDict(
            batch_size=(batch, T),
            source={
                "observation": obs * mask.to(obs.dtype),
                "next_observation": next_obs * mask.to(obs.dtype),
                "done": done,
                "mask": mask,
                "reward": reward * mask.to(obs.dtype),
                "action": action * mask.to(obs.dtype),
                "sample_log_prob": torch.randn_like(action[..., :1])
                / 10
                * mask.to(obs.dtype),
                "loc": params_mean * mask.to(obs.dtype),
                "scale": params_scale * mask.to(obs.dtype),
            },
        )
        return td

    @pytest.mark.parametrize("loss_class", (PPOLoss, ClipPPOLoss, KLPENPPOLoss))
    @pytest.mark.parametrize("gradient_mode", (True, False))
    @pytest.mark.parametrize("advantage", ("gae", "td", "td_lambda"))
    @pytest.mark.parametrize("device", get_available_devices())
    def test_ppo(self, loss_class, device, gradient_mode, advantage):
        torch.manual_seed(self.seed)
        td = self._create_seq_mock_data_ppo(device=device)

        actor = self._create_mock_actor(device=device)
        value = self._create_mock_value(device=device)
        if advantage == "gae":
            advantage = GAE(
                gamma=0.9, lmbda=0.9, value_network=value, gradient_mode=gradient_mode
            )
        elif advantage == "td":
            advantage = TDEstimate(
                gamma=0.9, value_network=value, gradient_mode=gradient_mode
            )
        elif advantage == "td_lambda":
            advantage = TDLambdaEstimate(
                gamma=0.9, lmbda=0.9, value_network=value, gradient_mode=gradient_mode
            )
        else:
            raise NotImplementedError

        loss_fn = loss_class(
            actor, value, advantage_module=advantage, gamma=0.9, loss_critic_type="l2"
        )

        loss = loss_fn(td)
        loss_critic = loss["loss_critic"]
        loss_objective = loss["loss_objective"] + loss.get("loss_entropy", 0.0)
        loss_critic.backward(retain_graph=True)
        # check that grads are independent and non null
        named_parameters = loss_fn.named_parameters()
        for name, p in named_parameters:
            if p.grad is not None and p.grad.norm() > 0.0:
                assert "actor" not in name
                assert "critic" in name
            if p.grad is None:
                assert "actor" in name
                assert "critic" not in name

        value.zero_grad()
        loss_objective.backward()
        named_parameters = loss_fn.named_parameters()
        for name, p in named_parameters:
            if p.grad is not None and p.grad.norm() > 0.0:
                assert "actor" in name
                assert "critic" not in name
            if p.grad is None:
                assert "actor" not in name
                assert "critic" in name
        actor.zero_grad()


class TestReinforce:
    @pytest.mark.parametrize("delay_value", [True, False])
    @pytest.mark.parametrize("gradient_mode", [True, False])
    @pytest.mark.parametrize("advantage", ["gae", "td", "td_lambda"])
    def test_reinforce_value_net(self, advantage, gradient_mode, delay_value):
        n_obs = 3
        n_act = 5
        batch = 4
        gamma = 0.9
        value_net = ValueOperator(nn.Linear(n_obs, 1), in_keys=["observation"])
        net = NormalParamWrapper(nn.Linear(n_obs, 2 * n_act))
        module = TensorDictModule(
            net, in_keys=["observation"], out_keys=["loc", "scale"]
        )
        actor_net = ProbabilisticActor(
            module,
            distribution_class=TanhNormal,
            return_log_prob=True,
            dist_param_keys=["loc", "scale"],
            spec=CompositeSpec(
                action=NdUnboundedContinuousTensorSpec(n_act), loc=None, scale=None
            ),
        )
        if advantage == "gae":
            advantage_module = GAE(
                gamma=gamma,
                lmbda=0.9,
                value_network=value_net.make_functional_with_buffers(clone=True)[0],
                gradient_mode=gradient_mode,
            )
        elif advantage == "td":
            advantage_module = TDEstimate(
                gamma=gamma,
                value_network=value_net.make_functional_with_buffers(clone=True)[0],
                gradient_mode=gradient_mode,
            )
        elif advantage == "td_lambda":
            advantage_module = TDLambdaEstimate(
                gamma=0.9,
                lmbda=0.9,
                value_network=value_net.make_functional_with_buffers(clone=True)[0],
                gradient_mode=gradient_mode,
            )
        else:
            raise NotImplementedError

        loss_fn = ReinforceLoss(
            actor_net,
            critic=value_net,
            gamma=gamma,
            advantage_module=advantage_module,
            delay_value=delay_value,
        )

        td = TensorDict(
            {
                "reward": torch.randn(batch, 1),
                "observation": torch.randn(batch, n_obs),
                "next_observation": torch.randn(batch, n_obs),
                "done": torch.zeros(batch, 1, dtype=torch.bool),
                "action": torch.randn(batch, n_act),
            },
            [batch],
        )

        loss_td = loss_fn(td)
        grad_actor = autograd.grad(
            loss_td.get("loss_actor"),
            actor_net.parameters(),
            retain_graph=True,
        )
        grad_value = autograd.grad(
            loss_td.get("loss_value"),
            value_net.parameters(),
            retain_graph=True,
        )
        with pytest.raises(RuntimeError, match="One of the "):
            grad_actor = autograd.grad(
                loss_td.get("loss_actor"),
                value_net.parameters(),
                retain_graph=True,
                allow_unused=False,
            )
        with pytest.raises(RuntimeError, match="One of the "):
            grad_value = autograd.grad(
                loss_td.get("loss_value"),
                actor_net.parameters(),
                retain_graph=True,
                allow_unused=False,
            )


def test_hold_out():
    net = torch.nn.Linear(3, 4)
    x = torch.randn(1, 3)
    x_rg = torch.randn(1, 3, requires_grad=True)
    y = net(x)
    assert y.requires_grad
    with hold_out_net(net):
        y = net(x)
        assert not y.requires_grad
        y = net(x_rg)
        assert y.requires_grad

    y = net(x)
    assert y.requires_grad

    # nested case
    with hold_out_net(net):
        y = net(x)
        assert not y.requires_grad
        with hold_out_net(net):
            y = net(x)
            assert not y.requires_grad
            y = net(x_rg)
            assert y.requires_grad

    y = net(x)
    assert y.requires_grad

    # exception
    with pytest.raises(
        RuntimeError,
        match="hold_out_net requires the network parameter set to be non-empty.",
    ):
        net = torch.nn.Sequential()
        with hold_out_net(net):
            pass


@pytest.mark.parametrize("mode", ["hard", "soft"])
@pytest.mark.parametrize("value_network_update_interval", [100, 1000])
@pytest.mark.parametrize("device", get_available_devices())
def test_updater(mode, value_network_update_interval, device):
    torch.manual_seed(100)

    class custom_module_error(nn.Module):
        def __init__(self):
            super().__init__()
            self._target_params = [torch.randn(3, 4)]
            self._target_error_params = [torch.randn(3, 4)]
            self.params = nn.ParameterList(
                [nn.Parameter(torch.randn(3, 4, requires_grad=True))]
            )

    module = custom_module_error().to(device)
    with pytest.raises(
        RuntimeError, match="Your module seems to have a _target tensor list "
    ):
        if mode == "hard":
            upd = HardUpdate(module, value_network_update_interval)
        elif mode == "soft":
            upd = SoftUpdate(module, 1 - 1 / value_network_update_interval)

    class custom_module(_LossModule):
        def __init__(self):
            super().__init__()
            module1 = torch.nn.BatchNorm2d(10).eval()
            self.convert_to_functional(module1, "module1", create_target_params=True)
            module2 = torch.nn.BatchNorm2d(10).eval()
            self.module2 = module2
            for target in self.target_module1_params:
                target.data.normal_()
            for target in self.target_module1_buffers:
                if target.dtype is not torch.int64:
                    target.data.normal_()
                else:
                    target.data += 10

    module = custom_module().to(device)
    if mode == "hard":
        upd = HardUpdate(
            module, value_network_update_interval=value_network_update_interval
        )
    elif mode == "soft":
        upd = SoftUpdate(module, 1 - 1 / value_network_update_interval)
    upd.init_()
    for _, v in upd._targets.items():
        for _v in v:
            if _v.dtype is not torch.int64:
                _v.copy_(torch.randn_like(_v))
            else:
                _v += 10

    # total dist
    d0 = sum(
        [
            (target_val[0] - val[0]).norm().item()
            for (_, target_val), (_, val) in zip(
                upd._targets.items(), upd._sources.items()
            )
        ]
    )
    assert d0 > 0
    if mode == "hard":
        for i in range(value_network_update_interval + 1):
            d1 = sum(
                [
                    (target_val[0] - val[0]).norm().item()
                    for (_, target_val), (_, val) in zip(
                        upd._targets.items(), upd._sources.items()
                    )
                ]
            )
            assert d1 == d0, i
            assert upd.counter == i
            upd.step()
        assert upd.counter == 0
        d1 = sum(
            [
                (target_val[0] - val[0]).norm().item()
                for (_, target_val), (_, val) in zip(
                    upd._targets.items(), upd._sources.items()
                )
            ]
        )
        assert d1 < d0

    elif mode == "soft":
        upd.step()
        d1 = sum(
            [
                (target_val[0] - val[0]).norm().item()
                for (_, target_val), (_, val) in zip(
                    upd._targets.items(), upd._sources.items()
                )
            ]
        )
        assert d1 < d0

    upd.init_()
    upd.step()
    d2 = sum(
        [
            (target_val[0] - val[0]).norm().item()
            for (_, target_val), (_, val) in zip(
                upd._targets.items(), upd._sources.items()
            )
        ]
    )
    assert d2 < 1e-6


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("gamma", [0.1, 0.5, 0.99])
@pytest.mark.parametrize("lmbda", [0.1, 0.5, 0.99])
@pytest.mark.parametrize("N", [(3,), (7, 3)])
@pytest.mark.parametrize("T", [3, 5, 200])
def test_tdlambda(device, gamma, lmbda, N, T):
    torch.manual_seed(0)

    done = torch.zeros(*N, T, 1, device=device, dtype=torch.bool).bernoulli_(0.1)
    reward = torch.randn(*N, T, 1, device=device)
    state_value = torch.randn(*N, T, 1, device=device)
    next_state_value = torch.randn(*N, T, 1, device=device)

    r1 = vec_td_lambda_advantage_estimate(
        gamma, lmbda, state_value, next_state_value, reward, done
    )
    r2 = td_lambda_advantage_estimate(
        gamma, lmbda, state_value, next_state_value, reward, done
    )
    torch.testing.assert_close(r1, r2, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
