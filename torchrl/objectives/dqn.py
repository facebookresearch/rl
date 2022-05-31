# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torchrl.data import TensorDict
from torchrl.data.tensordict.tensordict import _TensorDict
from torchrl.envs.utils import step_tensordict
from torchrl.modules import (
    DistributionalQValueActor,
    QValueActor,
)
from .common import _LossModule
from .utils import distance_loss, next_state_value

__all__ = [
    "DQNLoss",
    "DistributionalDQNLoss",
]


class DQNLoss(_LossModule):
    """
    The DQN Loss class.
    Args:
        value_network (ProbabilisticTDModule): a Q value operator.
        gamma (scalar): a discount factor for return computation.
        loss_function (str): loss function for the value discrepancy. Can be one of "l1", "l2" or "smooth_l1".
        delay_value (bool, optional): whether to duplicate the value network into a new target value network to
            create a double DQN. Default is `False`.
    """

    def __init__(
        self,
        value_network: QValueActor,
        gamma: float,
        loss_function: str = "l2",
        priority_key: str = "td_error",
        delay_value: bool = False,
    ) -> None:

        super().__init__()
        self.delay_value = delay_value
        self.convert_to_functional(
            value_network,
            "value_network",
            create_target_params=self.delay_value,
        )

        self.value_network_in_keys = value_network.in_keys
        if not isinstance(value_network, QValueActor):
            raise TypeError(
                f"DQNLoss requires value_network to be of QValueActor dtype, got {type(value_network)}"
            )
        self.gamma = gamma
        self.loss_function = loss_function
        self.priority_key = priority_key

    def forward(self, input_tensordict: _TensorDict) -> TensorDict:
        """
        Computes the DQN loss given a tensordict sampled from the replay buffer.
        This function will also write a "td_error" key that can be used by prioritized replay buffers to assign
            a priority to items in the tensordict.

        Args:
            input_tensordict (_TensorDict): a tensordict with keys ["done", "reward", "action"] and the in_keys of
                the value network.

        Returns:
            a tensor containing the DQN loss.

        """

        device = self.device if self.device is not None else input_tensordict.device
        tensordict = input_tensordict.to(device)
        if tensordict.device != device:
            raise RuntimeError(
                f"device {device} was expected for "
                f"{tensordict.__class__.__name__} but {tensordict.device} was found"
            )

        for k, t in tensordict.items():
            if t.device != device:
                raise RuntimeError(
                    f"found key value pair {k}-{t.shape} "
                    f"with device {t.device} when {device} was required"
                )

        action = tensordict.get("action")

        action = action.to(torch.float)
        td_copy = tensordict.clone()
        if td_copy.device != tensordict.device:
            raise RuntimeError(f"{tensordict} and {td_copy} have different devices")
        self.value_network(
            td_copy,
            params=self.value_network_params,
            buffers=self.value_network_buffers,
        )

        pred_val = td_copy.get("action_value")
        pred_val_index = (pred_val * action).sum(-1)

        with torch.no_grad():
            target_value = next_state_value(
                tensordict,
                self.value_network,
                gamma=self.gamma,
                params=self.target_value_network_params,
                buffers=self.target_value_network_buffers,
                next_val_key="chosen_action_value",
            )
        priority_tensor = abs(pred_val_index - target_value)
        priority_tensor = priority_tensor.detach().unsqueeze(-1)
        priority_tensor = priority_tensor.to(input_tensordict.device)

        input_tensordict.set(
            self.priority_key,
            priority_tensor,
            inplace=True,
        )
        loss = distance_loss(pred_val_index, target_value, self.loss_function)
        return TensorDict({"loss": loss.mean()}, [])


class DistributionalDQNLoss(_LossModule):
    """
    A distributional DQN loss class.
    Distributional DQN uses a value network that outputs a distribution of
    values over a discrete support of discounted returns (unlike regular DQN
    where the value network outputs a single point prediction of the
    disctounted return).

    For more details regarding Distributional DQN, refer to "A Distributional
    Perspective on Reinforcement Learning",
    https://arxiv.org/pdf/1707.06887.pdf

    Args:
        value_network (DistributionalQValueActor): the distributional Q
            value operator.
        gamma (scalar): a discount factor for return computation.
        delay_value (bool): whether to duplicate the value network into a new target value network to create double DQN
    """

    def __init__(
        self,
        value_network: DistributionalQValueActor,
        gamma: float,
        priority_key: str = "td_error",
        delay_value: bool = False,
    ):
        super().__init__()
        self.gamma = gamma
        self.priority_key = priority_key
        self.delay_value = delay_value
        if not isinstance(value_network, DistributionalQValueActor):
            raise TypeError(
                "Expected value_network to be of type "
                "DistributionalQValueActor "
                f"but got {type(value_network)}"
            )
        self.convert_to_functional(
            value_network,
            "value_network",
            create_target_params=self.delay_value,
        )

    def forward(self, input_tensordict: _TensorDict) -> TensorDict:
        # from https://github.com/Kaixhin/Rainbow/blob/9ff5567ad1234ae0ed30d8471e8f13ae07119395/agent.py
        device = self.device
        tensordict = TensorDict(
            source=input_tensordict, batch_size=input_tensordict.batch_size
        ).to(device)

        if tensordict.batch_dims != 1:
            raise RuntimeError(
                f"{self.__class__.__name___} expects a 1-dimensional "
                "tensordict as input"
            )
        batch_size = tensordict.batch_size[0]
        support = self.value_network.support
        atoms = support.numel()
        Vmin = support.min().item()
        Vmax = support.max().item()
        delta_z = (Vmax - Vmin) / (atoms - 1)

        action = tensordict.get("action")
        reward = tensordict.get("reward")
        done = tensordict.get("done")

        steps_to_next_obs = tensordict.get("steps_to_next_obs", 1)
        discount = self.gamma ** steps_to_next_obs

        # Calculate current state probabilities (online network noise already
        # sampled)
        td_clone = tensordict.clone()
        self.value_network(
            td_clone,
            params=self.value_network_params,
            buffers=self.value_network_buffers,
        )  # Log probabilities log p(s_t, ·; θonline)
        action_log_softmax = td_clone.get("action_value")
        action_expand = action.unsqueeze(-2).expand_as(action_log_softmax)
        log_ps_a = action_log_softmax.masked_select(action_expand.to(torch.bool))
        log_ps_a = log_ps_a.view(batch_size, atoms)  # log p(s_t, a_t; θonline)

        with torch.no_grad():
            # Calculate nth next state probabilities
            next_td = step_tensordict(tensordict)
            self.value_network(
                next_td,
                params=self.value_network_params,
                buffers=self.value_network_buffers,
            )  # Probabilities p(s_t+n, ·; θonline)
            argmax_indices_ns = next_td.get("action").argmax(-1)  # one-hot encoding

            self.value_network(
                next_td,
                params=self.target_value_network_params,
                buffers=self.target_value_network_buffers,
            )  # Probabilities p(s_t+n, ·; θtarget)
            pns = next_td.get("action_value").exp()
            # Double-Q probabilities
            # p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)
            pns_a = pns[range(batch_size), :, argmax_indices_ns]

            # Compute Tz (Bellman operator T applied to z)
            # Tz = R^n + (γ^n)z (accounting for terminal states)
            if isinstance(discount, torch.Tensor):
                discount = discount.to("cpu")
            done = done.to("cpu")
            reward = reward.to("cpu")
            support = support.to("cpu")
            pns_a = pns_a.to("cpu")
            Tz = reward + (1 - done.to(reward.dtype)) * discount * support
            if Tz.shape != torch.Size([batch_size, atoms]):
                raise RuntimeError(
                    "Tz shape must be torch.Size([batch_size, atoms]), "
                    f"got Tz.shape={Tz.shape} and batch_size={batch_size}, "
                    f"atoms={atoms}"
                )
            # Clamp between supported values
            Tz = Tz.clamp_(min=Vmin, max=Vmax)
            if not torch.isfinite(Tz).all():
                raise RuntimeError("Tz has some non-finite elements")
            # Compute L2 projection of Tz onto fixed support z
            b = (Tz - Vmin) / delta_z  # b = (Tz - Vmin) / Δz
            low, up = b.floor().to(torch.int64), b.ceil().to(torch.int64)
            # Fix disappearing probability mass when l = b = u (b is int)
            low[(up > 0) & (low == up)] -= 1
            up[(low < (atoms - 1)) & (low == up)] += 1

            # Distribute probability of Tz
            m = torch.zeros(batch_size, atoms)
            offset = torch.linspace(
                0,
                ((batch_size - 1) * atoms),
                batch_size,
                dtype=torch.int64,
                # device=device,
            )
            offset = offset.unsqueeze(1).expand(batch_size, atoms)
            index = (low + offset).view(-1)
            tensor = (pns_a * (up.float() - b)).view(-1)
            # m_l = m_l + p(s_t+n, a*)(u - b)
            m.view(-1).index_add_(0, index, tensor)
            index = (up + offset).view(-1)
            tensor = (pns_a * (b - low.float())).view(-1)
            # m_u = m_u + p(s_t+n, a*)(b - l)
            m.view(-1).index_add_(0, index, tensor)

        # Cross-entropy loss (minimises DKL(m||p(s_t, a_t)))
        loss = -torch.sum(m.to(device) * log_ps_a, 1)
        input_tensordict.set(
            self.priority_key,
            loss.detach().unsqueeze(1).to(input_tensordict.device),
            inplace=True,
        )
        loss_td = TensorDict({"loss": loss.mean()}, [])
        return loss_td
