from typing import Optional

import numpy as np
import torch
from smac.env import StarCraft2Env
from torchrl.data import (
    TensorDict,
    NdUnboundedContinuousTensorSpec,
    UnboundedContinuousTensorSpec,
)
from torchrl.data.tensor_specs import (
    CustomNdOneHotDiscreteTensorSpec,
)
from torchrl.data.tensordict.tensordict import _TensorDict
from torchrl.envs.common import GymLikeEnv


class SCEnv(GymLikeEnv):
    available_envs = ["8m"]
    # TODO: add to parent class
    supplementary_keys = ["available_actions"]

    @property
    def observation_spec(self):
        info = self._env.get_env_info()
        dim = (info["n_agents"], info["obs_shape"])
        return NdUnboundedContinuousTensorSpec(dim)

    @property
    def action_spec(self):
        # info = self._env.get_env_info()
        return CustomNdOneHotDiscreteTensorSpec(
            torch.tensor(self._env.get_avail_actions())
        )

    @property
    def reward_spec(self):
        return UnboundedContinuousTensorSpec()

    def _build_env(self, map_name: str, taskname=None, **kwargs) -> None:
        if taskname:
            raise RuntimeError

        env = StarCraft2Env(map_name=map_name)
        self._env = env
        return env

    def _output_transform(self, step_result):
        reward, done, *other = step_result
        obs = self._env.get_obs()
        available_actions = self._env.get_avail_actions()
        return (obs, reward, done, available_actions, *other)

    def _reset(
        self, tensor_dict: Optional[_TensorDict] = None, **kwargs
    ) -> _TensorDict:
        obs = self._env.get_obs()

        tensor_dict_out = TensorDict(
            source=self._read_obs(np.array(obs)), batch_size=self.batch_size
        )
        self._is_done = torch.zeros(1, dtype=torch.bool)
        tensor_dict_out.set("done", self._is_done)
        available_actions = self._env.get_avail_actions()
        tensor_dict_out.set("available_actions", available_actions)
        return tensor_dict_out

    def _init_env(self, seed=None):
        self._env.reset()
        if seed is not None:
            self.set_seed(seed)

    # TODO: check that actions match avail
    def _action_transform(self, action):
        action_np = self.action_spec.to_numpy(action)
        return action_np

    # TODO: move to GymLike
    def _step(self, tensor_dict: _TensorDict) -> _TensorDict:
        action = tensor_dict.get("action")
        action_np = self._action_transform(action)

        reward = 0.0
        for _ in range(self.wrapper_frame_skip):
            obs, _reward, done, *info = self._output_transform(
                self._env.step(action_np)
            )
            if _reward is None:
                _reward = 0.0
            reward += _reward
            if done:
                break

        obs_dict = self._read_obs(np.array(obs))

        if reward is None:
            reward = np.nan
        reward = self._to_tensor(reward, dtype=self.reward_spec.dtype)
        done = self._to_tensor(done, dtype=torch.bool)
        self._is_done = done
        self._current_tensordict = obs_dict

        tensor_dict_out = TensorDict({}, batch_size=tensor_dict.batch_size)
        for key, value in obs_dict.items():
            tensor_dict_out.set(f"next_{key}", value)
        tensor_dict_out.set("reward", reward)
        tensor_dict_out.set("done", done)
        for k, value in zip(self.supplementary_keys, info):
            tensor_dict_out.set(k, value)

        return tensor_dict_out
