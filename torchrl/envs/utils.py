# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Union

import pkg_resources
from torch.autograd.grad_mode import _DecoratorContextManager

from torchrl.data.tensordict.tensordict import _TensorDict

AVAILABLE_LIBRARIES = {pkg.key for pkg in pkg_resources.working_set}


class classproperty(property):
    def __get__(self, cls, owner):
        return classmethod(self.fget).__get__(None, owner)()


def step_tensordict(
    tensordict: _TensorDict,
    next_tensordict: _TensorDict = None,
    keep_other: bool = True,
    exclude_reward: bool = True,
    exclude_done: bool = True,
    exclude_action: bool = True,
) -> _TensorDict:
    """
    Given a tensordict retrieved after a step, returns another tensordict with all the 'next_' prefixes are removed,
    i.e. all the `'next_some_other_string'` keys will be renamed onto `'some_other_string'` keys.


    Args:
        tensordict (_TensorDict): tensordict with keys to be renamed
        next_tensordict (_TensorDict, optional): destination tensordict
        keep_other (bool, optional): if True, all keys that do not start with `'next_'` will be kept.
            Default is True.
        exclude_reward (bool, optional): if True, the `"reward"` key will be discarded
            from the resulting tensordict.
            Default is True.
        exclude_done (bool, optional): if True, the `"done"` key will be discarded
            from the resulting tensordict.
            Default is True.
        exclude_action (bool, optional): if True, the `"action"` key will be discarded
            from the resulting tensordict.
            Default is True.

    Returns:
         A new tensordict (or next_tensordict) with the "next_*" keys renamed without the "next_" prefix.

    Examples:
    This funtion allows for this kind of loop to be used:
        >>> td_out = []
        >>> env = make_env()
        >>> policy = make_policy()
        >>> td = env.reset()
        >>> for i in range(max_steps):
        >>>     td = env.step(td)
        >>>     next_td = step_tensordict(td)
        >>>     assert next_td is not td # make sure that keys are not overwritten
        >>>     td_out.append(td)
        >>>     td = next_td
        >>> td_out = torch.stack(td_out, 0)
        >>> print(td_out) # should contain keys 'observation', 'next_observation', 'action', 'reward', 'done' or similar

    """
    other_keys = []
    to_exclude = []
    if exclude_done:
        to_exclude.append("done")
    if exclude_reward:
        to_exclude.append("reward")
    if exclude_action:
        to_exclude.append("action")
    if to_exclude:
        # we exclude those keys to make sure we aren't reporting time-specific values at the next step.
        tensordict = tensordict.exclude(*to_exclude)
    keys = [key for key in tensordict.keys() if key.startswith("next_")]
    if len(keys) == 0:
        raise RuntimeError(
            "There was no key starting with 'next_' in the provided TensorDict: ",
            tensordict,
        )
    new_keys = [key[5:] for key in keys]
    if keep_other:
        prohibited = set(keys).union(new_keys)
        other_keys = [key for key in tensordict.keys() if key not in prohibited]
    select_tensordict = tensordict.select(*other_keys, *keys)
    for new_key, key in zip(new_keys, keys):
        select_tensordict.rename_key(key, new_key, safe=True)
    if next_tensordict is not None:
        return next_tensordict.update(select_tensordict)
    else:
        return select_tensordict


def get_available_libraries():
    """

    Returns:
         all the supported libraries

    """
    return SUPPORTED_LIBRARIES


def _check_gym():
    """

    Returns:
         True if the gym library is installed

    """
    return "gym" in AVAILABLE_LIBRARIES


def _check_gym_atari():
    """

    Returns:
         True if the gym library is installed and atari envs can be found.

    """
    if not _check_gym():
        return False
    return "atari-py" in AVAILABLE_LIBRARIES


def _check_mario():
    """

    Returns:
         True if the "gym-super-mario-bros" library is installed.

    """

    return "gym-super-mario-bros" in AVAILABLE_LIBRARIES


def _check_dmcontrol():
    """

    Returns:
         True if the "dm-control" library is installed.

    """

    return "dm-control" in AVAILABLE_LIBRARIES


def _check_dmlab():
    """

    Returns:
         True if the "deepmind-lab" library is installed.

    """

    return "deepmind-lab" in AVAILABLE_LIBRARIES


SUPPORTED_LIBRARIES = {
    "gym": _check_gym(),  # OpenAI
    "gym[atari]": _check_gym_atari(),  #
    "vizdoom": None,  # 1.2k, https://github.com/mwydmuch/ViZDoom
    "ml-agents": None,
    # 11.5k, unity, https://github.com/Unity-Technologies/ml-agents
    "pysc2": None,  # 7.3k, DM, https://github.com/deepmind/pysc2
    "deepmind_lab": _check_dmlab(),
    # 6.5k DM, https://github.com/deepmind/lab, https://github.com/deepmind/lab/tree/master/python/pip_package
    "serpent.ai": None,  # 6k, https://github.com/SerpentAI/SerpentAI
    "gfootball": None,  # 2.8k G, https://github.com/google-research/football
    "dm_control": _check_dmcontrol(),
    # 2.3k DM, https://github.com/deepmind/dm_control
    "habitat": None,
    # 1.2k FB, https://github.com/facebookresearch/habitat-sim
    "meta-world": None,  # 500, https://github.com/rlworkgroup/metaworld
    "minerl": None,  # 300, https://github.com/minerllabs/minerl
    "multi-agent-emergence-environments": None,
    # 1.2k, OpenAI, https://github.com/openai/multi-agent-emergence-environments
    "openspiel": None,  # 2.8k, DM, https://github.com/deepmind/open_spiel
    "procgen": None,  # 500, OpenAI, https://github.com/openai/procgen
    "pybullet": None,  # 641, https://github.com/benelot/pybullet-gym
    "realworld_rl_suite": None,
    # 250, G, https://github.com/google-research/realworldrl_suite
    "rlcard": None,  # 1.4k, https://github.com/datamllab/rlcard
    "screeps": None,  # 2.3k https://github.com/screeps/screeps
    "gym-super-mario-bros": _check_mario(),
}

EXPLORATION_MODE = None


class set_exploration_mode(_DecoratorContextManager):
    """
    Sets the exploration mode of all ProbabilisticTDModules to the desired mode.

    Args:
        mode (str): mode to use when the policy is being called.

    Examples:
        >>> policy = Actor(action_spec, module=network, default_interaction_mode="mode")
        >>> env.rollout(policy=policy, max_steps=100)  # rollout with the "mode" interaction mode
        >>> with set_exploration_mode("random"):
        >>>     env.rollout(policy=policy, max_steps=100)  # rollout with the "random" interaction mode
    """

    def __init__(self, mode: str = "mode"):
        super().__init__()
        self.mode = mode

    def __enter__(self) -> None:
        global EXPLORATION_MODE
        self.prev = EXPLORATION_MODE
        EXPLORATION_MODE = self.mode

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        global EXPLORATION_MODE
        EXPLORATION_MODE = self.prev


def exploration_mode() -> Union[str, None]:
    """Returns the exploration mode currently set."""
    return EXPLORATION_MODE
