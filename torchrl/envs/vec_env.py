# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
from collections import OrderedDict
from multiprocessing import connection
from typing import Callable, Optional, Sequence, Union, Any, List

import torch
from torch import multiprocessing as mp

from torchrl.data import TensorDict, TensorSpec
from torchrl.data.tensordict.tensordict import _TensorDict
from torchrl.data.utils import CloudpickleWrapper, DEVICE_TYPING
from torchrl.envs.common import _EnvClass, make_tensordict

__all__ = ["SerialEnv", "ParallelEnv"]


def _check_start(fun):
    def decorated_fun(self: _BatchedEnv, *args, **kwargs):
        if self.is_closed:
            self._create_td()
            self._start_workers()
        return fun(self, *args, **kwargs)

    return decorated_fun


class _dispatch_caller_parallel:
    def __init__(self, attr, parallel_env):
        self.attr = attr
        self.parallel_env = parallel_env

    def __call__(self, *args, **kwargs):
        # remove self from args
        args = [_arg if _arg is not self.parallel_env else "_self" for _arg in args]
        for i, channel in enumerate(self.parallel_env.parent_channels):
            channel.send((self.attr, (args, kwargs)))

        results = []
        for channel in self.parallel_env.parent_channels:
            msg, result = channel.recv()
            results.append(result)

        return results

    def __iter__(self):
        # if the object returned is not a callable
        return iter(self.__call__())


class _dispatch_caller_serial:
    def __init__(self, list_callable: List[Callable, Any]):
        self.list_callable = list_callable

    def __call__(self, *args, **kwargs):
        return [_callable(*args, **kwargs) for _callable in self.list_callable]


class _BatchedEnv(_EnvClass):
    """

    Batched environments allow the user to query an arbitrary method / attribute of the environment running remotely.
    Those queries will return a list of length equal to the number of workers containing the
    values resulting from those queries.
        >>> env = ParallelEnv(3, my_env_fun)
        >>> custom_attribute_list = env.custom_attribute
        >>> custom_method_list = env.custom_method(*args)

    Args:
        num_workers: number of workers (i.e. env instances) to be deployed simultaneously;
        create_env_fn (callable or list of callables): function (or list of functions) to be used for the environment
            creation;
        create_env_kwargs (dict or list of dicts, optional): kwargs to be used with the environments being created;
        device (str, int, torch.device): device of the environment;
        action_keys (list of str, optional): list of keys that are to be considered policy-output. If the policy has it,
            the attribute policy.out_keys can be used.
            Providing the action_keys permit to select which keys to update after the policy is called, which can
            drastically decrease the IO burden when the tensordict is placed in shared memory / memory map.
        pin_memory (bool): if True and device is "cpu", calls `pin_memory` on the tensordicts when created.
        selected_keys (list of str, optional): keys that have to be returned by the environment.
            When creating a batch of environment, it might be the case that only some of the keys are to be returned.
            For instance, if the environment returns 'observation_pixels' and 'observation_vector', the user might only
            be interested in, say, 'observation_vector'. By indicating which keys must be returned in the tensordict,
            one can easily control the amount of data occupied in memory (for instance to limit the memory size of a
            replay buffer) and/or limit the amount of data passed from one process to the other;
        excluded_keys (list of str, optional): list of keys to be excluded from the returned tensordicts.
            See selected_keys for more details;
        share_individual_td (bool): if True, a different tensordict is created for every process/worker and a lazy
            stack is returned.
            default = False;
        shared_memory (bool): whether or not the returned tensordict will be placed in shared memory;
        memmap (bool): whether or not the returned tensordict will be placed in memory map.

    """

    _verbose: bool = False

    def __init__(
        self,
        num_workers: int,
        create_env_fn: Union[
            Callable[[], _EnvClass], Sequence[Callable[[], _EnvClass]]
        ],
        create_env_kwargs: Union[dict, Sequence[dict]] = None,
        device: DEVICE_TYPING = "cpu",
        action_keys: Optional[Sequence[str]] = None,
        pin_memory: bool = False,
        selected_keys: Optional[Sequence[str]] = None,
        excluded_keys: Optional[Sequence[str]] = None,
        share_individual_td: bool = False,
        shared_memory: bool = True,
        memmap: bool = False,
    ):
        super().__init__(device=device)
        self.is_closed = True

        create_env_kwargs = dict() if create_env_kwargs is None else create_env_kwargs
        if callable(create_env_fn):
            create_env_fn = [create_env_fn for _ in range(num_workers)]
        else:
            if len(create_env_fn) != num_workers:
                raise RuntimeError(
                    f"num_workers and len(create_env_fn) mismatch, "
                    f"got {len(create_env_fn)} and {num_workers}"
                )
        if isinstance(create_env_kwargs, dict):
            create_env_kwargs = [create_env_kwargs for _ in range(num_workers)]
        self._dummy_env = create_env_fn[0](**create_env_kwargs[0])
        self.num_workers = num_workers
        self.create_env_fn = create_env_fn
        self.create_env_kwargs = create_env_kwargs
        self.action_keys = action_keys
        self.pin_memory = pin_memory
        self.selected_keys = selected_keys
        self.excluded_keys = excluded_keys
        self.share_individual_td = share_individual_td
        self._share_memory = shared_memory
        self._memmap = memmap
        if self._share_memory and self._memmap:
            raise RuntimeError(
                "memmap and shared memory are mutually exclusive features."
            )

        self.batch_size = torch.Size([self.num_workers, *self._dummy_env.batch_size])
        self._action_spec = self._dummy_env.action_spec
        self._observation_spec = self._dummy_env.observation_spec
        self._reward_spec = self._dummy_env.reward_spec
        self._dummy_env.close()

    def state_dict(self, destination: Optional[OrderedDict] = None) -> OrderedDict:
        raise NotImplementedError

    def load_state_dict(self, state_dict: OrderedDict) -> None:
        raise NotImplementedError

    @property
    def action_spec(self) -> TensorSpec:
        return self._action_spec

    @property
    def observation_spec(self) -> TensorSpec:
        return self._observation_spec

    @property
    def reward_spec(self) -> TensorSpec:
        return self._reward_spec

    def is_done_set_fn(self, value: bool) -> None:
        self._is_done = value.all()

    def _create_td(self) -> None:
        """Creates self.shared_tensordict_parent, a TensorDict used to store the most recent observations."""
        shared_tensordict_parent = make_tensordict(
            self._dummy_env,
            None,
        )

        shared_tensordict_parent = shared_tensordict_parent.expand(
            self.num_workers
        ).clone()

        raise_no_selected_keys = False
        if self.selected_keys is None:
            self.selected_keys = list(shared_tensordict_parent.keys())
            if self.excluded_keys is not None:
                self.selected_keys = set(self.selected_keys) - set(self.excluded_keys)
            else:
                raise_no_selected_keys = True
            if self.action_keys is not None:
                if not all(
                    action_key in self.selected_keys for action_key in self.action_keys
                ):
                    raise KeyError(
                        "One of the action keys is not part of the selected keys or is part of the excluded keys. Action "
                        "keys need to be part of the selected keys for env.step() to be called."
                    )
            else:
                self.action_keys = [
                    key for key in self.selected_keys if key.startswith("action")
                ]
                if not len(self.action_keys):
                    raise RuntimeError(
                        f"found 0 action keys in {sorted(list(self.selected_keys))}"
                    )
        shared_tensordict_parent = shared_tensordict_parent.select(*self.selected_keys)
        self.shared_tensordict_parent = shared_tensordict_parent.to(self.device)

        if self.share_individual_td:
            self.shared_tensordicts = [
                td.clone() for td in self.shared_tensordict_parent.unbind(0)
            ]
            if self._share_memory:
                for td in self.shared_tensordicts:
                    td.share_memory_()
            elif self._memmap:
                for td in self.shared_tensordicts:
                    td.memmap_()
            self.shared_tensordict_parent = torch.stack(self.shared_tensordicts, 0)
        else:
            if self._share_memory:
                self.shared_tensordict_parent.share_memory_()
                if not self.shared_tensordict_parent.is_shared():
                    raise RuntimeError("share_memory_() failed")
            elif self._memmap:
                self.shared_tensordict_parent.memmap_()
                if not self.shared_tensordict_parent.is_memmap():
                    raise RuntimeError("memmap_() failed")

            self.shared_tensordicts = self.shared_tensordict_parent.unbind(0)
        if self.pin_memory:
            self.shared_tensordict_parent.pin_memory()

        if raise_no_selected_keys:
            if self._verbose:
                print(
                    f"\n {self.__class__.__name__}.shared_tensordict_parent is \n{self.shared_tensordict_parent}. \n"
                    f"You can select keys to be synchronised by setting the selected_keys and/or excluded_keys "
                    f"arguments when creating the batched environment."
                )

    def _start_workers(self) -> None:
        """Starts the various envs."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(\n\tenv={self._dummy_env}, \n\tbatch_size={self.batch_size})"

    def __del__(self) -> None:
        if not self.is_closed:
            raise RuntimeError(
                "Batched environment must be explicitely closed before it "
                "turns out of scope."
            )

    def close(self) -> None:
        if self.is_closed:
            raise RuntimeError("trying to close a closed environment")
        if self._verbose:
            print(f"closing {self.__class__.__name__}")
        self._shutdown_workers()
        self.is_closed = True

    def _shutdown_workers(self) -> None:
        raise NotImplementedError


class SerialEnv(_BatchedEnv):
    """
    Creates a series of environments in the same process.

    """

    __doc__ += _BatchedEnv.__doc__

    _share_memory = False

    def _start_workers(self) -> None:
        _num_workers = self.num_workers

        self._envs = []

        for idx in range(_num_workers):
            env = self.create_env_fn[idx](**self.create_env_kwargs[idx])
            self._envs.append(env)
        self.is_closed = False

    @_check_start
    def state_dict(self, destination: Optional[OrderedDict] = None) -> OrderedDict:
        state_dict = OrderedDict()
        for idx, env in enumerate(self._envs):
            state_dict[f"worker{idx}"] = env.state_dict()

        if destination is not None:
            destination.update(state_dict)
            return destination
        return state_dict

    @_check_start
    def load_state_dict(self, state_dict: OrderedDict) -> None:
        if "worker0" not in state_dict:
            state_dict = OrderedDict(
                **{f"worker{idx}": state_dict for idx in range(self.num_workers)}
            )
        for idx, env in enumerate(self._envs):
            env.load_state_dict(state_dict[f"worker{idx}"])

    @_check_start
    def _step(
        self,
        tensordict: TensorDict,
    ) -> TensorDict:
        self._assert_tensordict_shape(tensordict)

        self.shared_tensordict_parent.update_(tensordict)
        for i in range(self.num_workers):
            self._envs[i].step(self.shared_tensordicts[i])

        return self.shared_tensordict_parent

    def _shutdown_workers(self) -> None:
        if not self.is_closed:
            for env in self._envs:
                env.close()
            del self._envs

    @_check_start
    def set_seed(self, seed: int) -> int:
        for i, env in enumerate(self._envs):
            env.set_seed(seed)
            if i < self.num_workers - 1:
                seed = seed + 1
        return seed

    @_check_start
    def _reset(self, tensordict: _TensorDict, **kwargs) -> _TensorDict:
        if tensordict is not None and "reset_workers" in tensordict.keys():
            self._assert_tensordict_shape(tensordict)
            reset_workers = tensordict.get("reset_workers")
        else:
            reset_workers = torch.ones(self.num_workers, 1, dtype=torch.bool)

        keys = set()
        for i, _env in enumerate(self._envs):
            if not reset_workers[i]:
                continue
            _td = _env.reset(**kwargs)
            keys = keys.union(_td.keys())
            self.shared_tensordicts[i].update_(_td)

        return self.shared_tensordict_parent.select(*keys).clone()

    def __getattr__(self, attr: str) -> Any:
        if attr in self.__dir__():
            return self.__getattribute__(
                attr
            )  # make sure that appropriate exceptions are raised
        elif attr.startswith("__"):
            raise AttributeError(
                "dispatching built-in private methods is "
                f"not permitted with type {type(self)}. "
                f"Got attribute {attr}."
            )
        else:
            try:
                # determine if attr is a callable
                callable_attr = callable(getattr(self._dummy_env, attr))
                list_attr = [getattr(env, attr) for env in self._envs]
                if callable_attr:
                    if self.is_closed:
                        raise RuntimeError(
                            "Trying to access attributes of closed/non started "
                            "environments. Check that the batched environment "
                            "has been started (e.g. by calling env.reset)"
                        )
                    return _dispatch_caller_serial(list_attr)
                else:
                    return list_attr
            except AttributeError:
                raise AttributeError(
                    f"attribute {attr} not found in "
                    f"{self._dummy_env.__class__.__name__}"
                )


class ParallelEnv(_BatchedEnv):
    """
    Creates one environment per process.
    TensorDicts are passed via shared memory or memory map.

    """

    __doc__ += _BatchedEnv.__doc__

    def _start_workers(self) -> None:

        _num_workers = self.num_workers
        ctx = mp.get_context("spawn")

        self.parent_channels = []
        self._workers = []

        for idx in range(_num_workers):
            if self._verbose:
                print(f"initiating worker {idx}")
            # No certainty which module multiprocessing_context is
            channel1, channel2 = ctx.Pipe()
            env_fun = self.create_env_fn[idx]
            if env_fun.__class__.__name__ != "EnvCreator":
                env_fun = CloudpickleWrapper(env_fun)

            w = mp.Process(
                target=_run_worker_pipe_shared_mem,
                args=(
                    idx,
                    channel1,
                    channel2,
                    env_fun,
                    self.create_env_kwargs[idx],
                    False,
                    self.action_keys,
                ),
            )
            w.daemon = True
            w.start()
            channel2.close()
            self.parent_channels.append(channel1)
            self._workers.append(w)

        # send shared tensordict to workers
        for channel, shared_tensordict in zip(
            self.parent_channels, self.shared_tensordicts
        ):
            channel.send(("init", shared_tensordict))
        self.is_closed = False

    @_check_start
    def state_dict(self, destination: Optional[OrderedDict] = None) -> OrderedDict:
        state_dict = OrderedDict()
        for idx, channel in enumerate(self.parent_channels):
            channel.send(("state_dict", None))
        for idx, channel in enumerate(self.parent_channels):
            msg, _state_dict = channel.recv()
            if msg != "state_dict":
                raise RuntimeError(f"Expected 'state_dict' but received {msg}")
            state_dict[f"worker{idx}"] = _state_dict

        if destination is not None:
            destination.update(state_dict)
            return destination
        return state_dict

    @_check_start
    def load_state_dict(self, state_dict: OrderedDict) -> None:
        if "worker0" not in state_dict:
            state_dict = OrderedDict(
                **{f"worker{idx}": state_dict for idx in range(self.num_workers)}
            )
        for i, channel in enumerate(self.parent_channels):
            channel.send(("load_state_dict", state_dict[f"worker{i}"]))
        for channel in self.parent_channels:
            msg, _ = channel.recv()
            if msg != "loaded":
                raise RuntimeError(f"Expected 'loaded' but received {msg}")

    @_check_start
    def _step(self, tensordict: _TensorDict) -> _TensorDict:
        self._assert_tensordict_shape(tensordict)

        self.shared_tensordict_parent.update_(tensordict.select(*self.action_keys))
        for i in range(self.num_workers):
            self.parent_channels[i].send(("step", None))

        keys = set()
        for i in range(self.num_workers):
            msg, data = self.parent_channels[i].recv()
            if msg != "step_result":
                if msg != "done":
                    raise RuntimeError(
                        f"Expected 'done' but received {msg} from worker {i}"
                    )
            # data is the set of updated keys
            keys = keys.union(data)
        return self.shared_tensordict_parent.select(*keys)

    @_check_start
    def _shutdown_workers(self) -> None:
        if self.is_closed:
            raise RuntimeError(
                "calling {self.__class__.__name__}._shutdown_workers only allowed when env.is_closed = False"
            )
        for i, channel in enumerate(self.parent_channels):
            if self._verbose:
                print(f"closing {i}")
            # try:
            channel.send(("close", None))
            # except:
            #     raise RuntimeError(f"closing {channel} number {i} failed")
            msg, _ = channel.recv()
            if msg != "closing":
                raise RuntimeError(
                    f"Expected 'closing' but received {msg} from worker {i}"
                )

        del self.shared_tensordicts, self.shared_tensordict_parent

        for channel in self.parent_channels:
            channel.close()
        for proc in self._workers:
            proc.join()
        del self._workers
        del self.parent_channels

    @_check_start
    def set_seed(self, seed: int) -> int:
        for i, channel in enumerate(self.parent_channels):
            channel.send(("seed", seed))
            if i < self.num_workers - 1:
                seed = seed + 1
        for channel in self.parent_channels:
            msg, _ = channel.recv()
            if msg != "seeded":
                raise RuntimeError(f"Expected 'seeded' but received {msg}")
        return seed

    @_check_start
    def _reset(self, tensordict: _TensorDict, **kwargs) -> _TensorDict:
        cmd_out = "reset"
        if tensordict is not None and "reset_workers" in tensordict.keys():
            self._assert_tensordict_shape(tensordict)
            reset_workers = tensordict.get("reset_workers")
        else:
            reset_workers = torch.ones(self.num_workers, 1, dtype=torch.bool)

        for i, channel in enumerate(self.parent_channels):
            if not reset_workers[i]:
                continue
            channel.send((cmd_out, kwargs))

        keys = set()
        for i, channel in enumerate(self.parent_channels):
            if not reset_workers[i]:
                continue
            cmd_in, new_keys = channel.recv()
            keys = keys.union(new_keys)
            if cmd_in != "reset_obs":
                raise RuntimeError(f"received cmd {cmd_in} instead of reset_obs")
        if self.shared_tensordict_parent.get("done").any():
            raise RuntimeError("Envs have just been reset but some are still done")
        return self.shared_tensordict_parent.select(*keys).clone()

    def __reduce__(self):
        if not self.is_closed:
            # ParallelEnv contains non-instantiated envs, thus it can be
            # closed and serialized if the environment building functions
            # permit it
            self.close()
        return super().__reduce__()

    def __getattr__(self, attr: str) -> Any:
        if attr in self.__dir__():
            return self.__getattribute__(
                attr
            )  # make sure that appropriate exceptions are raised
        elif attr.startswith("__"):
            raise AttributeError(
                "dispatching built-in private methods is not permitted."
            )
        else:
            try:
                _ = getattr(self._dummy_env, attr)
                if self.is_closed:
                    raise RuntimeError(
                        "Trying to access attributes of closed/non started "
                        "environments. Check that the batched environment "
                        "has been started (e.g. by calling env.reset)"
                    )
                # dispatch to workers
                return _dispatch_caller_parallel(attr, self)
            except AttributeError:
                raise AttributeError(
                    f"attribute {attr} not found in "
                    f"{self._dummy_env.__class__.__name__}"
                )


def _run_worker_pipe_shared_mem(
    idx: int,
    parent_pipe: connection.Connection,
    child_pipe: connection.Connection,
    env_fun: Union[_EnvClass, Callable],
    env_fun_kwargs: dict,
    pin_memory: bool,
    action_keys: dict,
    verbose: bool = False,
) -> None:
    parent_pipe.close()
    pid = os.getpid()
    if not isinstance(env_fun, _EnvClass):
        env = env_fun(**env_fun_kwargs)
    else:
        if env_fun_kwargs:
            raise RuntimeError(
                "env_fun_kwargs must be empty if an environment is passed to a process."
            )
        env = env_fun
    i = -1
    initialized = False

    # make sure that process can be closed
    tensordict = None
    _td = None
    data = None

    while True:
        try:
            cmd, data = child_pipe.recv()
        except EOFError:
            raise EOFError(f"proc {pid} failed, last command: {cmd}")
        if cmd == "seed":
            if not initialized:
                raise RuntimeError("call 'init' before closing")
            # torch.manual_seed(data)
            # np.random.seed(data)
            env.set_seed(data)
            child_pipe.send(("seeded", None))

        elif cmd == "init":
            if verbose:
                print(f"initializing {pid}")
            if initialized:
                raise RuntimeError("worker already initialized")
            i = 0
            tensordict = data
            if not (tensordict.is_shared() or tensordict.is_memmap()):
                raise RuntimeError(
                    "tensordict must be placed in shared memory (share_memory_() or memmap_())"
                )
            initialized = True

        elif cmd == "reset":
            reset_kwargs = data
            if verbose:
                print(f"resetting worker {pid}")
            if not initialized:
                raise RuntimeError("call 'init' before resetting")
            # _td = tensordict.select("observation").to(env.device).clone()
            _td = env.reset(**reset_kwargs)
            keys = set(_td.keys())
            if pin_memory:
                _td.pin_memory()
            tensordict.update_(_td)
            child_pipe.send(("reset_obs", keys))
            just_reset = True
            if env.is_done:
                raise RuntimeError(
                    f"{env.__class__.__name__}.is_done is {env.is_done} after reset"
                )

        elif cmd == "step":
            if not initialized:
                raise RuntimeError("called 'init' before step")
            i += 1
            _td = tensordict.select(*action_keys).to(env.device).clone()
            if env.is_done:
                raise RuntimeError(
                    f"calling step when env is done, just reset = {just_reset}"
                )
            _td = env.step(_td)
            keys = set(_td.keys()) - {key for key in action_keys}
            if pin_memory:
                _td.pin_memory()
            tensordict.update_(_td.select(*keys))
            if _td.get("done"):
                msg = "done"
            else:
                msg = "step_result"
            data = (msg, keys)
            child_pipe.send(data)
            just_reset = False

        elif cmd == "close":
            del tensordict, _td, data
            if not initialized:
                raise RuntimeError("call 'init' before closing")
            env.close()
            del env

            child_pipe.send(("closing", None))
            child_pipe.close()
            if verbose:
                print(f"{pid} closed")
            break

        elif cmd == "load_state_dict":
            env.load_state_dict(data)
            msg = "loaded"
            child_pipe.send((msg, None))

        elif cmd == "state_dict":
            state_dict = env.state_dict()
            msg = "state_dict"
            child_pipe.send((msg, state_dict))

        else:
            err_msg = f"{cmd} from env"
            try:
                attr = getattr(env, cmd)
                if callable(attr):
                    args, kwargs = data
                    args_replace = []
                    for _arg in args:
                        if isinstance(_arg, str) and _arg == "_self":
                            continue
                        else:
                            args_replace.append(_arg)
                    result = attr(*args_replace, **kwargs)
                else:
                    result = attr
            except Exception as err:
                raise RuntimeError(
                    f"querying {err_msg} resulted in the following error: " f"{err}"
                )
            child_pipe.send(("_".join([cmd, "done"]), result))
