# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Type, Union, Dict, Any

from torchrl.collectors.collectors import (
    _DataCollector,
    _MultiDataCollector,
    MultiaSyncDataCollector,
    MultiSyncDataCollector,
)
from torchrl.data import MultiStep
from torchrl.data.tensordict.tensordict import _TensorDict
from torchrl.envs import ParallelEnv

__all__ = [
    "sync_sync_collector",
    "sync_async_collector",
    "make_collector_offpolicy",
    "make_collector_onpolicy",
]

from torchrl.envs.common import _EnvClass
from torchrl.modules import TensorDictModuleWrapper, ProbabilisticTensorDictModule


def sync_async_collector(
    env_fns: Union[Callable, List[Callable]],
    env_kwargs: Optional[Union[dict, List[dict]]],
    num_env_per_collector: Optional[int] = None,
    num_collectors: Optional[int] = None,
    **kwargs,
) -> MultiaSyncDataCollector:
    """
    Runs asynchronous collectors, each running synchronous environments.

    .. aafig::


            +----------------------------------------------------------------------+
            |           "MultiConcurrentCollector"                |                |
            |~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|                |
            |  "Collector 1"  |  "Collector 2"  |  "Collector 3"  |     "Main"     |
            |~~~~~~~~~~~~~~~~~|~~~~~~~~~~~~~~~~~|~~~~~~~~~~~~~~~~~|~~~~~~~~~~~~~~~~|
            | "env1" | "env2" | "env3" | "env4" | "env5" | "env6" |                |
            |~~~~~~~~|~~~~~~~~|~~~~~~~~|~~~~~~~~|~~~~~~~~|~~~~~~~~|~~~~~~~~~~~~~~~~|
            |"reset" |"reset" |"reset" |"reset" |"reset" |"reset" |                |
            |        |        |        |        |        |        |                |
            |       "actor"   |        |        |       "actor"   |                |
            |                 |        |        |                 |                |
            | "step" | "step" |       "actor"   |                 |                |
            |        |        |                 |                 |                |
            |        |        |                 | "step" | "step" |                |
            |        |        |                 |        |        |                |
            |       "actor    | "step" | "step" |       "actor"   |                |
            |                 |        |        |                 |                |
            | "yield batch 1" |       "actor"   |                 |"collect, train"|
            |                 |                 |                 |                |
            | "step" | "step" |                 | "yield batch 2" |"collect, train"|
            |        |        |                 |                 |                |
            |        |        | "yield batch 3" |                 |"collect, train"|
            |        |        |                 |                 |                |
            +----------------------------------------------------------------------+

    Environment types can be identical or different. In the latter case, env_fns should be a list with all the creator
    fns for the various envs,
    and the policy should handle those envs in batch.

    Args:
        env_fns: Callable (or list of Callables) returning an instance of _EnvClass class.
        env_kwargs: Optional. Dictionary (or list of dictionaries) containing the kwargs for the environment being created.
        num_env_per_collector: Number of environments per data collector. The product
            num_env_per_collector * num_collectors should be less or equal to the number of workers available.
        num_collectors: Number of data collectors to be run in parallel.
        **kwargs: Other kwargs passed to the data collectors

    """

    return _make_collector(
        MultiaSyncDataCollector,
        env_fns=env_fns,
        env_kwargs=env_kwargs,
        num_env_per_collector=num_env_per_collector,
        num_collectors=num_collectors,
        **kwargs,
    )


def sync_sync_collector(
    env_fns: Union[Callable, List[Callable]],
    env_kwargs: Optional[Union[dict, List[dict]]],
    num_env_per_collector: Optional[int] = None,
    num_collectors: Optional[int] = None,
    **kwargs,
) -> MultiSyncDataCollector:
    """
    Runs synchronous collectors, each running synchronous environments.

    E.g.

    .. aafig::

            +----------------------------------------------------------------------+
            |            "MultiConcurrentCollector"               |                |
            |~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|                |
            |   "Collector 1" |  "Collector 2"  |  "Collector 3"  |     Main       |
            |~~~~~~~~~~~~~~~~~|~~~~~~~~~~~~~~~~~|~~~~~~~~~~~~~~~~~|~~~~~~~~~~~~~~~~|
            | "env1" | "env2" | "env3" | "env4" | "env5" | "env6" |                |
            |~~~~~~~~|~~~~~~~~|~~~~~~~~|~~~~~~~~|~~~~~~~~|~~~~~~~~|~~~~~~~~~~~~~~~~|
            |"reset" |"reset" |"reset" |"reset" |"reset" |"reset" |                |
            |        |        |        |        |        |        |                |
            |       "actor"   |        |        |       "actor"   |                |
            |                 |        |        |                 |                |
            | "step" | "step" |       "actor"   |                 |                |
            |        |        |                 |                 |                |
            |        |        |                 | "step" | "step" |                |
            |        |        |                 |        |        |                |
            |       "actor"   | "step" | "step" |       "actor"   |                |
            |                 |        |        |                 |                |
            |                 |       "actor"   |                 |                |
            |                 |                 |                 |                |
            |                       "yield batch of traj 1"------->"collect, train"|
            |                                                     |                |
            | "step" | "step" | "step" | "step" | "step" | "step" |                |
            |        |        |        |        |        |        |                |
            |       "actor"   |       "actor"   |        |        |                |
            |                 | "step" | "step" |       "actor"   |                |
            |                 |        |        |                 |                |
            | "step" | "step" |       "actor"   | "step" | "step" |                |
            |        |        |                 |        |        |                |
            |       "actor"   |                 |       "actor"   |                |
            |                       "yield batch of traj 2"------->"collect, train"|
            |                                                     |                |
            +----------------------------------------------------------------------+

    Envs can be identical or different. In the latter case, env_fns should be a list with all the creator fns
    for the various envs,
    and the policy should handle those envs in batch.

    Args:
        env_fns: Callable (or list of Callables) returning an instance of _EnvClass class.
        env_kwargs: Optional. Dictionary (or list of dictionaries) containing the kwargs for the environment being created.
        num_env_per_collector: Number of environments per data collector. The product
            num_env_per_collector * num_collectors should be less or equal to the number of workers available.
        num_collectors: Number of data collectors to be run in parallel.
        **kwargs: Other kwargs passed to the data collectors

    """
    return _make_collector(
        MultiSyncDataCollector,
        env_fns=env_fns,
        env_kwargs=env_kwargs,
        num_env_per_collector=num_env_per_collector,
        num_collectors=num_collectors,
        **kwargs,
    )


def _make_collector(
    collector_class: Type,
    env_fns: Union[Callable, List[Callable]],
    env_kwargs: Optional[Union[dict, List[dict]]],
    policy: Callable[[_TensorDict], _TensorDict],
    max_frames_per_traj: int = -1,
    frames_per_batch: int = 200,
    total_frames: Optional[int] = None,
    postproc: Optional[Callable] = None,
    num_env_per_collector: Optional[int] = None,
    num_collectors: Optional[int] = None,
    **kwargs,
) -> _MultiDataCollector:
    if env_kwargs is None:
        env_kwargs = dict()
    if isinstance(env_fns, list):
        num_env = len(env_fns)
        if num_env_per_collector is None:
            num_env_per_collector = -(num_env // -num_collectors)
        elif num_collectors is None:
            num_collectors = -(num_env // -num_env_per_collector)
        else:
            if num_env_per_collector * num_collectors < num_env:
                raise ValueError(
                    f"num_env_per_collector * num_collectors={num_env_per_collector * num_collectors} "
                    f"has been found to be less than num_env={num_env}"
                )
    else:
        try:
            num_env = num_env_per_collector * num_collectors
            env_fns = [env_fns for _ in range(num_env)]
        except (TypeError):
            raise Exception(
                "num_env was not a list but num_env_per_collector and num_collectors were not both specified,"
                f"got num_env_per_collector={num_env_per_collector} and num_collectors={num_collectors}"
            )
    if not isinstance(env_kwargs, list):
        env_kwargs = [env_kwargs for _ in range(num_env)]

    env_fns_split = [
        env_fns[i : i + num_env_per_collector]
        for i in range(0, num_env, num_env_per_collector)
    ]
    env_kwargs_split = [
        env_kwargs[i : i + num_env_per_collector]
        for i in range(0, num_env, num_env_per_collector)
    ]
    if len(env_fns_split) != num_collectors:
        raise RuntimeError(
            f"num_collectors={num_collectors} differs from len(env_fns_split)={len(env_fns_split)}"
        )

    if num_env_per_collector == 1:
        env_fns = [_env_fn[0] for _env_fn in env_fns_split]
        env_kwargs = [_env_kwargs[0] for _env_kwargs in env_kwargs_split]
    else:
        env_fns = [
            lambda: ParallelEnv(
                num_workers=len(_env_fn),
                create_env_fn=_env_fn,
                create_env_kwargs=_env_kwargs,
            )
            for _env_fn, _env_kwargs in zip(env_fns_split, env_kwargs_split)
        ]
        env_kwargs = None
    return collector_class(
        create_env_fn=env_fns,
        create_env_kwargs=env_kwargs,
        policy=policy,
        total_frames=total_frames,
        max_frames_per_traj=max_frames_per_traj,
        frames_per_batch=frames_per_batch,
        postproc=postproc,
        **kwargs,
    )


def make_collector_offpolicy(
    make_env: Callable[[], _EnvClass],
    actor_model_explore: Union[TensorDictModuleWrapper, ProbabilisticTensorDictModule],
    cfg: "DictConfig",
    make_env_kwargs: Optional[Dict] = None,
) -> _DataCollector:
    """
    Returns a data collector for off-policy algorithms.

    Args:
        make_env (Callable): environment creator
        actor_model_explore (TensorDictModule): Model instance used for evaluation and exploration update
        cfg (DictConfig): config for creating collector object
        make_env_kwargs (dict): kwargs for the env creator

    """
    if cfg.async_collection:
        collector_helper = sync_async_collector
    else:
        collector_helper = sync_sync_collector

    if cfg.multi_step:
        ms = MultiStep(
            gamma=cfg.gamma,
            n_steps_max=cfg.n_steps_return,
        )
    else:
        ms = None

    env_kwargs = {}
    if make_env_kwargs is not None and isinstance(make_env_kwargs, dict):
        env_kwargs.update(make_env_kwargs)
    elif make_env_kwargs is not None:
        env_kwargs = make_env_kwargs
    cfg.collector_devices = (
        cfg.collector_devices
        if len(cfg.collector_devices) > 1
        else cfg.collector_devices[0]
    )
    collector_helper_kwargs = {
        "env_fns": make_env,
        "env_kwargs": env_kwargs,
        "policy": actor_model_explore,
        "max_frames_per_traj": cfg.max_frames_per_traj,
        "frames_per_batch": cfg.frames_per_batch,
        "total_frames": cfg.total_frames,
        "postproc": ms,
        "num_env_per_collector": 1,
        # we already took care of building the make_parallel_env function
        "num_collectors": -cfg.num_workers // -cfg.env_per_collector,
        "devices": cfg.collector_devices,
        "passing_devices": cfg.collector_devices,
        "init_random_frames": cfg.init_random_frames,
        "pin_memory": cfg.pin_memory,
        "split_trajs": ms is not None,
        # trajectories must be separated if multi-step is used
        "init_with_lag": cfg.init_with_lag if hasattr(cfg, "init_with_lag") else False,
        "exploration_mode": cfg.exploration_mode,
    }

    collector = collector_helper(**collector_helper_kwargs)
    collector.set_seed(cfg.seed)
    return collector


def make_collector_onpolicy(
    make_env: Callable[[], _EnvClass],
    actor_model_explore: Union[TensorDictModuleWrapper, ProbabilisticTensorDictModule],
    cfg: "DictConfig",
    make_env_kwargs: Optional[Dict] = None,
) -> _DataCollector:
    collector_helper = sync_sync_collector

    ms = None

    env_kwargs = {}
    if make_env_kwargs is not None and isinstance(make_env_kwargs, dict):
        env_kwargs.update(make_env_kwargs)
    elif make_env_kwargs is not None:
        env_kwargs = make_env_kwargs
    cfg.collector_devices = (
        cfg.collector_devices
        if len(cfg.collector_devices) > 1
        else cfg.collector_devices[0]
    )
    collector_helper_kwargs = {
        "env_fns": make_env,
        "env_kwargs": env_kwargs,
        "policy": actor_model_explore,
        "max_frames_per_traj": cfg.max_frames_per_traj,
        "frames_per_batch": cfg.frames_per_batch,
        "total_frames": cfg.total_frames,
        "postproc": ms,
        "num_env_per_collector": 1,
        # we already took care of building the make_parallel_env function
        "num_collectors": -cfg.num_workers // -cfg.env_per_collector,
        "devices": cfg.collector_devices,
        "passing_devices": cfg.collector_devices,
        "pin_memory": cfg.pin_memory,
        "split_trajs": True,
        # trajectories must be separated in online settings
        "init_with_lag": cfg.init_with_lag if hasattr(cfg, "init_with_lag") else False,
        "exploration_mode": cfg.exploration_mode,
    }

    collector = collector_helper(**collector_helper_kwargs)
    collector.set_seed(cfg.seed)
    return collector


@dataclass
class OnPolicyCollectorConfig:
    collector_devices: Any = field(default_factory=lambda: ["cpu"])
    # device on which the data collector should store the trajectories to be passed to this script.
    # If the collector device differs from the policy device (cuda:0 if available), then the
    # weights of the collector policy are synchronized with collector.update_policy_weights_().
    pin_memory: bool = False
    # if True, the data collector will call pin_memory before dispatching tensordicts onto the passing device
    init_with_lag: bool = False
    # if True, the first trajectory will be truncated earlier at a random step. This is helpful
    # to desynchronize the environments, such that steps do no match in all collected
    # rollouts. Especially useful for online training, to prevent cyclic sample indices.
    frames_per_batch: int = 1000
    # number of steps executed in the environment per collection.
    # This value represents how many steps will the data collector execute and return in *each*
    # environment that has been created in between two rounds of optimization
    # (see the optim_steps_per_batch above).
    # On the one hand, a low value will enhance the data throughput between processes in async
    # settings, which can make the accessing of data a computational bottleneck.
    # High values will on the other hand lead to greater tensor sizes in memory and disk to be
    # written and read at each global iteration. One should look at the number of frames per second
    # in the log to assess the efficiency of the configuration.
    total_frames: int = 50000000
    # total number of frames collected for training. Does account for frame_skip (i.e. will be
    # divided by the frame_skip). Default=50e6.
    num_workers: int = 32
    # Number of workers used for data collection.
    env_per_collector: int = 8
    # Number of environments per collector. If the env_per_collector is in the range:
    # 1<env_per_collector<=num_workers, then the collector runs
    # ceil(num_workers/env_per_collector) in parallel and executes the policy steps synchronously
    # for each of these parallel wrappers. If env_per_collector=num_workers, no parallel wrapper is created
    seed: int = 42
    # seed used for the environment, pytorch and numpy.
    exploration_mode: str = ""
    # exploration mode of the data collector.
    async_collection: bool = False
    # whether data collection should be done asynchrously. Asynchrounous data collection means
    # that the data collector will keep on running the environment with the previous weights
    # configuration while the optimization loop is being done. If the algorithm is trained
    # synchronously, data collection and optimization will occur iteratively, not concurrently.


@dataclass
class OffPolicyCollectorConfig(OnPolicyCollectorConfig):
    multi_step: bool = False
    # whether or not multi-step rewards should be used.
    n_steps_return: int = 3
    # If multi_step is set to True, this value defines the number of steps to look ahead for the reward computation.
    init_random_frames: int = 50000
