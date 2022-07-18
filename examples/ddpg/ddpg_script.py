# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


# Disclaimer: This is "flat" (single script) implementation of DDPG.
# The ddpg.py file abstracts away most of these components in a more efficient
# and modular way.


import dataclasses
import uuid
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import hydra
import torch.cuda
import tqdm
from hydra.core.config_store import ConfigStore
from hypothesis.strategies._internal.core import kwargs
from torch import optim, distributions as d
from torchrl.data import CompositeSpec
from torchrl.envs import (
    ParallelEnv,
    EnvCreator,
    DMControlEnv,
    CatTensors,
    ObservationNorm,
    DoubleToFloat,
    gSDENoise,
)
from torchrl.envs.transforms import RewardScaling, TransformedEnv
from torchrl.envs.utils import set_exploration_mode, step_tensordict
from torchrl.modules import (
    OrnsteinUhlenbeckProcessWrapper,
    MLP,
    TensorDictModule,
    TensorDictSequence,
    ProbabilisticActor,
    ValueOperator,
)
from torchrl.modules.distributions.continuous import SafeTanhTransform, TanhDelta
from torchrl.modules.models.exploration import LazygSDEModule
from torchrl.objectives import hold_out_params
from torchrl.record import VideoRecorder
from torchrl.trainers import Recorder
from torchrl.trainers.helpers.collectors import (
    make_collector_offpolicy,
)
from torchrl.trainers.helpers.envs import (
    correct_for_frame_skip,
    get_stats_random_rollout,
    LIBS,
)
from torchrl.trainers.helpers.models import (
    make_ddpg_actor,
    ACTIVATIONS,
)
from torchrl.trainers.helpers.replay_buffer import (
    make_replay_buffer,
)


@dataclass
class TrainerConfig:
    optim_steps_per_batch: int = 1000
    # Number of optimization steps in between two collection of data. See frames_per_batch below.
    batch_size: int = 256
    # batch size of the TensorDict retrieved from the replay buffer. Default=256.
    log_interval: int = 10000
    # logging interval, in terms of optimization steps. Default=10000.
    lr: float = 3e-4
    # Learning rate used for the optimizer. Default=3e-4.
    weight_decay: float = 0.0
    # Weight-decay to be used with the optimizer. Default=0.0.
    clip_norm: float = 1000.0
    # value at which the total gradient norm should be clipped. Default=1000.0
    normalize_rewards_online: bool = True
    # Computes the running statistics of the rewards and normalizes them before they are passed to the loss module.
    normalize_rewards_online_scale: float = 5.0
    # Scale factor of the normalized rewards.


@dataclass
class OffPolicyCollectorConfig:
    collector_devices: Any = dataclasses.field(default_factory=lambda: ["cpu"])
    # device on which the data collector should store the trajectories to be passed to this script.
    # If the collector device differs from the policy device (cuda:0 if available), then the
    # weights of the collector policy are synchronized with collector.update_policy_weights_().
    pin_memory: bool = False
    # if True, the data collector will call pin_memory before dispatching tensordicts onto the passing device
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
    total_frames: int = 1000000
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
    async_collection: bool = True
    # whether data collection should be done asynchrously. Asynchrounous data collection means
    # that the data collector will keep on running the environment with the previous weights
    # configuration while the optimization loop is being done. If the algorithm is trained
    # synchronously, data collection and optimization will occur iteratively, not concurrently.
    multi_step: bool = False
    # whether or not multi-step rewards should be used.
    n_steps_return: int = 3
    # If multi_step is set to True, this value defines the number of steps to look ahead for the reward computation.
    init_random_frames: int = 50000
    exploration_mode: str = ""


@dataclass
class EnvConfig:
    env_library: str = "gym"
    # env_library used for the simulated environment. Default=gym
    env_name: str = "HalfCheetah-v4"
    # name of the environment to be created. Default=Humanoid-v2
    env_task: str = ""
    # task (if any) for the environment. Default=run
    from_pixels: bool = False
    # whether the environment output should be state vector(s) (default) or the pixels.
    frame_skip: int = 1
    # frame_skip for the environment. Note that this value does NOT impact the buffer size,
    # maximum steps per trajectory, frames per batch or any other factor in the algorithm,
    # e.g. if the total number of frames that has to be computed is 50e6 and the frame skip is 4
    # the actual number of frames retrieved will be 200e6. Default=1.
    reward_scaling: Optional[float] = None
    # scale of the reward.
    init_env_steps: int = 1000
    # number of random steps to compute normalizing constants
    norm_stats: bool = True
    # Deactivates the normalization based on random collection of data.
    center_crop: Any = dataclasses.field(default_factory=lambda: [])
    # center crop size.
    grayscale: bool = True
    # Disables grayscale transform.
    max_frames_per_traj: int = 10000
    # Number of steps before a reset of the environment is called (if it has not been flagged as done before).


@dataclass
class LossConfig:
    hard_update: bool = False
    # whether soft-update should be used with double SAC loss (default) or hard updates.
    loss_function: str = "l2"
    # loss function for the value network. Either one of l1, l2 or smooth_l1.
    tau: float = 0.002
    # tau factor for target network update
    gamma: float = 0.99
    # Decay factor for return computation. Default=0.99.
    target_entropy: Any = None
    # Target entropy for the policy distribution. Default is None (auto calculated as the `target_entropy = -action_dim`)


@dataclass
class DDPGModelConfig:
    annealing_frames: int = 1000000
    # float of frames used for annealing of the OrnsteinUhlenbeckProcess. Default=1e6.
    ou_exploration: bool = True
    # wraps the policy in an OU exploration wrapper, similar to DDPG. SAC being designed for
    # efficient entropy-based exploration, this should be left for experimentation only.
    ou_sigma: float = 0.2
    # Ornstein-Uhlenbeck sigma
    ou_theta: float = 0.15
    # Aimed at superseeding --ou_exploration.
    gSDE: bool = False
    # if True, exploration is achieved using the gSDE technique.
    activation: str = "elu"
    # activation function, either relu or elu or tanh, Default=tanh
    num_cells: int = 256
    # Number of cells in the MLP
    num_layers: int = 2
    # Number of layers in the MLP


@dataclass
class RecorderConfig:
    record_video: bool = True
    # whether a video of the task should be rendered during logging.
    video_tag: str = ""
    # video tag. If not specified, the exp_name will be used.
    exp_name: str = ""
    # experiment name. Used for logging directory.
    # A date and uuid will be joined to account for multiple experiments with the same name.
    record_interval: int = 1000
    # number of batch collections in between two collections of validation rollouts. Default=1000.
    record_frames: int = 1000
    # number of steps in validation rollouts. " "Default=1000.
    recorder_log_keys: Any = dataclasses.field(default_factory=lambda: ["reward"])
    # Keys to log in the recorder


@dataclass
class ReplayArgsConfig:
    buffer_size: int = 1000000
    # buffer size, in number of frames stored. Default=1e6
    prb: bool = True
    # whether a Prioritized replay buffer should be used instead of a more basic circular one.
    buffer_scratch_dir: Optional[str] = None
    # directory where the buffer data should be stored. If none is passed, they will be placed in /tmp/
    buffer_prefetch: int = 32
    # prefetching queue length for the replay buffer


config_fields = [
    (config_field.name, config_field.type, config_field)
    for config_cls in (
        TrainerConfig,
        OffPolicyCollectorConfig,
        EnvConfig,
        LossConfig,
        DDPGModelConfig,
        RecorderConfig,
        ReplayArgsConfig,
    )
    for config_field in dataclasses.fields(config_cls)
]
Config = dataclasses.make_dataclass(cls_name="Config", fields=config_fields)
cs = ConfigStore.instance()
cs.store(name="config_script", node=Config)

DEFAULT_REWARD_SCALING = {
    "Hopper-v1": 5,
    "Walker2d-v1": 5,
    "HalfCheetah-v1": 5,
    "cheetah": 5,
    "Ant-v2": 5,
    "Humanoid-v2": 20,
    "humanoid": 100,
}


def make_env(cfg):
    """
    Create a base env

    """
    env_name = cfg.env_name
    env_task = cfg.env_task
    env_library = LIBS[cfg.env_library]
    frame_skip = cfg.frame_skip
    from_pixels = cfg.from_pixels

    env_kwargs = {
        "env_name": env_name,
        "device": "cpu",
        "frame_skip": frame_skip,
        "from_pixels": from_pixels or cfg.record_video,
        "pixels_only": from_pixels,
    }
    if env_library is DMControlEnv:
        env_kwargs.update({"task_name": env_task})
    env = env_library(**env_kwargs)
    return env


def make_transformed_env(
    env, cfg, stats=None, action_dim_gsde=None, state_dim_gsde=None
):
    """
    Apply transforms to the env (such as reward scaling and state normalization)

    """
    env_library = cfg.env_library

    env = TransformedEnv(env)

    if cfg.reward_scaling:
        env.append_transform(RewardScaling(loc=0.0, scale=cfg.reward_scaling))

    double_to_float_list = []
    double_to_float_inv_list = []
    if env_library is DMControlEnv:
        # DMControl requires double-precision
        double_to_float_list += [
            "reward",
        ]
        double_to_float_inv_list += ["action"]

    selected_keys = [key for key in env.observation_spec.keys() if "pixels" not in key]

    # We concatenate all states into a single "next_observation_vector"
    # even if there is a single tensor, it'll be renamed in "next_observation_vector"
    out_key = "next_observation_vector"
    env.append_transform(CatTensors(keys_in=selected_keys, out_key=out_key))

    #  we normalize the states
    if stats is None:
        _stats = {"loc": 0.0, "scale": 1.0}
    else:
        _stats = stats
    env.append_transform(
        ObservationNorm(**_stats, keys_in=[out_key], standard_normal=True)
    )

    double_to_float_list.append(out_key)
    env.append_transform(
        DoubleToFloat(
            keys_in=double_to_float_list, keys_inv_in=double_to_float_inv_list
        )
    )

    if cfg.gSDE:
        env.append_transform(
            gSDENoise(action_dim=action_dim_gsde, state_dim=state_dim_gsde)
        )

    return env


def parallel_env_constructor(
    cfg,
    stats,
    **env_kwargs,
):
    if cfg.env_per_collector == 1:
        env_creator = EnvCreator(
            lambda: make_transformed_env(make_env(cfg), cfg, stats, **env_kwargs)
        )
        return env_creator

    parallel_env = ParallelEnv(
        num_workers=cfg.env_per_collector,
        create_env_fn=EnvCreator(lambda: make_env(cfg)),
        create_env_kwargs=None,
        pin_memory=cfg.pin_memory,
    )
    env = make_transformed_env(parallel_env, cfg, stats, **env_kwargs)
    return env


def get_env_stats(cfg):
    """
    Gets the stats of an environment

    """
    proof_env = make_transformed_env(make_env(cfg), cfg, None)
    stats = get_stats_random_rollout(
        cfg, proof_env, key="next_pixels" if cfg.from_pixels else None
    )
    # make sure proof_env is closed
    proof_env.close()
    return stats


def make_ddpg_actor(
    cfg,
    stats,
    device="cpu",
):
    proof_environment = make_transformed_env(make_env(cfg), cfg, stats)

    from_pixels = cfg.from_pixels

    env_specs = proof_environment.specs
    out_features = env_specs["action_spec"].shape[0]

    if from_pixels:
        raise NotImplementedError
    else:
        actor_net = MLP(
            num_cells=[cfg.num_cells] * cfg.num_layers,
            activation_class=ACTIVATIONS[cfg.activation],
            out_features=out_features,
        )
        in_keys = ["observation_vector"]
        gSDE_state_key = "observation_vector"
        out_keys = ["param"]

    actor_module = TensorDictModule(actor_net, in_keys=in_keys, out_keys=out_keys)

    if cfg.gSDE:
        min = env_specs["action_spec"].space.minimum
        max = env_specs["action_spec"].space.maximum
        transform = SafeTanhTransform()
        if (min != -1).any() or (max != 1).any():
            transform = d.ComposeTransform(
                transform, d.AffineTransform(loc=(max + min) / 2, scale=(max - min) / 2)
            )
        actor_module = TensorDictSequence(
            actor_module,
            TensorDictModule(
                LazygSDEModule(transform=transform, learn_sigma=False),
                in_keys=["param", gSDE_state_key, "_eps_gSDE"],
                out_keys=["loc", "scale", "action", "_eps_gSDE"],
            ),
        )

    # We use a ProbabilisticActor to make sure that we map the network output
    # to the right space using a TanhDelta distribution.
    actor = ProbabilisticActor(
        module=actor_module,
        dist_param_keys=["param"],
        spec=CompositeSpec(action=env_specs["action_spec"]),
        safe=True,
        distribution_class=TanhDelta,
        distribution_kwargs={
            "min": env_specs["action_spec"].space.minimum,
            "max": env_specs["action_spec"].space.maximum,
        },
    ).to(device)

    if from_pixels:
        raise NotImplementedError
    else:
        q_net = MLP(
            num_cells=[cfg.num_cells] * cfg.num_layers,
            activation_class=ACTIVATIONS[cfg.activation],
            out_features=1,
        )

    in_keys = in_keys + ["action"]
    qnet = ValueOperator(
        in_keys=in_keys,
        module=q_net,
    ).to(device)

    # init: since we have lazy layers, we should run the network once to initialize them
    with torch.no_grad(), set_exploration_mode("random"):
        td = proof_environment.rollout(max_steps=1000)
        td = td.to(device)
        actor(td)
        qnet(td)

    return actor, qnet


def make_recorder(cfg, writer, actor_model_explore, stats):
    env_name = cfg.env_name
    video_tag = cfg.video_tag

    base_env = make_env(cfg)
    transformed_env = make_transformed_env(base_env, cfg, stats)
    recorder = TransformedEnv(base_env)
    # a bit of baking to make the recorder:
    if cfg.record_video:
        center_crop = cfg.center_crop
        if center_crop:
            center_crop = center_crop[0]
        recorder.append_transform(
            VideoRecorder(
                writer=writer,
                tag=f"{video_tag}_{env_name}_video",
                center_crop=center_crop,
            ),
        )
    for transform in transformed_env.transform:
        if not isinstance(transform, RewardScaling):
            recorder.append_transform(transform)

    recorder_obj = Recorder(
        record_frames=cfg.record_frames,
        frame_skip=cfg.frame_skip,
        policy_exploration=actor_model_explore,
        recorder=recorder,
        record_interval=cfg.record_interval,
        log_keys=cfg.recorder_log_keys,
    )
    return recorder_obj


@hydra.main(version_base=None, config_path=".", config_name="config_script")
def main(cfg: "DictConfig"):
    from torch.utils.tensorboard import SummaryWriter

    cfg = correct_for_frame_skip(cfg)
    print(cfg)

    # define experiment name
    exp_name = "_".join(
        [
            "DDPG",
            cfg.exp_name,
            str(uuid.uuid4())[:8],
            datetime.now().strftime("%y_%m_%d-%H_%M_%S"),
        ]
    )
    if cfg.record_video and not len(cfg.video_tag):
        cfg.video_tag = exp_name + "_video"
    writer = SummaryWriter(f"ddpg_logging/{exp_name}")

    # get stats for normalization
    stats = get_env_stats(cfg)

    # set reward normalization
    if not isinstance(cfg.reward_scaling, float) and not cfg.normalize_rewards_online:
        cfg.reward_scaling = DEFAULT_REWARD_SCALING.get(cfg.env_name, 5.0)

    # execute on cuda if available
    device = (
        torch.device("cpu")
        if torch.cuda.device_count() == 0
        else torch.device("cuda:0")
    )

    # Actor and qnet instantiation
    actor, qnet = make_ddpg_actor(
        cfg=cfg,
        stats=stats,
        device=device,
    )
    # Target network
    qnet_target = deepcopy(qnet).requires_grad_(False)

    # Exploration wrappers:
    actor_model_explore = actor
    if cfg.ou_exploration:
        if cfg.gSDE:
            raise RuntimeError("gSDE and ou_exploration are incompatible")
        actor_model_explore = OrnsteinUhlenbeckProcessWrapper(
            actor_model_explore,
            annealing_num_steps=cfg.annealing_frames,
            sigma=cfg.ou_sigma,
            theta=cfg.ou_theta,
        ).to(device)
    if device == torch.device("cpu"):
        # mostly for debugging
        actor_model_explore.share_memory()

    env_kwargs = {}
    if cfg.gSDE:
        proof_env = make_transformed_env(make_env(cfg), cfg)
        with torch.no_grad(), set_exploration_mode("random"):
            # get dimensions to build the parallel env
            proof_td = actor_model_explore(proof_env.reset().to(device))
        action_dim_gsde, state_dim_gsde = proof_td.get("_eps_gSDE").shape[-2:]
        env_kwargs["action_dim_gsde"] = action_dim_gsde
        env_kwargs["state_dim_gsde"] = state_dim_gsde
        proof_env.close()
        del proof_td, proof_env

    # Environment setting:
    create_env_fn = parallel_env_constructor(
        cfg=cfg,
        stats=stats,
        **env_kwargs,
    )

    # Batch collector:
    collector = make_collector_offpolicy(
        make_env=create_env_fn,
        actor_model_explore=actor_model_explore,
        cfg=cfg,
    )

    # Replay buffer:
    replay_buffer = make_replay_buffer(device, cfg)

    # trajectory recorder
    recorder = make_recorder(cfg, writer, actor_model_explore, stats)

    # Optimizer
    optimizer_actor = optim.Adam(
        actor.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    optimizer_qnet = optim.Adam(
        qnet.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    # Main loop
    collected_frames = 0
    pbar = tqdm.tqdm(total=cfg.total_frames)
    for i, tensordict in enumerate(collector):
        pbar.update(tensordict.numel())
        # extend the replay buffer with the new data
        if "mask" in tensordict.keys():
            # if multi-step, a mask is present to help filter padded values
            current_frames = tensordict["mask"].sum()
            tensordict = tensordict[tensordict.get("mask").squeeze(-1)]
        else:
            tensordict = tensordict.view(-1)
            current_frames = tensordict.numel()
        collected_frames += current_frames
        replay_buffer.extend(tensordict.cpu())

        # optimization steps
        if collected_frames >= cfg.init_random_frames:
            for j in range(cfg.optim_steps_per_batch):
                # sample from replay buffer
                sampled_tensordict = replay_buffer.sample(cfg.batch_size)

                # compute loss for qnet and backprop
                with hold_out_params(actor.parameters()):
                    # get next state value
                    next_tensordict = step_tensordict(sampled_tensordict)
                    qnet_target(actor(next_tensordict))
                    next_value = next_tensordict["state_action_value"]
                value_est = (
                    sampled_tensordict["reward"]
                    + cfg.gamma * (1 - sampled_tensordict["done"].float()) * next_value
                )
                value = qnet(sampled_tensordict)["state_action_value"]
                value_loss = (value - value_est).pow(2).mean()
                # we write the td_error in the sampled_tensordict for priority update
                # because the indices of the samples is tracked in sampled_tensordict
                # and the replay buffer will know which priorities to update.
                sampled_tensordict["td_error"] = abs(value - value_est)

                value_loss.backward()
                # TODO: clip gradients
                optimizer_qnet.step()
                optimizer_qnet.zero_grad()

                # compute loss for actor and backprop
                sampled_tensordict_actor = sampled_tensordict.select(*actor.in_keys)
                with hold_out_params(qnet.parameters()):
                    qnet(actor(sampled_tensordict_actor))
                actor_loss = sampled_tensordict_actor["state_action_value"]
                actor_loss.mean().backward()
                # TODO: clip gradients
                optimizer_actor.step()
                optimizer_actor.zero_grad()

                # update qnet_target params
                for (p_in, p_dest) in zip(qnet.parameters(), qnet_target.parameters()):
                    p_dest.data.copy_(cfg.tau * p_in.data + (1 - cfg.tau) * p_dest.data)
                for (p_in, p_dest) in zip(qnet.buffers(), qnet_target.buffers()):
                    p_dest.data.copy_(cfg.tau * p_in.data + (1 - cfg.tau) * p_dest.data)

                # update priority
                if cfg.prb:
                    replay_buffer.update_priority(sampled_tensordict)

        # update the exploration strategy
        actor_model_explore.step(current_frames)

        # some logging
        writer.add_scalar(
            "r_training", tensordict.get("reward").mean(), global_step=collected_frames
        )
        recorder(None)

        # update weights of the inference policy
        collector.update_policy_weights_()


if __name__ == "__main__":
    main()
