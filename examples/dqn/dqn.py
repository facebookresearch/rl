# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import uuid
from datetime import datetime

from torchrl.envs import ParallelEnv, EnvCreator
from torchrl.record import VideoRecorder

try:
    import configargparse as argparse

    _configargparse = True
except ImportError:
    import argparse

    _configargparse = False
import torch.cuda
from torchrl.envs.transforms import RewardScaling, TransformedEnv
from torchrl.modules import EGreedyWrapper
from torchrl.trainers.helpers.collectors import (
    make_collector_offpolicy,
    parser_collector_args_offpolicy,
)
from torchrl.trainers.helpers.envs import (
    correct_for_frame_skip,
    get_stats_random_rollout,
    parallel_env_constructor,
    parser_env_args,
    transformed_env_constructor,
)
from torchrl.trainers.helpers.losses import make_dqn_loss, parser_loss_args
from torchrl.trainers.helpers.models import (
    make_dqn_actor,
    parser_model_args_discrete,
)
from torchrl.trainers.helpers.recorder import parser_recorder_args
from torchrl.trainers.helpers.replay_buffer import (
    make_replay_buffer,
    parser_replay_args,
)
from torchrl.trainers.helpers.trainers import make_trainer, parser_trainer_args


def make_args():
    parser = argparse.ArgumentParser()
    if _configargparse:
        parser.add_argument(
            "-c",
            "--config",
            required=True,
            is_config_file=True,
            help="config file path",
        )
    parser_trainer_args(parser)
    parser_collector_args_offpolicy(parser)
    parser_env_args(parser)
    parser_loss_args(parser, algorithm="DQN")
    parser_model_args_discrete(parser)
    parser_recorder_args(parser)
    parser_replay_args(parser)
    return parser


parser = make_args()


def main(args):
    from torch.utils.tensorboard import SummaryWriter

    args = correct_for_frame_skip(args)

    if not isinstance(args.reward_scaling, float):
        args.reward_scaling = 1.0

    device = (
        torch.device("cpu")
        if torch.cuda.device_count() == 0
        else torch.device("cuda:0")
    )

    exp_name = "_".join(
        [
            "DQN",
            args.exp_name,
            str(uuid.uuid4())[:8],
            datetime.now().strftime("%y_%m_%d-%H_%M_%S"),
        ]
    )
    writer = SummaryWriter(f"dqn_logging/{exp_name}")
    video_tag = exp_name if args.record_video else ""

    stats = None
    if not args.vecnorm and args.norm_stats:
        proof_env = transformed_env_constructor(args=args, use_env_creator=False)()
        stats = get_stats_random_rollout(
            args, proof_env, key="next_pixels" if args.from_pixels else None
        )
        # make sure proof_env is closed
        proof_env.close()
    elif args.from_pixels:
        stats = {"loc": 0.5, "scale": 0.5}
    proof_env = transformed_env_constructor(
        args=args, use_env_creator=False, stats=stats
    )()
    model = make_dqn_actor(
        proof_environment=proof_env,
        args=args,
        device=device,
    )

    loss_module, target_net_updater = make_dqn_loss(model, args)
    model_explore = EGreedyWrapper(model, annealing_num_steps=args.annealing_frames).to(
        device
    )
    action_dim_gsde, state_dim_gsde = None, None
    proof_env.close()
    create_env_fn = parallel_env_constructor(
        args=args,
        stats=stats,
        action_dim_gsde=action_dim_gsde,
        state_dim_gsde=state_dim_gsde,
    )

    collector = make_collector_offpolicy(
        make_env=create_env_fn,
        actor_model_explore=model_explore,
        args=args,
        # make_env_kwargs=[
        #     {"device": device} if device >= 0 else {}
        #     for device in args.env_rendering_devices
        # ],
    )

    replay_buffer = make_replay_buffer(device, args)

    recorder = transformed_env_constructor(
        args,
        video_tag=video_tag,
        norm_obs_only=True,
        stats=stats,
        writer=writer,
    )()

    # remove video recorder from recorder to have matching state_dict keys
    if args.record_video:
        recorder_rm = TransformedEnv(recorder.env)
        for transform in recorder.transform:
            if not isinstance(transform, VideoRecorder):
                recorder_rm.append_transform(transform)
    else:
        recorder_rm = recorder

    if isinstance(create_env_fn, ParallelEnv):
        recorder_rm.load_state_dict(create_env_fn.state_dict()["worker0"])
        create_env_fn.close()
    elif isinstance(create_env_fn, EnvCreator):
        recorder_rm.load_state_dict(create_env_fn().state_dict())
    else:
        recorder_rm.load_state_dict(create_env_fn.state_dict())
    # reset reward scaling
    for t in recorder.transform:
        if isinstance(t, RewardScaling):
            t.scale.fill_(1.0)
            t.loc.fill_(0.0)

    trainer = make_trainer(
        collector,
        loss_module,
        recorder,
        target_net_updater,
        model_explore,
        replay_buffer,
        writer,
        args,
    )

    def select_keys(batch):
        return batch.select(
            "reward",
            "done",
            "steps_to_next_obs",
            "pixels",
            "next_pixels",
            "observation_vector",
            "next_observation_vector",
            "action",
        )

    trainer.register_op("batch_process", select_keys)

    final_seed = collector.set_seed(args.seed)
    print(f"init seed: {args.seed}, final seed: {final_seed}")

    trainer.train()
    return (writer.log_dir, trainer._log_dict, trainer.state_dict())


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
