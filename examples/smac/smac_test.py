from env import SCEnv
from torch import nn
from torchrl.agents.helpers import sync_async_collector
from torchrl.data import TensorDictPrioritizedReplayBuffer
from torchrl.envs import TransformedEnv, ObservationNorm
from torchrl.modules import (
    ProbabilisticTDModule,
    OneHotCategorical,
    QValueActor,
    MaskedLogitPolicy,
)

if __name__ == "__main__":
    # create an env
    env = SCEnv("8m")

    # reset
    td = env.reset()
    print("tensordict after reset: ")
    print(td)

    # apply a sequence of transforms
    env = TransformedEnv(env, ObservationNorm(0, 1, standard_normal=True))

    # Get policy
    policy = nn.LazyLinear(env.action_spec.shape[-1])
    policy_wrap = MaskedLogitPolicy(policy)
    policy_td_module = ProbabilisticTDModule(
        module=policy_wrap,
        spec=None,
        in_keys=["observation", "available_actions"],
        out_keys=["action"],
        distribution_class=OneHotCategorical,
        save_dist_params=True,
    )

    # Test the policy
    policy_td_module(td)
    print(td)
    print("param: ", td.get("action_dist_param_0"))
    print("action: ", td.get("action"))
    print("mask: ", td.get("available_actions"))
    print("mask from env: ", env.env._env.get_avail_actions())

    # check that an ation can be performed in the env with this
    env.step(td)
    print(td)

    # we can also have a regular Q-Value actor
    print("\n\nQValue")
    policy_td_module = QValueActor(
        policy_wrap,
        spec=None,
        in_keys=["observation", "available_actions"],
        # out_keys=["actions"]
    )
    td = env.reset()
    policy_td_module(td)
    print("action: ", td.get("action"))
    env.step(td)
    print("next_obs: ", td.get("next_observation"))

    # now let's collect data, see MultiaSyncDataCollector for info
    print("\n\nCollector")
    collector = sync_async_collector(
        env_fns=lambda: SCEnv("8m"),
        env_kwargs=None,
        num_collectors=4,  # 4 main processes
        num_env_per_collector=8,  # each of the 4 collectors has 8 processes
        policy=policy_td_module,
        devices=["cuda:0"] * 4,  # each collector will execute the policy on cuda
        total_frames=1000,  # we'd like to have a total of 1000 frames
        max_frames_per_traj=10,  # we'll reset after 10 steps
        frames_per_batch=64,  # each batch should have 64 frames
        init_random_frames=0,  # we won't execute random actions
    )
    print("replay buffer")
    rb = TensorDictPrioritizedReplayBuffer(size=100, alpha=0.7, beta=1.1)
    for td in collector:
        print(f"collected tensordict has shape [Batch x Time]={td.shape}")
        rb.extend(td.view(-1))  # we split each action
        # rb.extend(td.unbind(0))  # we split each trajectory -- WIP

        collector.update_policy_weights_()  # if you have updated the local
        # policy (on cpu) you may want to sync the collectors' policies to it
        print("rb sample: ", rb.sample(2))
