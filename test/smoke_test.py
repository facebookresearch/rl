def test_imports():
    from torchrl.data import (
        ReplayBuffer,
        PrioritizedReplayBuffer,
        TensorDict,
        TensorSpec,
    )
    from torchrl.envs import GymLikeEnv
    from torchrl.envs import TransformedEnv, Transform
    from torchrl.modules import TensorDictModule
    from torchrl.objectives.common import _LossModule
