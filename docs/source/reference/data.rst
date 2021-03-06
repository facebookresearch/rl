.. currentmodule:: torchrl.data

torchrl.data package
====================

Replay Buffers
--------------

Replay buffers are a central part of off-policy RL algorithms. TorchRL provides an efficient implementation of a few,
widely used replay buffers:


.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    ReplayBuffer
    PrioritizedReplayBuffer
    TensorDictReplayBuffer
    TensorDictPrioritizedReplayBuffer


TensorDict
----------

Passing data across objects can become a burdensome task when designing high-level classes: for instance it can be
hard to design an actor class that can take an arbitrary number of inputs and return an arbitrary number of inputs. The
`TensorDict` class simplifies this process by packing together a bag of tensors in a dictionary-like object. This
class supports a set of basic operations on tensors to facilitate the manipulation of entire batch of data (e.g.
`torch.cat`, `torch.stack`, `.to(device)` etc.).


.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    TensorDict
    SubTensorDict
    LazyStackedTensorDict

TensorSpec
----------

The `TensorSpec` parent class and subclasses define the basic properties of observations and actions in TorchRL, such
as shape, device, dtype and domain.


.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    TensorSpec
    BoundedTensorSpec
    OneHotDiscreteTensorSpec
    UnboundedContinuousTensorSpec
    NdBoundedTensorSpec
    NdUnboundedContinuousTensorSpec
    BinaryDiscreteTensorSpec
    MultOneHotDiscreteTensorSpec
    CompositeSpec

Transforms
----------

In most cases, the raw output of an environment must be treated before being passed to another object (such as a
policy or a value operator). To do this, TorchRL provides a set of transforms that aim at reproducing the transform
logic of `torch.distributions.Transform` and `torchvision.transforms`.


.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    Transform
    TransformedEnv
    Compose
    CatTensors
    CatFrames
    RewardClipping
    Resize
    GrayScale
    ToTensorImage
    ObservationNorm
    RewardScaling
    ObservationTransform
    FiniteTensorDictCheck
    DoubleToFloat
    NoopResetEnv
    BinerizeReward
    PinMemoryTransform
    VecNorm
