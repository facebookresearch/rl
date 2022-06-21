# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import pytest
import torch
from _utils_internal import get_available_devices, generate_seeds
from mocking_classes import (
    ContinuousActionConvMockEnvNumpy,
    ContinuousActionVecMockEnv,
    DiscreteActionVecMockEnv,
    DiscreteActionConvMockEnvNumpy,
)
from torchrl.envs.libs.gym import _has_gym
from torchrl.envs.utils import set_exploration_mode
from torchrl.trainers.helpers import parser_env_args, transformed_env_constructor
from torchrl.trainers.helpers.models import (
    make_dqn_actor,
    parser_model_args_discrete,
    parser_model_args_continuous,
    make_ddpg_actor,
    make_ppo_model,
    make_sac_model,
    make_redq_model,
)

## these tests aren't truly unitary but setting up a fake env for the
# purpose of building a model with args is a lot of unstable scaffoldings
# with unclear benefits


def _assert_keys_match(td, expeceted_keys):
    td_keys = list(td.keys())
    d = set(td_keys) - set(expeceted_keys)
    assert len(d) == 0, f"{d} is in tensordict but unexpected"
    d = set(expeceted_keys) - set(td_keys)
    assert len(d) == 0, f"{d} is expected but not in tensordict"
    assert len(td_keys) == len(expeceted_keys)


@pytest.mark.skipif(not _has_gym, reason="No gym library found")
@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("noisy", [tuple(), ("--noisy",)])
@pytest.mark.parametrize("distributional", [tuple(), ("--distributional",)])
@pytest.mark.parametrize(
    "from_pixels", [tuple(), ("--from_pixels", "--catframes", "4")]
)
def test_dqn_maker(device, noisy, distributional, from_pixels):
    flags = list(noisy + distributional + from_pixels) + ["--env_name=CartPole-v1"]
    parser = argparse.ArgumentParser()
    parser = parser_env_args(parser)
    parser = parser_model_args_discrete(parser)
    args = parser.parse_args(flags)

    env_maker = (
        DiscreteActionConvMockEnvNumpy if from_pixels else DiscreteActionVecMockEnv
    )
    env_maker = transformed_env_constructor(
        args, use_env_creator=False, custom_env_maker=env_maker
    )
    proof_environment = env_maker()

    actor = make_dqn_actor(proof_environment, args, device)
    td = proof_environment.reset().to(device)
    actor(td)

    expected_keys = ["done", "action", "action_value"]
    if from_pixels:
        expected_keys += ["pixels"]
    else:
        expected_keys += ["observation_vector"]

    if not distributional:
        expected_keys += ["chosen_action_value"]
    try:
        _assert_keys_match(td, expected_keys)
    except AssertionError:
        proof_environment.close()
        raise
    proof_environment.close()


@pytest.mark.skipif(not _has_gym, reason="No gym library found")
@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize(
    "from_pixels", [tuple(), ("--from_pixels", "--catframes", "4")]
)
@pytest.mark.parametrize("gsde", [tuple(), ("--gSDE",)])
@pytest.mark.parametrize("exploration", ["random", "mode"])
def test_ddpg_maker(device, from_pixels, gsde, exploration):
    if not gsde and exploration != "random":
        pytest.skip("no need to test this setting")
    device = torch.device("cpu")
    flags = list(from_pixels + gsde)
    parser = argparse.ArgumentParser()
    parser = parser_env_args(parser)
    parser = parser_model_args_continuous(parser, algorithm="DDPG")
    args = parser.parse_args(flags)

    env_maker = (
        ContinuousActionConvMockEnvNumpy if from_pixels else ContinuousActionVecMockEnv
    )
    env_maker = transformed_env_constructor(
        args, use_env_creator=False, custom_env_maker=env_maker
    )
    proof_environment = env_maker()
    actor, value = make_ddpg_actor(proof_environment, device=device, args=args)
    td = proof_environment.reset().to(device)
    with set_exploration_mode(exploration):
        actor(td)
    expected_keys = ["done", "action", "param"]
    if from_pixels:
        expected_keys += ["pixels", "hidden"]
    else:
        expected_keys += ["observation_vector"]

    if args.gSDE:
        expected_keys += ["scale", "loc", "_eps_gSDE"]

    try:
        _assert_keys_match(td, expected_keys)
    except AssertionError:
        proof_environment.close()
        raise

    if args.gSDE:
        tsf_loc = actor.module[-1].module.transform(td.get("loc"))
        if exploration == "random":
            with pytest.raises(AssertionError):
                torch.testing.assert_allclose(td.get("action"), tsf_loc)
        else:
            torch.testing.assert_allclose(td.get("action"), tsf_loc)

    value(td)
    expected_keys += ["state_action_value"]
    try:
        _assert_keys_match(td, expected_keys)
    except AssertionError:
        proof_environment.close()
        raise

    proof_environment.close()
    del proof_environment


@pytest.mark.skipif(not _has_gym, reason="No gym library found")
@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize(
    "from_pixels", [tuple(), ("--from_pixels", "--catframes", "4")]
)
@pytest.mark.parametrize("gsde", [tuple(), ("--gSDE",)])
@pytest.mark.parametrize("shared_mapping", [tuple(), ("--shared_mapping",)])
@pytest.mark.parametrize("exploration", ["random", "mode"])
def test_ppo_maker(device, from_pixels, shared_mapping, gsde, exploration):
    if not gsde and exploration != "random":
        pytest.skip("no need to test this setting")
    flags = list(from_pixels + shared_mapping + gsde)
    # if gsde and from_pixels:
    #     pytest.skip("gsde and from_pixels are incompatible")
    parser = argparse.ArgumentParser()
    parser = parser_env_args(parser)
    parser = parser_model_args_continuous(parser, algorithm="PPO")
    args = parser.parse_args(flags)

    env_maker = (
        ContinuousActionConvMockEnvNumpy if from_pixels else ContinuousActionVecMockEnv
    )
    env_maker = transformed_env_constructor(
        args, use_env_creator=False, custom_env_maker=env_maker
    )
    proof_environment = env_maker()

    if args.from_pixels and not args.shared_mapping:
        with pytest.raises(
            RuntimeError,
            match="PPO learnt from pixels require the shared_mapping to be set to True",
        ):
            actor_value = make_ppo_model(
                proof_environment,
                device=device,
                args=args,
            )
        return

    actor_value = make_ppo_model(
        proof_environment,
        device=device,
        args=args,
    )
    actor = actor_value.get_policy_operator()
    expected_keys = [
        "done",
        "pixels" if len(from_pixels) else "observation_vector",
        "action",
        "sample_log_prob",
        "loc",
        "scale",
    ]
    if shared_mapping:
        expected_keys += ["hidden"]
    if len(gsde):
        expected_keys += ["_eps_gSDE"]

    td = proof_environment.reset().to(device)
    td_clone = td.clone()
    with set_exploration_mode(exploration):
        actor(td_clone)
    try:
        _assert_keys_match(td_clone, expected_keys)
    except AssertionError:
        proof_environment.close()
        raise

    if args.gSDE:
        if args.shared_mapping:
            tsf_loc = actor[-1].module[-1].module.transform(td_clone.get("loc"))
        else:
            tsf_loc = actor.module[-1].module.transform(td_clone.get("loc"))

        if exploration == "random":
            with pytest.raises(AssertionError):
                torch.testing.assert_allclose(td_clone.get("action"), tsf_loc)
        else:
            torch.testing.assert_allclose(td_clone.get("action"), tsf_loc)

    value = actor_value.get_value_operator()
    expected_keys = [
        "done",
        "pixels" if len(from_pixels) else "observation_vector",
        "state_value",
    ]
    if shared_mapping:
        expected_keys += ["hidden"]
    if len(gsde):
        expected_keys += ["_eps_gSDE"]

    td_clone = td.clone()
    value(td_clone)
    try:
        _assert_keys_match(td_clone, expected_keys)
    except AssertionError:
        proof_environment.close()
        raise
    proof_environment.close()
    del proof_environment


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("gsde", [tuple(), ("--gSDE",)])
@pytest.mark.parametrize("from_pixels", [tuple()])
@pytest.mark.parametrize("tanh_loc", [tuple(), ("--tanh_loc",)])
@pytest.mark.skipif(not _has_gym, reason="No gym library found")
@pytest.mark.parametrize("exploration", ["random", "mode"])
def test_sac_make(device, gsde, tanh_loc, from_pixels, exploration):
    if not gsde and exploration != "random":
        pytest.skip("no need to test this setting")
    flags = list(gsde + tanh_loc + from_pixels)
    if gsde and from_pixels:
        pytest.skip("gsde and from_pixels are incompatible")

    parser = argparse.ArgumentParser()
    parser = parser_env_args(parser)
    parser = parser_model_args_continuous(parser, algorithm="SAC")
    args = parser.parse_args(flags)
    if from_pixels:
        args.catframes = 4

    env_maker = (
        ContinuousActionConvMockEnvNumpy if from_pixels else ContinuousActionVecMockEnv
    )
    env_maker = transformed_env_constructor(
        args, use_env_creator=False, custom_env_maker=env_maker
    )
    proof_environment = env_maker()

    model = make_sac_model(
        proof_environment,
        device=device,
        args=args,
    )

    actor, qvalue, value = model
    td = proof_environment.reset().to(device)
    td_clone = td.clone()
    with set_exploration_mode(exploration):
        actor(td_clone)
    expected_keys = [
        "done",
        "pixels" if len(from_pixels) else "observation_vector",
        "action",
        "loc",
        "scale",
    ]
    if len(gsde):
        expected_keys += ["_eps_gSDE"]

    if args.gSDE:
        tsf_loc = actor.module[-1].module.transform(td_clone.get("loc"))
        if exploration == "random":
            with pytest.raises(AssertionError):
                torch.testing.assert_allclose(td_clone.get("action"), tsf_loc)
        else:
            torch.testing.assert_allclose(td_clone.get("action"), tsf_loc)

    try:
        _assert_keys_match(td_clone, expected_keys)
    except AssertionError:
        proof_environment.close()
        raise

    qvalue(td_clone)
    expected_keys = [
        "done",
        "observation_vector",
        "action",
        "state_action_value",
        "loc",
        "scale",
    ]
    if len(gsde):
        expected_keys += ["_eps_gSDE"]

    try:
        _assert_keys_match(td_clone, expected_keys)
    except AssertionError:
        proof_environment.close()
        raise

    value(td)
    expected_keys = ["done", "observation_vector", "state_value"]
    if len(gsde):
        expected_keys += ["_eps_gSDE"]

    try:
        _assert_keys_match(td, expected_keys)
    except AssertionError:
        proof_environment.close()
        raise
    proof_environment.close()
    del proof_environment


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize(
    "from_pixels", [tuple(), ("--from_pixels", "--catframes", "4")]
)
@pytest.mark.parametrize("gsde", [tuple(), ("--gSDE",)])
@pytest.mark.skipif(not _has_gym, reason="No gym library found")
@pytest.mark.parametrize("exploration", ["random", "mode"])
def test_redq_make(device, from_pixels, gsde, exploration):
    if not gsde and exploration != "random":
        pytest.skip("no need to test this setting")
    flags = list(from_pixels + gsde)
    if gsde and from_pixels:
        pytest.skip("gsde and from_pixels are incompatible")

    parser = argparse.ArgumentParser()
    parser = parser_env_args(parser)
    parser = parser_model_args_continuous(parser, algorithm="REDQ")
    args = parser.parse_args(flags)

    env_maker = (
        ContinuousActionConvMockEnvNumpy if from_pixels else ContinuousActionVecMockEnv
    )
    env_maker = transformed_env_constructor(
        args, use_env_creator=False, custom_env_maker=env_maker
    )
    proof_environment = env_maker()

    model = make_redq_model(
        proof_environment,
        device=device,
        args=args,
    )
    actor, qvalue = model
    td = proof_environment.reset().to(device)
    with set_exploration_mode(exploration):
        actor(td)
    expected_keys = [
        "done",
        "action",
        "sample_log_prob",
        "loc",
        "scale",
    ]
    if len(gsde):
        expected_keys += ["_eps_gSDE"]
    if from_pixels:
        expected_keys += ["hidden", "pixels"]
    else:
        expected_keys += ["observation_vector"]

    try:
        _assert_keys_match(td, expected_keys)
    except AssertionError:
        proof_environment.close()
        raise

    if args.gSDE:
        tsf_loc = actor.module[-1].module.transform(td.get("loc"))
        if exploration == "random":
            with pytest.raises(AssertionError):
                torch.testing.assert_allclose(td.get("action"), tsf_loc)
        else:
            torch.testing.assert_allclose(td.get("action"), tsf_loc)

    qvalue(td)
    expected_keys = [
        "done",
        "action",
        "sample_log_prob",
        "state_action_value",
        "loc",
        "scale",
    ]
    if len(gsde):
        expected_keys += ["_eps_gSDE"]
    if from_pixels:
        expected_keys += ["hidden", "pixels"]
    else:
        expected_keys += ["observation_vector"]
    try:
        _assert_keys_match(td, expected_keys)
    except AssertionError:
        proof_environment.close()
        raise
    proof_environment.close()
    del proof_environment


@pytest.mark.parametrize("initial_seed", range(5))
def test_seed_generator(initial_seed):
    num_seeds = 100
    prev_seeds = []

    # Check unique seed generation
    if initial_seed == 0:
        with pytest.raises(ValueError) as e:
            seeds0 = generate_seeds(initial_seed - 1, num_seeds)
        return
    else:
        seeds0 = generate_seeds(initial_seed - 1, num_seeds)
    seeds1 = generate_seeds(initial_seed, num_seeds)
    assert len(seeds1) == num_seeds
    assert len(seeds1) == len(set(seeds1))
    assert len(set(seeds0).intersection(set(seeds1))) == 0

    # Check deterministic seed generation
    seeds0 = generate_seeds(initial_seed, num_seeds)
    seeds1 = generate_seeds(initial_seed, num_seeds)
    assert seeds0 == seeds1


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
