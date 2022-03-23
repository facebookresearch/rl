import argparse

import pytest
import torch
from _utils_internal import get_available_devices
from mocking_classes import (
    ContinuousActionConvMockEnvNumpy,
    ContinuousActionVecMockEnv,
    DiscreteActionVecMockEnv,
    DiscreteActionConvMockEnvNumpy,
)
from torchrl.agents.helpers import parser_env_args, transformed_env_constructor
from torchrl.agents.helpers.models import (
    make_dqn_actor,
    parser_model_args_discrete,
    parser_model_args_continuous,
    make_ddpg_actor,
    make_ppo_model,
    make_sac_model,
    make_redq_model,
)
from torchrl.envs.libs.gym import _has_gym


## these tests aren't truly unitary but setting up a fake env for the
# purpose of building a model with args is a lot of unstable scaffoldings
# with unclear benefits


def _assert_keys_match(td, expeceted_keys):
    td_keys = list(td.keys())
    d = set(td_keys) - set(expeceted_keys)
    assert len(d) == 0, f"{d} is in tensordict but unexpected"
    d = set(expeceted_keys) - set(td_keys)
    assert len(d) == 0, f"{d} is expecter but not in tensordict"
    assert len(td_keys) == len(expeceted_keys)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("noisy", [tuple(), ("--noisy",)])
@pytest.mark.parametrize("distributional", [tuple(), ("--distributional",)])
@pytest.mark.parametrize("from_pixels", [tuple(), ("--from_pixels",)])
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
        expected_keys += ["observation_pixels"]
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


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("from_pixels", [tuple(), ("--from_pixels",)])
def test_ddpg_maker(device, from_pixels):
    device = torch.device("cpu")
    flags = list(from_pixels)
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
    actor(td)
    expected_keys = ["done", "action"]
    if from_pixels:
        expected_keys += ["observation_pixels"]
    else:
        expected_keys += ["observation_vector"]

    try:
        _assert_keys_match(td, expected_keys)
    except AssertionError:
        proof_environment.close()
        raise

    value(td)
    expected_keys += ["state_action_value"]
    try:
        _assert_keys_match(td, expected_keys)
    except AssertionError:
        proof_environment.close()
        raise

    proof_environment.close()
    del proof_environment


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("from_pixels", [tuple()])
@pytest.mark.parametrize("shared_mapping", [tuple(), ("--shared_mapping",)])
def test_ppo_maker(device, from_pixels, shared_mapping):
    flags = list(from_pixels + shared_mapping)

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

    actor_value = make_ppo_model(
        proof_environment,
        device=device,
        args=args,
    )
    actor = actor_value.get_policy_operator()
    expected_keys = [
        "done",
        "observation_vector",
        "action_dist_param_0",
        "action_dist_param_1",
        "action",
        "action_log_prob",
    ]
    if shared_mapping:
        expected_keys += ["hidden"]
    td = proof_environment.reset().to(device)
    td_clone = td.clone()
    actor(td_clone)
    try:
        _assert_keys_match(td_clone, expected_keys)
    except AssertionError:
        proof_environment.close()
        raise

    value = actor_value.get_value_operator()
    expected_keys = ["done", "observation_vector", "state_value"]
    if shared_mapping:
        expected_keys += ["hidden"]
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
def test_sac_make(device, gsde, tanh_loc, from_pixels):
    flags = list(gsde + tanh_loc + from_pixels)
    parser = argparse.ArgumentParser()
    parser = parser_env_args(parser)
    parser = parser_model_args_continuous(parser, algorithm="SAC")
    args = parser.parse_args(flags)

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
    actor(td_clone)
    expected_keys = ["done", "observation_vector", "action"]
    if len(gsde):
        expected_keys += ["_eps_gSDE"]

    try:
        _assert_keys_match(td_clone, expected_keys)
    except AssertionError:
        proof_environment.close()
        raise

    qvalue(td_clone)
    expected_keys = ["done", "observation_vector", "action", "state_action_value"]
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
@pytest.mark.parametrize("from_pixels", [tuple()])
@pytest.mark.skipif(not _has_gym, reason="No gym library found")
def test_redq_make(device, from_pixels):
    flags = list(from_pixels)
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
    td = proof_environment.reset()
    actor(td)
    expected_keys = ["done", "observation_vector", "action", "action_log_prob"]
    try:
        _assert_keys_match(td, expected_keys)
    except AssertionError:
        proof_environment.close()
        raise

    qvalue(td)
    expected_keys = [
        "done",
        "observation_vector",
        "action",
        "action_log_prob",
        "state_action_value",
    ]
    try:
        _assert_keys_match(td, expected_keys)
    except AssertionError:
        proof_environment.close()
        raise
    proof_environment.close()
    del proof_environment


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
