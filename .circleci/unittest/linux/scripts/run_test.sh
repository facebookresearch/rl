#!/usr/bin/env bash

set -e

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

export PYTORCH_TEST_WITH_SLOW='1'
python -m torch.utils.collect_env
export MJLIB_PATH=$root_dir/.mujoco/mujoco-2.1.1/bin/libmujoco210.so
MUJOCO_GL=glfw pytest  --cov=torchrl --junitxml=test-results/junit.xml -v --durations 20 --ignore third_party test
