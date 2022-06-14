#!/usr/bin/env bash

set -e

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

export PYTORCH_TEST_WITH_SLOW='1'
python -m torch.utils.collect_env
# Avoid error: "fatal: unsafe repository"
git config --global --add safe.directory '*'

root_dir="$(git rev-parse --show-toplevel)"
env_dir="${root_dir}/env"
lib_dir="${env_dir}/lib"

# solves ImportError: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.21' not found
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$lib_dir
if [[ $OSTYPE == 'darwin'* ]]; then
  PRIVATE_MUJOCO_GL=glfw
else
  PRIVATE_MUJOCO_GL=egl
fi
# pygame.error: No available video device
export SDL_VIDEODRIVER=dummy
# solves AttributeError: 'NoneType' object has no attribute 'eglQueryString'
export MUJOCO_GL=$PRIVATE_MUJOCO_GL

pytest test/smoke_test.py -v --durations 20
pytest test/smoke_test_deps.py -v --durations 20
pytest -v --durations 20
