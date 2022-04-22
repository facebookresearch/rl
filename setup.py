# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import distutils.command.clean
import os
import shutil
import subprocess
from pathlib import Path

from setuptools import setup, find_packages

from build_tools import setup_helpers


def _get_pytorch_version():
    if "PYTORCH_VERSION" in os.environ:
        return f"torch=={os.environ['PYTORCH_VERSION']}"
    return "torch"


def _get_packages():
    exclude = [
        "build*",
        "test*",
        "torchrl.csrc*",
        "third_party*",
        "tools*",
    ]
    return find_packages(exclude=exclude)


ROOT_DIR = Path(__file__).parent.resolve()


class clean(distutils.command.clean.clean):
    def run(self):
        # Run default behavior first
        distutils.command.clean.clean.run(self)

        # Remove torchrl extension
        for path in (ROOT_DIR / "torchrl").glob("**/*.so"):
            print(f"removing '{path}'")
            path.unlink()
        # Remove build directory
        build_dirs = [
            ROOT_DIR / "build",
        ]
        for path in build_dirs:
            if path.exists():
                print(f"removing '{path}' (and everything under it)")
                shutil.rmtree(str(path), ignore_errors=True)


def _run_cmd(cmd):
    try:
        return subprocess.check_output(cmd, cwd=ROOT_DIR).decode("ascii").strip()
    except Exception:
        return None


def _main():
    pytorch_package_dep = _get_pytorch_version()
    print("-- PyTorch dependency:", pytorch_package_dep)
    # branch = _run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    # tag = _run_cmd(["git", "describe", "--tags", "--exact-match", "@"])

    setup(
        name="torchrl",
        version="0.1",
        author="torchrl contributors",
        author_email="vmoens@fb.com",
        packages=_get_packages(),
        ext_modules=setup_helpers.get_ext_modules(),
        cmdclass={
            "build_ext": setup_helpers.CMakeBuild,
            "clean": clean,
        },
        install_requires=[pytorch_package_dep, "numpy"],
        extras_require={
            "atari": ["gym", "atari-py", "ale-py", "gym[accept-rom-license]", "pygame"],
            "dm_control": ["dm_control"],
            "gym_continuous": ["mujoco-py", "mujoco"],
            "rendering": ["moviepy"],
            "tests": ["pytest"],
            "utils": ["tqdm", "configargparse"],
            "smac": ["smac @ git+https://github.com/oxwhirl/smac.git"],
        },
    )


if __name__ == "__main__":
    _main()
