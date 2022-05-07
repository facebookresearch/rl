# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

__all__ = [
    #"parser_recorder_args"
    ]



@dataclass
class RecorderConfig:
    record_video: bool = False
    # whether a video of the task should be rendered during logging.

    exp_name: str = ""
    # experiment name. Used for logging directory.
    # A date and uuid will be joined to account for multiple experiments with the same name.
    
    record_interval: int = 50
    # task (if any) for the environment

    record_frames: int = 1000
    # number of batch collections in between two collections of validation rollouts.
