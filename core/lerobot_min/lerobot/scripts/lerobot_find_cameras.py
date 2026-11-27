#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Helper to find the camera devices available in your system.

Example:

```shell
lerobot-find-cameras
```

Extracted from LeRobot v0.7.0: OpenCV cameras only (no RealSense), images saved with OpenCV.
"""

import argparse
from pathlib import Path

from lerobot.scripts.camera_capture import save_images_from_all_cameras
from lerobot.utils.utils import init_logging


def main():
    init_logging()

    parser = argparse.ArgumentParser(
        description="Unified camera utility script for listing cameras and capturing images."
    )
    parser.add_argument(
        "camera_type",
        type=str,
        nargs="?",
        default=None,
        choices=["opencv"],
        help="Camera type to capture from ('opencv'). Captures from all if omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="outputs/captured_images",
        help="Directory to save images. Default: outputs/captured_images",
    )
    parser.add_argument(
        "--record-time-s",
        type=float,
        default=2.0,
        help="Time duration to attempt capturing frames. Default: 2 seconds.",
    )
    parser.add_argument(
        "--warmup-s",
        type=int,
        default=1,
        help="Time duration to warmup camera before attempting to capture frames. Default: 1 second.",
    )
    args = parser.parse_args()
    save_images_from_all_cameras(**vars(args))


if __name__ == "__main__":
    main()
