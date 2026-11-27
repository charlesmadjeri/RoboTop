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

import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from lerobot.cameras import Camera, ColorMode
from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
from lerobot.scripts.camera_discovery import find_and_print_cameras

logger = logging.getLogger(__name__)


def save_image(
    img_array: np.ndarray,
    camera_identifier: str | int,
    images_dir: Path,
    camera_type: str,
) -> None:
    """
    Saves a single RGB image to disk with OpenCV.
    """
    try:
        safe_identifier = str(camera_identifier).replace("/", "_").replace("\\", "_")
        filename_prefix = f"{camera_type.lower()}_{safe_identifier}"
        filename = f"{filename_prefix}.png"

        path = images_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
        logger.info(f"Saved image: {path}")
    except Exception as e:
        logger.error(f"Failed to save image for camera {camera_identifier} (type {camera_type}): {e}")


def create_camera_instance(cam_meta: dict[str, Any], *, warmup_s: int = 1) -> dict[str, Any] | None:
    """Create and connect to a camera instance based on metadata."""
    cam_type = cam_meta.get("type")
    cam_id = cam_meta.get("id")
    instance: Camera | None = None

    logger.info(f"Preparing {cam_type} ID {cam_id} with default profile")

    if cam_id is None:
        logger.warning(f"Camera metadata of type {cam_type} has no 'id'. Skipping.")
        return None

    try:
        if cam_type == "OpenCV":
            cv_config = OpenCVCameraConfig(
                index_or_path=cam_id,
                color_mode=ColorMode.RGB,
                warmup_s=warmup_s,
            )
            instance = OpenCVCamera(cv_config)
        else:
            logger.warning(f"Unknown camera type: {cam_type} for ID {cam_id}. Skipping.")
            return None

        logger.info(f"Connecting to {cam_type} camera: {cam_id}...")
        instance.connect(warmup=True)
        return {"instance": instance, "meta": cam_meta}
    except Exception as e:
        logger.error(f"Failed to connect or configure {cam_type} camera {cam_id}: {e}")
        if instance and instance.is_connected:
            instance.disconnect()
        return None


def process_camera_image(cam_dict: dict[str, Any], output_dir: Path, current_time: float) -> None:
    """Capture and process an image from a single camera."""
    cam = cam_dict["instance"]
    meta = cam_dict["meta"]
    cam_type_str = str(meta.get("type", "unknown"))
    cam_id_str = str(meta.get("id", "unknown"))

    try:
        image_data = cam.read()

        save_image(
            image_data,
            cam_id_str,
            output_dir,
            cam_type_str,
        )
    except TimeoutError:
        logger.warning(
            f"Timeout reading from {cam_type_str} camera {cam_id_str} at time {current_time:.2f}s."
        )
    except Exception as e:
        logger.error(f"Error reading from {cam_type_str} camera {cam_id_str}: {e}")
    return None


def cleanup_camera(cam_dict: dict[str, Any]) -> None:
    """Disconnect all cameras."""
    logger.info(f"Disconnecting camera with ID {cam_dict['meta'].get('id')}...")
    try:
        if cam_dict["instance"] and cam_dict["instance"].is_connected:
            cam_dict["instance"].disconnect()
    except Exception as e:
        logger.error(f"Error disconnecting camera {cam_dict['meta'].get('id')}: {e}")


def save_images_from_all_cameras(
    output_dir: Path,
    record_time_s: float = 2.0,
    camera_type: str | None = None,
    warmup_s: int = 1,
):
    """
    Connects to detected cameras (optionally filtered by type) and saves images from each.
    Uses default stream profiles for width, height, and FPS.

    Args:
        output_dir: Directory to save images.
        record_time_s: Duration in seconds to record images.
        camera_type: Optional string to filter cameras ("opencv").
                            If None, uses all detected cameras.
        warmup_s: Duration in seconds to warmup camera before recording images.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving images to {output_dir}")
    all_camera_metadata = find_and_print_cameras(camera_type_filter=camera_type)

    if not all_camera_metadata:
        logger.warning("No cameras detected matching the criteria. Cannot save images.")
        return

    logger.info(
        f"Starting image capture for {record_time_s} seconds from {len(all_camera_metadata)} cameras."
    )

    try:
        for cam_meta in all_camera_metadata:
            cam_dict = create_camera_instance(cam_meta, warmup_s=warmup_s)
            if cam_dict is None:
                continue
            start_time = time.perf_counter()
            while time.perf_counter() - start_time < record_time_s:
                current_capture_time = time.perf_counter()
                process_camera_image(cam_dict, output_dir, current_capture_time)
            cleanup_camera(cam_dict)
    except KeyboardInterrupt:
        logger.info("Capture interrupted by user.")
    finally:
        print(f"Image capture finished. Images saved to {output_dir}")
