from collections.abc import Callable

import numpy as np

from perception_bench.methods import aruco, background_diff, hsv_fixed, hsv_normalized, lab_geometry, yolo, yolo_vulkan

Detector = Callable[[np.ndarray], dict]
Factory = Callable[[np.ndarray], Detector]

METHODS: dict[str, Factory] = {
    "hsv_fixed": hsv_fixed.make,
    "hsv_normalized": hsv_normalized.make,
    "lab_geometry": lab_geometry.make,
    "background_diff": background_diff.make,
    "aruco": aruco.make,
    "yolo": yolo.make,
    "yolo_vulkan": yolo_vulkan.make,
}
MARKER_METHODS = {"aruco"}
LEARNED_METHODS = {"yolo", "yolo_vulkan"}
DEFAULT_METHODS = [m for m in METHODS if m not in MARKER_METHODS | LEARNED_METHODS]
