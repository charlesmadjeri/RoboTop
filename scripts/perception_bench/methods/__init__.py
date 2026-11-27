from collections.abc import Callable

import numpy as np

from perception_bench.methods import background_diff, hsv_fixed, hsv_normalized, lab_geometry

Detector = Callable[[np.ndarray], dict]
Factory = Callable[[np.ndarray], Detector]

METHODS: dict[str, Factory] = {
    "hsv_fixed": hsv_fixed.make,
    "hsv_normalized": hsv_normalized.make,
    "lab_geometry": lab_geometry.make,
    "background_diff": background_diff.make,
}
