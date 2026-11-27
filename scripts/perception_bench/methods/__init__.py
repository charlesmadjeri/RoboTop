from collections.abc import Callable

import numpy as np

from perception_bench.methods import hsv_fixed

Detector = Callable[[np.ndarray], dict]
Factory = Callable[[np.ndarray], Detector]

METHODS: dict[str, Factory] = {
    "hsv_fixed": hsv_fixed.make,
}
