import cv2
import numpy as np

from perception_bench.methods.blobs import Shape, best_blob

CUBE_AB = (190.0, 150.0)
BOX_AB = (143.0, 150.0)
CUBE_RADII = (14.0, 22.0, 32.0)
BOX_RADII = (6.0, 9.0, 13.0)
MIN_LIGHTNESS = 25
CUBE_SHAPE = Shape(max_area=0.03, min_solidity=0.8, max_aspect=2.0)
BOX_SHAPE = Shape(max_area=0.10, min_solidity=0.7)


def make(empty: np.ndarray):
    return detect


def detect(rgb: np.ndarray) -> dict:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    return {
        "cube": _search(lab, CUBE_AB, CUBE_RADII, CUBE_SHAPE),
        "box": _search(lab, BOX_AB, BOX_RADII, BOX_SHAPE),
    }


def _search(lab: np.ndarray, reference: tuple[float, float], radii: tuple, shape: Shape):
    distance = np.hypot(lab[..., 1] - reference[0], lab[..., 2] - reference[1])
    bright = lab[..., 0] >= MIN_LIGHTNESS
    for radius in radii:
        found = best_blob(((distance <= radius) & bright).astype(np.uint8) * 255, shape)
        if found is not None:
            return found
    return None
