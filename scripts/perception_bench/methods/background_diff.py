import cv2
import numpy as np

from perception_bench.methods.blobs import Shape, best_blob
from so101_perception.detection import find_drop_box

BLUR = (5, 5)
MIN_CHANGE = 30
RED_OVER_OTHERS = 2.2
CUBE_SHAPE = Shape(max_area=0.03, min_solidity=0.7, max_aspect=2.5)


def make(empty: np.ndarray):
    references: dict[tuple, np.ndarray] = {}

    def reference_for(rgb: np.ndarray) -> np.ndarray:
        if rgb.shape not in references:
            size = (rgb.shape[1], rgb.shape[0])
            references[rgb.shape] = cv2.GaussianBlur(cv2.resize(empty, size), BLUR, 0).astype(np.float32)
        return references[rgb.shape]

    def detect(rgb: np.ndarray) -> dict:
        return {"cube": _find_cube(rgb, reference_for(rgb)), "box": find_drop_box(rgb)}

    return detect


def _find_cube(rgb: np.ndarray, reference: np.ndarray):
    current = cv2.GaussianBlur(rgb, BLUR, 0).astype(np.float32)
    changed = np.abs(current - reference).max(axis=2) > MIN_CHANGE
    red, green, blue = (rgb[..., i].astype(np.float32) for i in range(3))
    reddish = red > RED_OVER_OTHERS * np.maximum(np.maximum(green, blue), 1.0)
    return best_blob((changed & reddish).astype(np.uint8) * 255, CUBE_SHAPE)
