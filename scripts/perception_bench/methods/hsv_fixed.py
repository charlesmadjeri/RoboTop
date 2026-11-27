import numpy as np

from so101_perception.detection import find_cube, find_drop_box


def make(empty: np.ndarray):
    return detect


def detect(rgb: np.ndarray) -> dict:
    return {"cube": find_cube(rgb), "box": find_drop_box(rgb)}
