from dataclasses import dataclass

import numpy as np

from so101_perception.detection import Pixel, find_cube, find_drop_box
from so101_perception.normalization import normalize, shrink_to_width


@dataclass(frozen=True)
class Detection:
    cube: Pixel | None
    box: Pixel | None


def scaled(pixel: Pixel | None, scale: float) -> Pixel | None:
    return None if pixel is None else (pixel[0] * scale, pixel[1] * scale)


def detect(rgb: np.ndarray) -> Detection:
    small, scale = shrink_to_width(rgb)
    cube, box = find_cube(small), find_drop_box(small)
    if cube is None or box is None:
        balanced = normalize(small)
        cube = find_cube(balanced) if cube is None else cube
        box = find_drop_box(balanced) if box is None else box
    return Detection(scaled(cube, scale), scaled(box, scale))


def detect_cube(rgb: np.ndarray) -> Pixel | None:
    small, scale = shrink_to_width(rgb)
    cube = find_cube(small)
    return scaled(find_cube(normalize(small)) if cube is None else cube, scale)
