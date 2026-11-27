from dataclasses import dataclass

import numpy as np

from so101_perception.detection import BOX_HSV_RANGES, Pixel, find_cube, find_drop_box
from so101_perception.normalization import normalize, shrink_to_width

BOX_MIN_SATURATION = 150
NORMALIZED_BOX_RANGES = tuple(
    ((hue, BOX_MIN_SATURATION, value), high) for (hue, _, value), high in BOX_HSV_RANGES
)


@dataclass(frozen=True)
class Detection:
    cube: Pixel | None
    box: Pixel | None


def scaled(pixel: Pixel | None, scale: float) -> Pixel | None:
    return None if pixel is None else (pixel[0] * scale, pixel[1] * scale)


def detect(rgb: np.ndarray) -> Detection:
    small, scale = shrink_to_width(rgb)
    balanced = normalize(small)
    cube = find_cube(balanced)
    box = find_drop_box(balanced, NORMALIZED_BOX_RANGES)
    return Detection(scaled(cube, scale), scaled(box, scale))


def detect_cube(rgb: np.ndarray) -> Pixel | None:
    small, scale = shrink_to_width(rgb)
    return scaled(find_cube(normalize(small)), scale)
