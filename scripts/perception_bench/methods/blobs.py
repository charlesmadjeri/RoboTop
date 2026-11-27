from dataclasses import dataclass

import cv2
import numpy as np

from perception_bench.truth import Point

KERNEL = np.ones((3, 3), np.uint8)


@dataclass(frozen=True)
class Shape:
    min_area: float = 0.0008
    max_area: float = 1.0
    min_solidity: float = 0.0
    max_aspect: float = 100.0


def clean(mask: np.ndarray) -> np.ndarray:
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, KERNEL)


def best_blob(mask: np.ndarray, shape: Shape) -> Point | None:
    contours, _ = cv2.findContours(clean(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    accepted = [c for c in contours if _fits(c, shape, mask.size)]
    if not accepted:
        return None
    moments = cv2.moments(max(accepted, key=cv2.contourArea))
    return moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]


def _fits(contour: np.ndarray, shape: Shape, size: int) -> bool:
    area = cv2.contourArea(contour)
    if not shape.min_area * size <= area <= shape.max_area * size:
        return False
    hull = cv2.contourArea(cv2.convexHull(contour))
    (_, _), (width, height), _ = cv2.minAreaRect(contour)
    sides = sorted((width, height))
    aspect = sides[1] / sides[0] if sides[0] > 0 else shape.max_aspect + 1
    return hull > 0 and area / hull >= shape.min_solidity and aspect <= shape.max_aspect
