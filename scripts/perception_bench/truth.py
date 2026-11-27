from dataclasses import dataclass
from math import sqrt

import numpy as np

Point = tuple[float, float]
Bounds = tuple[int, int, int, int]


@dataclass(frozen=True)
class Target:
    visible_px: int
    full_px: int
    centroid: Point | None
    bounds: Bounds | None = None

    @property
    def visible_fraction(self) -> float:
        return self.visible_px / self.full_px if self.full_px else 0.0

    @property
    def width(self) -> float:
        return sqrt(self.full_px)


@dataclass(frozen=True)
class Truth:
    cube: Target
    box: Target


@dataclass(frozen=True)
class Frame:
    rgb: np.ndarray
    truth: Truth
