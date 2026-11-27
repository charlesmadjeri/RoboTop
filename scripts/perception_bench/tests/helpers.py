import numpy as np

from perception_bench.truth import Frame, Target, Truth

BLANK = np.zeros((4, 4, 3), np.uint8)


def target(visible: int = 400, full: int = 400, centroid=(100.0, 50.0)) -> Target:
    return Target(visible, full, centroid if visible else None)


def frame(cube: Target, box: Target | None = None) -> Frame:
    return Frame(BLANK, Truth(cube, box or target(1800, 1800, (200.0, 160.0))))


def detection(cube=None, box=None) -> dict:
    return {"cube": cube, "box": box}
