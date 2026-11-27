from collections.abc import Callable

import numpy as np

from perception_bench.render import capture
from perception_bench.scene import Scene
from perception_bench.truth import Frame

RADIUS = (0.18, 0.30)
ANGLE_DEG = (-60.0, 50.0)
TRIES_PER_FRAME = 60


def random_cube(scene: Scene, rng: np.random.Generator) -> None:
    radius = rng.uniform(*RADIUS)
    angle = np.radians(rng.uniform(*ANGLE_DEG))
    scene.set_cube(radius * np.cos(angle), radius * np.sin(angle), rng.uniform(0, np.pi / 2))


def random_arm(scene: Scene, rng: np.random.Generator) -> None:
    low, high = scene.arm_ranges().T
    scene.set_arm(rng.uniform(low, high))


def sample(
    scene: Scene,
    rng: np.random.Generator,
    count: int,
    setup: Callable[[Scene, np.random.Generator], None],
    accept: Callable[[Frame], bool],
    tries_per_frame: int = TRIES_PER_FRAME,
) -> list[Frame]:
    frames: list[Frame] = []
    for _ in range(count * tries_per_frame):
        scene.reset()
        setup(scene, rng)
        scene.forward()
        frame = capture(scene)
        if accept(frame):
            frames.append(frame)
            if len(frames) == count:
                break
    return frames
