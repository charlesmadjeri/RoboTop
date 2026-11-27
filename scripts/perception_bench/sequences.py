from dataclasses import dataclass

import numpy as np

from perception_bench.render import capture, render_rgb
from perception_bench.scene import Scene
from perception_bench.truth import Frame

START_X = 0.20
TURN_X = 0.62
LANE_Y = 0.0
SUBFRAMES = 4
YAW = 0.4


@dataclass(frozen=True)
class Motion:
    name: str
    metres_per_frame: float


MOTIONS = (Motion("motion_slow", 0.006), Motion("motion_fast", 0.02))


def path(speed: float) -> np.ndarray:
    out = np.arange(START_X, TURN_X, speed)
    return np.concatenate([out, out[::-1]])


def generate(motion: Motion, scene: Scene) -> list[Frame]:
    scene.reset()
    frames = []
    for x in path(motion.metres_per_frame):
        scene.set_cube(x, LANE_Y, YAW)
        scene.forward()
        truth = capture(scene).truth
        frames.append(Frame(_exposure(scene, x, motion.metres_per_frame), truth))
    return frames


def _exposure(scene: Scene, x: float, speed: float) -> np.ndarray:
    offsets = (np.arange(SUBFRAMES) + 0.5) / SUBFRAMES - 0.5
    shots = []
    for offset in offsets:
        scene.set_cube(x + offset * speed, LANE_Y, YAW)
        scene.forward()
        shots.append(render_rgb(scene).astype(np.float32))
    scene.set_cube(x, LANE_Y, YAW)
    return np.clip(np.rint(np.mean(shots, axis=0)), 0, 255).astype(np.uint8)
