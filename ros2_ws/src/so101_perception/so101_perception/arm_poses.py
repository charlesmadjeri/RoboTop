import json
from pathlib import Path

import numpy as np

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
JOINT_LIMITS = np.array(
    [[-1.91986, 1.91986], [-1.74533, 1.74533], [-1.69, 1.69], [-1.65806, 1.65806], [-2.74385, 2.84121]]
)
LIMIT_MARGIN = 0.1

PRESET_POSES = np.array(
    [
        [0.41, 0.24, -0.11, 0.58, -0.35],
        [-0.23, 0.51, -0.12, 0.21, -0.71],
        [-0.8, -0.25, 0.4, 0.99, -0.86],
        [-0.49, -0.28, 0.63, -0.51, -0.84],
        [0.34, -0.26, 0.75, -0.8, -0.83],
        [0.8, 0.45, 0.48, -0.88, -0.64],
        [0.78, 0.1, 0.5, -0.95, 0.66],
        [-0.12, -0.15, 0.62, -0.71, 0.83],
        [-0.79, 0.49, -0.34, -0.3, 0.79],
        [-0.58, 0.43, -0.4, 0.82, 0.88],
        [0.07, -0.25, 0.47, 0.88, 0.82],
        [0.79, -0.18, 0.26, 0.95, 0.82],
    ]
)


def load_poses(path: Path) -> np.ndarray:
    poses = np.array(json.loads(path.read_text()), dtype=float)
    if poses.ndim != 2 or poses.shape[1] != len(ARM_JOINTS):
        raise ValueError(f"{path} must hold a list of poses of {len(ARM_JOINTS)} joint angles, in radians")
    return poses


def check_within_limits(poses: np.ndarray) -> None:
    low = JOINT_LIMITS[:, 0] + LIMIT_MARGIN
    high = JOINT_LIMITS[:, 1] - LIMIT_MARGIN
    for index, pose in enumerate(poses):
        outside = np.flatnonzero(~((pose >= low) & (pose <= high)))
        if outside.size:
            raise ValueError(f"pose {index + 1} puts {ARM_JOINTS[outside[0]]} too close to its limit")


def interpolate(start: np.ndarray, end: np.ndarray, step: float) -> np.ndarray:
    count = max(1, int(np.ceil(np.abs(end - start).max() / step)))
    return start + (end - start) * (np.arange(1, count + 1) / count)[:, None]
