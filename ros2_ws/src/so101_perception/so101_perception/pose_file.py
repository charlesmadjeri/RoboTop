import json
from pathlib import Path

import numpy as np

KEY = "robot_from_camera"


def save_pose(path: Path, robot_from_camera: np.ndarray) -> None:
    Path(path).write_text(json.dumps({KEY: np.asarray(robot_from_camera).tolist()}, indent=2))


def load_pose(path: Path) -> np.ndarray:
    pose = np.array(json.loads(Path(path).read_text())[KEY], dtype=float)
    if pose.shape != (4, 4):
        raise ValueError(f"{path}: {KEY} must be a 4x4 matrix, got shape {pose.shape}")
    return pose
