import cv2
import numpy as np
from cube_scene import K

GRASP_OFFSET = np.array([0.0, 0.01, 0.025])


def gripper_poses(count: int = 12, seed: int = 3, tilt: float = 0.4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    poses = np.tile(np.eye(4), (count, 1, 1))
    for pose in poses:
        pose[:3, :3] = cv2.Rodrigues(rng.normal(0, tilt, 3))[0]
        pose[:3, 3] = [rng.uniform(0.10, 0.30), rng.uniform(0.05, 0.25), rng.uniform(0.03, 0.20)]
    return poses


def cube_pixels(robot_from_camera: np.ndarray, poses: np.ndarray, offset: np.ndarray = GRASP_OFFSET) -> np.ndarray:
    points = np.array([(pose @ np.append(offset, 1.0))[:3] for pose in poses])
    in_camera = (np.linalg.inv(robot_from_camera) @ np.column_stack([points, np.ones(len(points))]).T)[:3]
    pixels = K @ in_camera
    return (pixels[:2] / pixels[2]).T


def noisy_poses(poses: np.ndarray, sigma: float, seed: int = 5) -> np.ndarray:
    shifted = poses.copy()
    shifted[:, :3, 3] += np.random.default_rng(seed).normal(0, sigma, (len(poses), 3))
    return shifted
