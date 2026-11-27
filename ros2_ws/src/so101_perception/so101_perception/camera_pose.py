import math

import cv2
import numpy as np


def intrinsics_from_horizontal_fov(width: int, height: int, fov_deg: float) -> np.ndarray:
    focal = (width / 2) / math.tan(math.radians(fov_deg) / 2)
    return np.array([[focal, 0.0, width / 2], [0.0, focal, height / 2], [0.0, 0.0, 1.0]])


def marker_corners(size: float) -> np.ndarray:
    half = size / 2
    return np.array([[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]])


def robot_from_marker(position: tuple[float, float, float], yaw_deg: float) -> np.ndarray:
    yaw = math.radians(yaw_deg)
    pose = np.eye(4)
    pose[:2, :2] = [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]
    pose[:3, 3] = position
    return pose


def solve_camera_pose(
    image_corners: np.ndarray, size: float, marker_pose: np.ndarray, K: np.ndarray
) -> tuple[np.ndarray, float]:
    corners = marker_corners(size)
    found, rotation_vector, translation = cv2.solvePnP(
        corners, image_corners.astype(np.float64), K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE
    )
    if not found:
        raise ValueError("the marker pose could not be solved")
    projected, _ = cv2.projectPoints(corners, rotation_vector, translation, K, None)
    error = float(np.linalg.norm(projected.reshape(-1, 2) - image_corners, axis=1).mean())
    camera_from_marker = np.eye(4)
    camera_from_marker[:3, :3] = cv2.Rodrigues(rotation_vector)[0]
    camera_from_marker[:3, 3] = translation.ravel()
    return marker_pose @ np.linalg.inv(camera_from_marker), error
