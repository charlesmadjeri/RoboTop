import math

import cv2
import numpy as np

MIN_PLACES = 4
MIN_SPREAD = 0.03


def intrinsics_from_horizontal_fov(width: int, height: int, fov_deg: float) -> np.ndarray:
    focal = (width / 2) / math.tan(math.radians(fov_deg) / 2)
    return np.array([[focal, 0.0, width / 2], [0.0, focal, height / 2], [0.0, 0.0, 1.0]])


def table_plane_pose(height: float) -> np.ndarray:
    pose = np.eye(4)
    pose[2, 3] = height
    return pose


def check_places(places: np.ndarray) -> None:
    if len(places) < MIN_PLACES:
        raise ValueError(f"at least {MIN_PLACES} cube places are needed, got {len(places)}")
    singular_values = np.linalg.svd(places - places.mean(axis=0), compute_uv=False)
    if singular_values[1] / math.sqrt(len(places)) < MIN_SPREAD:
        raise ValueError("the cube places lie on a line, spread them over the table")


def solve_camera_pose(
    pixels: np.ndarray, places: np.ndarray, height: float, K: np.ndarray
) -> tuple[np.ndarray, float]:
    pixels = np.asarray(pixels, dtype=np.float64)
    places = np.asarray(places, dtype=np.float64)
    check_places(places)
    points = np.column_stack([places, np.zeros(len(places))])
    found, rotation_vector, translation = cv2.solvePnP(points, pixels, K, None, flags=cv2.SOLVEPNP_IPPE)
    if not found:
        raise ValueError("the camera pose could not be solved")
    rotation_vector, translation = cv2.solvePnPRefineLM(points, pixels, K, None, rotation_vector, translation)
    projected, _ = cv2.projectPoints(points, rotation_vector, translation, K, None)
    error = float(np.linalg.norm(projected.reshape(-1, 2) - pixels, axis=1).mean())
    camera_from_plane = np.eye(4)
    camera_from_plane[:3, :3] = cv2.Rodrigues(rotation_vector)[0]
    camera_from_plane[:3, 3] = translation.ravel()
    robot_from_camera = table_plane_pose(height) @ np.linalg.inv(camera_from_plane)
    if robot_from_camera[2, 3] <= height:
        raise ValueError("the camera came out below the cube plane, check the cube places and their axes")
    return robot_from_camera, error
