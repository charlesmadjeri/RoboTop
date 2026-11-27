from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np

from so101_perception.levenberg_marquardt import levenberg_marquardt

MIN_POSES = 8
MIN_HEIGHT_SPREAD = 0.01
MIN_ORIENTATION_SPREAD = 0.1
ROBUST_SCALE_PX = 4.0
INLIER_FACTOR = 3.0
MIN_INLIER_PX = 3.0
MAX_MEDIAN_RESIDUAL_PX = 30.0


@dataclass(frozen=True)
class ArmCalibration:
    robot_from_camera: np.ndarray
    grasp_offset: np.ndarray
    residuals_px: np.ndarray
    inliers: np.ndarray


def cube_positions(gripper_poses: np.ndarray, grasp_offset: np.ndarray) -> np.ndarray:
    return gripper_poses[:, :3, 3] + np.einsum("nij,j->ni", gripper_poses[:, :3, :3], grasp_offset)


def grasp_offset_of(parameters: np.ndarray) -> np.ndarray:
    return parameters[6:9] if len(parameters) > 6 else np.zeros(3)


def camera_from_robot_of(parameters: np.ndarray) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = cv2.Rodrigues(parameters[:3])[0]
    matrix[:3, 3] = parameters[3:6]
    return matrix


def project_cube(parameters: np.ndarray, gripper_poses: np.ndarray, K: np.ndarray) -> np.ndarray:
    points = cube_positions(gripper_poses, grasp_offset_of(parameters))
    projected, _ = cv2.projectPoints(points, parameters[:3], parameters[3:6], K, None)
    return projected.reshape(-1, 2)


def reprojection_residuals(
    pixels: np.ndarray, gripper_poses: np.ndarray, K: np.ndarray
) -> Callable[[np.ndarray], np.ndarray]:
    return lambda parameters: (project_cube(parameters, gripper_poses, K) - pixels).ravel()


def check_poses(gripper_poses: np.ndarray, fit_offset: bool) -> None:
    count = len(gripper_poses)
    if count < MIN_POSES:
        raise ValueError(f"at least {MIN_POSES} arm poses are needed, got {count}")
    positions = gripper_poses[:, :3, 3]
    thinnest = np.linalg.svd(positions - positions.mean(axis=0), compute_uv=False)[2]
    if thinnest / np.sqrt(count) < MIN_HEIGHT_SPREAD:
        raise ValueError("the arm poses lie on a plane, lift the cube to different heights")
    if fit_offset:
        rotations = gripper_poses[:, :3, :3]
        weakest = np.linalg.svd((rotations - rotations.mean(axis=0)).reshape(-1, 3), compute_uv=False)[2]
        if weakest / np.sqrt(count) < MIN_ORIENTATION_SPREAD:
            raise ValueError("the gripper orientation barely changes, vary the wrist or fix the grasp offset")


def initial_parameters(pixels: np.ndarray, gripper_poses: np.ndarray, K: np.ndarray, fit_offset: bool) -> np.ndarray:
    found, rotation_vector, translation = cv2.solvePnP(
        np.ascontiguousarray(gripper_poses[:, :3, 3]), pixels, K, None, flags=cv2.SOLVEPNP_EPNP
    )
    if not found:
        raise ValueError("the camera pose could not be solved")
    pose = np.concatenate([rotation_vector.ravel(), translation.ravel()])
    return np.concatenate([pose, np.zeros(3)]) if fit_offset else pose


def pose_errors_px(parameters: np.ndarray, pixels: np.ndarray, gripper_poses: np.ndarray, K: np.ndarray) -> np.ndarray:
    return np.linalg.norm(project_cube(parameters, gripper_poses, K) - pixels, axis=1)


def solve_from_arm(
    pixels: np.ndarray, gripper_poses: np.ndarray, K: np.ndarray, fit_offset: bool = True
) -> ArmCalibration:
    pixels = np.asarray(pixels, dtype=np.float64)
    gripper_poses = np.asarray(gripper_poses, dtype=np.float64)
    check_poses(gripper_poses, fit_offset)
    start = initial_parameters(pixels, gripper_poses, K, fit_offset)
    robust = levenberg_marquardt(reprojection_residuals(pixels, gripper_poses, K), start, ROBUST_SCALE_PX)
    errors = pose_errors_px(robust, pixels, gripper_poses, K)
    if np.median(errors) > MAX_MEDIAN_RESIDUAL_PX:
        raise ValueError("the cube detections do not fit any camera pose, check the cube is held by the gripper")
    inliers = errors <= max(INLIER_FACTOR * float(np.median(errors)), MIN_INLIER_PX)
    if inliers.sum() < MIN_POSES:
        raise ValueError(f"only {int(inliers.sum())} of {len(inliers)} arm poses agree, check the cube detections")
    fitted = levenberg_marquardt(reprojection_residuals(pixels[inliers], gripper_poses[inliers], K), robust)
    camera_from_robot = camera_from_robot_of(fitted)
    grasp_offset = grasp_offset_of(fitted)
    depths = cube_positions(gripper_poses[inliers], grasp_offset) @ camera_from_robot[2, :3] + camera_from_robot[2, 3]
    if np.any(depths <= 0):
        raise ValueError("the camera pose came out behind the cube, check the detections")
    return ArmCalibration(
        robot_from_camera=np.linalg.inv(camera_from_robot),
        grasp_offset=grasp_offset,
        residuals_px=pose_errors_px(fitted, pixels, gripper_poses, K),
        inliers=inliers,
    )
