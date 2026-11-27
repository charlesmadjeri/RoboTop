import numpy as np

from so101_perception.arm_calibration import ArmCalibration, cube_positions, refine, robust_parameters, solve_from_arm
from so101_perception.projection import pixel_to_ray


def distance_to_pixel_ray(
    point: np.ndarray, pixel: np.ndarray, K: np.ndarray, robot_from_camera: np.ndarray
) -> float:
    direction = robot_from_camera[:3, :3] @ pixel_to_ray(pixel[0], pixel[1], K)
    offset = point - robot_from_camera[:3, 3]
    return float(np.linalg.norm(offset - (offset @ direction) * direction))


def held_out_error(
    calibration: ArmCalibration, pixel: np.ndarray, gripper_pose: np.ndarray, K: np.ndarray
) -> float:
    point = cube_positions(gripper_pose[None], calibration.grasp_offset)[0]
    return distance_to_pixel_ray(point, pixel, K, calibration.robot_from_camera)


def leave_one_out_errors(pixels: np.ndarray, gripper_poses: np.ndarray, K: np.ndarray) -> np.ndarray:
    pixels = np.asarray(pixels, dtype=np.float64)
    gripper_poses = np.asarray(gripper_poses, dtype=np.float64)
    kept = np.flatnonzero(solve_from_arm(pixels, gripper_poses, K).inliers)
    errors = []
    for left_out in kept:
        rest = kept[kept != left_out]
        start = robust_parameters(pixels[rest], gripper_poses[rest], K, True)
        calibration = refine(pixels[rest], gripper_poses[rest], K, start, np.ones(len(rest), dtype=bool))
        errors.append(held_out_error(calibration, pixels[left_out], gripper_poses[left_out], K))
    return np.array(errors)
