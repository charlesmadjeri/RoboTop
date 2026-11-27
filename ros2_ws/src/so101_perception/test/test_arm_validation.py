import numpy as np
import pytest
from arm_scene import cube_pixels, gripper_poses, noisy_poses
from cube_scene import K, ROBOT_FROM_CAMERA

from so101_perception.arm_validation import distance_to_pixel_ray, leave_one_out_errors


def test_exact_poses_predict_every_left_out_pose_within_a_millimeter():
    poses = gripper_poses()
    errors = leave_one_out_errors(cube_pixels(ROBOT_FROM_CAMERA, poses), poses, K)
    assert len(errors) == len(poses)
    assert errors.max() < 1e-3


def test_noisy_poses_predict_left_out_poses_within_a_centimeter():
    poses = gripper_poses()
    pixels = cube_pixels(ROBOT_FROM_CAMERA, poses) + np.random.default_rng(1).normal(0, 1.0, (len(poses), 2))
    errors = leave_one_out_errors(pixels, noisy_poses(poses, 0.003), K)
    assert 0 < np.median(errors) < 0.01


def test_a_wrong_detection_is_left_out_of_the_validation():
    poses = gripper_poses()
    pixels = cube_pixels(ROBOT_FROM_CAMERA, poses)
    pixels[4] += (150.0, 150.0)
    assert len(leave_one_out_errors(pixels, poses, K)) == len(poses) - 1


def test_nine_poses_are_the_minimum_for_the_validation():
    poses = gripper_poses(count=8)
    with pytest.raises(ValueError, match="at least 8 arm poses"):
        leave_one_out_errors(cube_pixels(ROBOT_FROM_CAMERA, poses), poses, K)


def test_a_point_on_the_pixel_ray_is_at_distance_zero_and_a_shifted_one_is_not():
    point = np.array([0.2, 0.1, 0.05])
    pixel = cube_pixels(ROBOT_FROM_CAMERA, np.array([np.block([[np.eye(3), point[:, None]], [0, 0, 0, 1]])]), np.zeros(3))[0]
    assert distance_to_pixel_ray(point, pixel, K, ROBOT_FROM_CAMERA) == pytest.approx(0, abs=1e-9)
    assert distance_to_pixel_ray(point + [0.0, 0.0, 0.02], pixel, K, ROBOT_FROM_CAMERA) > 0.005
