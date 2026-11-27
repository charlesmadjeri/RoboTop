import numpy as np
import pytest
from arm_scene import GRASP_OFFSET, cube_pixels, gripper_poses, noisy_poses
from cube_scene import K, PLACES, ROBOT_FROM_CAMERA, pose_error, project_places

from so101_perception.arm_calibration import solve_from_arm
from so101_perception.projection import CUBE_CENTER_HEIGHT, project_to_plane


def test_exact_arm_poses_give_back_the_camera_pose_and_the_grasp_offset():
    poses = gripper_poses()
    result = solve_from_arm(cube_pixels(ROBOT_FROM_CAMERA, poses), poses, K)
    translation, angle = pose_error(result.robot_from_camera, ROBOT_FROM_CAMERA)
    assert translation < 1e-4 and angle < 0.01
    assert result.grasp_offset == pytest.approx(GRASP_OFFSET, abs=1e-4)
    assert result.inliers.all()


def table_errors(robot_from_camera):
    pixels = project_places(ROBOT_FROM_CAMERA, PLACES)
    located = [project_to_plane(u, v, K, robot_from_camera, CUBE_CENTER_HEIGHT)[:2] for u, v in pixels]
    return np.linalg.norm(np.array(located) - np.array(PLACES), axis=1)


def test_pixel_noise_alone_locates_table_cubes_within_a_millimeter_or_two():
    poses = gripper_poses()
    pixels = cube_pixels(ROBOT_FROM_CAMERA, poses) + np.random.default_rng(1).normal(0, 1.0, (len(poses), 2))
    result = solve_from_arm(pixels, poses, K)
    assert table_errors(result.robot_from_camera).max() < 0.003
    assert pose_error(result.robot_from_camera, ROBOT_FROM_CAMERA)[1] < 1.0


def test_three_millimeters_of_joint_noise_locate_table_cubes_within_a_centimeter():
    poses = gripper_poses()
    pixels = cube_pixels(ROBOT_FROM_CAMERA, poses) + np.random.default_rng(1).normal(0, 1.0, (len(poses), 2))
    result = solve_from_arm(pixels, noisy_poses(poses, 0.003), K)
    assert table_errors(result.robot_from_camera).mean() < 0.01
    assert np.linalg.norm(result.grasp_offset - GRASP_OFFSET) < 0.02
    assert 0 < result.residuals_px.mean() < 5


def test_two_wrong_detections_are_dropped_and_flagged():
    poses = gripper_poses()
    pixels = cube_pixels(ROBOT_FROM_CAMERA, poses)
    pixels[[2, 7]] += (120.0, -90.0)
    result = solve_from_arm(pixels, poses, K)
    assert not result.inliers[[2, 7]].any()
    assert result.inliers.sum() == len(poses) - 2
    assert pose_error(result.robot_from_camera, ROBOT_FROM_CAMERA)[0] < 0.01


def test_a_fixed_grasp_offset_is_enough_when_the_gripper_never_rotates():
    poses = gripper_poses(tilt=0.0)
    result = solve_from_arm(cube_pixels(ROBOT_FROM_CAMERA, poses, np.zeros(3)), poses, K, fit_offset=False)
    assert pose_error(result.robot_from_camera, ROBOT_FROM_CAMERA)[0] < 1e-4
    assert result.grasp_offset == pytest.approx(np.zeros(3))


def test_fewer_than_eight_poses_are_refused():
    poses = gripper_poses(count=7)
    with pytest.raises(ValueError, match="at least 8 arm poses"):
        solve_from_arm(cube_pixels(ROBOT_FROM_CAMERA, poses), poses, K)


def test_poses_at_one_height_are_refused():
    poses = gripper_poses()
    poses[:, 2, 3] = 0.1
    with pytest.raises(ValueError, match="lie on a plane"):
        solve_from_arm(cube_pixels(ROBOT_FROM_CAMERA, poses), poses, K)


def test_a_gripper_that_never_rotates_cannot_fit_the_grasp_offset():
    poses = gripper_poses(tilt=0.0)
    with pytest.raises(ValueError, match="orientation barely changes"):
        solve_from_arm(cube_pixels(ROBOT_FROM_CAMERA, poses), poses, K)


def test_pixels_that_agree_with_no_pose_are_refused():
    poses = gripper_poses()
    random_pixels = np.random.default_rng(2).uniform(0, 600, (len(poses), 2))
    with pytest.raises(ValueError, match="do not fit any camera pose"):
        solve_from_arm(random_pixels, poses, K)
