import math

import numpy as np
import pytest
from marker_scene import K, ROBOT_FROM_CAMERA, WIDTH, HEIGHT, pose_error, project_marker

from so101_perception.camera_pose import intrinsics_from_horizontal_fov, robot_from_marker, solve_camera_pose
from so101_perception.pose_file import load_pose, save_pose

SIZE = 0.1


@pytest.mark.parametrize("position, yaw", [((0.2, 0.15, 0.0), 0.0), ((0.25, 0.05, 0.0), 25.0), ((0.1, 0.2, 0.0), -70.0)])
def test_exact_marker_corners_give_back_the_camera_pose(position, yaw):
    marker_pose = robot_from_marker(position, yaw)
    corners = project_marker(ROBOT_FROM_CAMERA, marker_pose, SIZE)
    found, error = solve_camera_pose(corners, SIZE, marker_pose, K)
    translation, angle = pose_error(found, ROBOT_FROM_CAMERA)
    assert translation < 1e-6 and angle < 1e-4
    assert error < 1e-6


def test_noisy_corners_move_the_pose_by_millimeters_and_raise_the_reprojection_error():
    marker_pose = robot_from_marker((0.2, 0.15, 0.0), 25.0)
    corners = project_marker(ROBOT_FROM_CAMERA, marker_pose, SIZE)
    noisy = corners + np.random.default_rng(1).normal(0, 0.3, corners.shape)
    found, error = solve_camera_pose(noisy, SIZE, marker_pose, K)
    translation, _ = pose_error(found, ROBOT_FROM_CAMERA)
    assert 0 < error < 1.0
    assert translation < 0.01


def test_marker_yaw_rotates_the_marker_around_the_vertical_axis():
    pose = robot_from_marker((0.1, 0.2, 0.0), 90.0)
    assert pose[:3, :3] @ [1, 0, 0] == pytest.approx([0, 1, 0], abs=1e-12)
    assert pose[:3, 3] == pytest.approx([0.1, 0.2, 0.0])


def test_field_of_view_sets_the_focal_length_and_centers_the_principal_point():
    matrix = intrinsics_from_horizontal_fov(WIDTH, HEIGHT, 70.0)
    assert matrix[0, 0] == pytest.approx(WIDTH / 2 / math.tan(math.radians(35)))
    assert matrix[0, 0] == matrix[1, 1]
    assert (matrix[0, 2], matrix[1, 2]) == (WIDTH / 2, HEIGHT / 2)


def test_saved_pose_loads_back_unchanged(tmp_path):
    save_pose(tmp_path / "pose.json", ROBOT_FROM_CAMERA)
    assert load_pose(tmp_path / "pose.json") == pytest.approx(ROBOT_FROM_CAMERA)


def test_pose_file_that_is_not_a_4x4_matrix_is_rejected(tmp_path):
    save_pose(tmp_path / "pose.json", np.eye(3))
    with pytest.raises(ValueError, match="4x4"):
        load_pose(tmp_path / "pose.json")
