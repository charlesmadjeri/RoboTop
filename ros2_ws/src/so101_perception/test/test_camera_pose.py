import math

import numpy as np
import pytest
from cube_scene import HEIGHT, K, PLACES, ROBOT_FROM_CAMERA, WIDTH, pose_error, project_places

from so101_perception.camera_pose import intrinsics_from_horizontal_fov, solve_camera_pose
from so101_perception.pose_file import load_pose, save_pose
from so101_perception.projection import CUBE_CENTER_HEIGHT


def test_exact_cube_pixels_give_back_the_camera_pose():
    pixels = project_places(ROBOT_FROM_CAMERA, PLACES)
    found, error = solve_camera_pose(pixels, np.array(PLACES), CUBE_CENTER_HEIGHT, K)
    translation, angle = pose_error(found, ROBOT_FROM_CAMERA)
    assert translation < 1e-6 and angle < 1e-4
    assert error < 1e-6


def test_four_places_are_enough():
    four = [PLACES[0], PLACES[1], PLACES[4], PLACES[5]]
    pixels = project_places(ROBOT_FROM_CAMERA, four)
    found, _ = solve_camera_pose(pixels, np.array(four), CUBE_CENTER_HEIGHT, K)
    assert pose_error(found, ROBOT_FROM_CAMERA)[0] < 1e-6


def test_noisy_pixels_move_the_pose_by_millimeters_and_raise_the_reprojection_error():
    pixels = project_places(ROBOT_FROM_CAMERA, PLACES)
    noisy = pixels + np.random.default_rng(1).normal(0, 0.5, pixels.shape)
    found, error = solve_camera_pose(noisy, np.array(PLACES), CUBE_CENTER_HEIGHT, K)
    translation, angle = pose_error(found, ROBOT_FROM_CAMERA)
    assert 0 < error < 1.5
    assert translation < 0.01 and angle < 1.0


def test_cube_height_shifts_the_plane_the_places_lie_on():
    lifted = np.array([[x, y, 0.05, 1.0] for x, y in PLACES]).T
    in_camera = (np.linalg.inv(ROBOT_FROM_CAMERA) @ lifted)[:3]
    pixels = K @ in_camera
    found, _ = solve_camera_pose((pixels[:2] / pixels[2]).T, np.array(PLACES), 0.05, K)
    assert pose_error(found, ROBOT_FROM_CAMERA)[0] < 1e-6


def test_fewer_than_four_places_are_refused():
    with pytest.raises(ValueError, match="at least 4 cube places"):
        solve_camera_pose(project_places(ROBOT_FROM_CAMERA, PLACES[:3]), np.array(PLACES[:3]), CUBE_CENTER_HEIGHT, K)


def test_places_on_a_line_are_refused():
    line = [(0.1 + 0.05 * step, 0.2) for step in range(5)]
    with pytest.raises(ValueError, match="lie on a line"):
        solve_camera_pose(project_places(ROBOT_FROM_CAMERA, line), np.array(line), CUBE_CENTER_HEIGHT, K)


def test_places_with_swapped_axes_are_refused_when_the_camera_comes_out_below_the_table():
    swapped = np.array(PLACES)[:, ::-1]
    with pytest.raises(ValueError, match="below the cube plane"):
        solve_camera_pose(project_places(ROBOT_FROM_CAMERA, PLACES), swapped, CUBE_CENTER_HEIGHT, K)


def test_field_of_view_sets_the_focal_length_and_centers_the_principal_point():
    matrix = intrinsics_from_horizontal_fov(WIDTH, HEIGHT, 75.0)
    assert matrix[0, 0] == pytest.approx(WIDTH / 2 / math.tan(math.radians(37.5)))
    assert matrix[0, 0] == matrix[1, 1]
    assert (matrix[0, 2], matrix[1, 2]) == (WIDTH / 2, HEIGHT / 2)


def test_saved_pose_loads_back_unchanged(tmp_path):
    save_pose(tmp_path / "pose.json", ROBOT_FROM_CAMERA)
    assert load_pose(tmp_path / "pose.json") == pytest.approx(ROBOT_FROM_CAMERA)


def test_pose_file_that_is_not_a_4x4_matrix_is_rejected(tmp_path):
    save_pose(tmp_path / "pose.json", np.eye(3))
    with pytest.raises(ValueError, match="4x4"):
        load_pose(tmp_path / "pose.json")
