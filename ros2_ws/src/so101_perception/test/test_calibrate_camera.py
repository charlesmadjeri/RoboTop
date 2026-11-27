import cv2
import numpy as np
import pytest
from marker_scene import HEIGHT, K, ROBOT_FROM_CAMERA, WIDTH, pose_error, project_marker, render_marker_view

from so101_perception.calibrate_camera import MarkerNotFound, calibrate, main
from so101_perception.camera_pose import robot_from_marker
from so101_perception.marker_detection import find_marker_corners
from so101_perception.pose_file import load_pose

SIZE = 0.1
VIEWS = [((0.2, 0.15, 0.0), 0.0), ((0.25, 0.05, 0.0), 25.0), ((0.1, 0.2, 0.0), -70.0)]
MARKER_ARGS = ["--marker-size", str(SIZE), "--marker-xyz", "0.25", "0.05", "0.0", "--marker-yaw", "25"]


@pytest.mark.parametrize("position, yaw", VIEWS)
def test_marker_corners_are_found_within_a_pixel_of_their_projection(position, yaw):
    marker_pose = robot_from_marker(position, yaw)
    found = find_marker_corners(render_marker_view(ROBOT_FROM_CAMERA, marker_pose, SIZE), 0)
    expected = project_marker(ROBOT_FROM_CAMERA, marker_pose, SIZE)
    assert np.abs(found - expected).max() < 1.0


def test_marker_with_another_id_is_ignored():
    view = render_marker_view(ROBOT_FROM_CAMERA, robot_from_marker((0.2, 0.15, 0.0), 0.0), SIZE, marker_id=7)
    assert find_marker_corners(view, 0) is None


def test_image_without_marker_has_no_corners():
    assert find_marker_corners(np.full((HEIGHT, WIDTH, 3), 128, np.uint8), 0) is None


@pytest.mark.parametrize("position, yaw", VIEWS)
def test_calibration_from_a_rendered_view_recovers_the_camera_pose(position, yaw):
    marker_pose = robot_from_marker(position, yaw)
    view = render_marker_view(ROBOT_FROM_CAMERA, marker_pose, SIZE)
    found, error = calibrate(view, K, 0, SIZE, marker_pose)
    translation, angle = pose_error(found, ROBOT_FROM_CAMERA)
    assert translation < 0.01 and angle < 1.0
    assert error < 1.0


def test_calibration_without_marker_names_the_missing_id():
    blank = np.full((HEIGHT, WIDTH, 3), 128, np.uint8)
    with pytest.raises(MarkerNotFound, match="marker 0"):
        calibrate(blank, K, 0, SIZE, np.eye(4))


def write_view(tmp_path, view):
    path = tmp_path / "frame.png"
    cv2.imwrite(str(path), cv2.cvtColor(view, cv2.COLOR_RGB2BGR))
    return path


def test_command_saves_the_pose_file_and_reports_the_error(tmp_path, capsys):
    view = render_marker_view(ROBOT_FROM_CAMERA, robot_from_marker((0.25, 0.05, 0.0), 25.0), SIZE)
    output = tmp_path / "pose.json"
    main(["--image", str(write_view(tmp_path, view)), "--output", str(output), *MARKER_ARGS])
    translation, angle = pose_error(load_pose(output), ROBOT_FROM_CAMERA)
    assert translation < 0.01 and angle < 1.0
    assert "reprojection error" in capsys.readouterr().out


def test_command_takes_measured_intrinsics_in_place_of_the_field_of_view(tmp_path):
    view = render_marker_view(ROBOT_FROM_CAMERA, robot_from_marker((0.25, 0.05, 0.0), 25.0), SIZE)
    output = tmp_path / "pose.json"
    intrinsics = [str(K[0, 0]), str(K[1, 1]), str(K[0, 2]), str(K[1, 2])]
    main(["--image", str(write_view(tmp_path, view)), "--fov-deg", "30", "--intrinsics", *intrinsics,
          "--output", str(output), *MARKER_ARGS])
    assert pose_error(load_pose(output), ROBOT_FROM_CAMERA)[0] < 0.01


def test_command_exits_with_the_reason_when_the_marker_is_missing(tmp_path):
    blank = np.full((HEIGHT, WIDTH, 3), 128, np.uint8)
    with pytest.raises(SystemExit, match="marker 0 not found"):
        main(["--image", str(write_view(tmp_path, blank)), "--output", str(tmp_path / "pose.json"), *MARKER_ARGS])
    assert not (tmp_path / "pose.json").exists()


def test_command_exits_when_the_image_cannot_be_read(tmp_path):
    with pytest.raises(SystemExit, match="cannot read the image"):
        main(["--image", str(tmp_path / "missing.png"), *MARKER_ARGS])
