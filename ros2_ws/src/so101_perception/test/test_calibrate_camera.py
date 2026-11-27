import numpy as np
import pytest
from cube_scene import HEIGHT, K, PLACES, ROBOT_FROM_CAMERA, WIDTH, pose_error, project_places, render_cube_view, write_view

from so101_perception.calibrate_camera import main
from so101_perception.pose_file import load_pose


def write_samples(tmp_path, places=PLACES):
    pixels = project_places(ROBOT_FROM_CAMERA, PLACES)
    arguments = []
    for index, (pixel, (x, y)) in enumerate(zip(pixels, places)):
        path = write_view(tmp_path / f"cube{index}.png", render_cube_view(pixel))
        arguments += ["--sample", str(path), str(x), str(y)]
    return arguments


def test_command_saves_the_pose_file_and_reports_the_error(tmp_path, capsys):
    output = tmp_path / "pose.json"
    main([*write_samples(tmp_path), "--fov-deg", "75", "--output", str(output)])
    translation, angle = pose_error(load_pose(output), ROBOT_FROM_CAMERA)
    assert translation < 0.01 and angle < 0.5
    report = capsys.readouterr().out
    assert "reprojection error" in report and "worst" in report


def test_command_takes_measured_intrinsics_in_place_of_the_field_of_view(tmp_path):
    output = tmp_path / "pose.json"
    intrinsics = [str(K[0, 0]), str(K[1, 1]), str(K[0, 2]), str(K[1, 2])]
    main([*write_samples(tmp_path), "--fov-deg", "30", "--intrinsics", *intrinsics, "--output", str(output)])
    assert pose_error(load_pose(output), ROBOT_FROM_CAMERA)[0] < 0.01


def test_command_names_the_place_that_was_measured_wrong(tmp_path, capsys):
    wrong = list(PLACES)
    wrong[2] = (wrong[2][0] + 0.06, wrong[2][1])
    main([*write_samples(tmp_path, wrong), "--fov-deg", "75", "--output", str(tmp_path / "pose.json")])
    assert "cube2.png" in capsys.readouterr().out.splitlines()[-1]


def test_command_exits_with_the_file_when_the_cube_is_missing(tmp_path):
    arguments = write_samples(tmp_path)
    blank = write_view(tmp_path / "blank.png", np.full((HEIGHT, WIDTH, 3), 128, np.uint8))
    with pytest.raises(SystemExit, match="cube not found in .*blank.png"):
        main([*arguments, "--sample", str(blank), "0.2", "0.2", "--output", str(tmp_path / "pose.json")])
    assert not (tmp_path / "pose.json").exists()


def test_command_exits_when_an_image_cannot_be_read(tmp_path):
    with pytest.raises(SystemExit, match="cannot read the image"):
        main(["--sample", str(tmp_path / "missing.png"), "0.2", "0.2"])


def test_command_exits_with_the_reason_when_too_few_places_are_given(tmp_path):
    arguments = write_samples(tmp_path)[:12]
    with pytest.raises(SystemExit, match="at least 4 cube places"):
        main([*arguments, "--output", str(tmp_path / "pose.json")])
