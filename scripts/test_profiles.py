import json
from pathlib import Path

import pytest

import profiles

ROOT = Path(__file__).resolve().parent.parent
LAUNCH_KEYS = {
    ROOT / "ros2_ws/src/so101_bringup/launch/so101.launch.py": profiles.ALLOWED,
    ROOT / "ros2_ws/src/so101_driver/launch/driver.launch.py": profiles.DRIVER_KEYS,
}


def write(directory, name, settings):
    (directory / f"{name}.json").write_text(json.dumps(settings))


def test_profile_settings_are_loaded_by_name(tmp_path):
    write(tmp_path, "camx", {"camera_fourcc": "MJPG", "camera_width": 1280})
    assert profiles.load("camx", tmp_path) == {"camera_fourcc": "MJPG", "camera_width": 1280}


def test_unknown_profile_lists_the_available_ones(tmp_path):
    write(tmp_path, "camx", {})
    write(tmp_path, "camy", {})
    with pytest.raises(ValueError, match="camx, camy"):
        profiles.load("other", tmp_path)


def test_unknown_key_is_rejected_with_its_name(tmp_path):
    write(tmp_path, "camx", {"camera_fourc": "MJPG"})
    with pytest.raises(ValueError, match="camera_fourc"):
        profiles.load("camx", tmp_path)


def test_launch_arguments_are_key_value_pairs():
    assert profiles.launch_arguments({"camera_fourcc": "MJPG", "camera_fov_deg": 75.0}, {}) == [
        "camera_fourcc:=MJPG",
        "camera_fov_deg:=75.0",
    ]


def test_explicit_value_wins_over_the_profile():
    arguments = profiles.launch_arguments({"camera_index": 0, "camera_fourcc": "MJPG"}, {"camera_index": "2"})
    assert "camera_index:=2" in arguments and "camera_index:=0" not in arguments
    assert "camera_fourcc:=MJPG" in arguments


def test_missing_explicit_value_keeps_the_profile():
    assert profiles.launch_arguments({"camera_index": 0}, {"camera_index": None}) == ["camera_index:=0"]


def test_no_profile_and_no_value_adds_no_argument():
    assert profiles.launch_arguments({}, {"camera_index": None}) == []


def test_the_shipped_profile_is_valid():
    assert profiles.load("camx")["camera_fourcc"] == "MJPG"


@pytest.mark.parametrize("launch_file", LAUNCH_KEYS, ids=lambda path: path.name)
def test_launch_files_declare_the_torque_argument(launch_file):
    assert '"enable_torque"' in launch_file.read_text()


@pytest.mark.parametrize("launch_file", LAUNCH_KEYS, ids=lambda path: path.name)
def test_launch_files_declare_every_profile_key(launch_file):
    text = launch_file.read_text()
    assert [key for key in LAUNCH_KEYS[launch_file] if f'"{key}"' not in text] == []
