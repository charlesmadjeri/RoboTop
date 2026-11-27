import numpy as np
import pytest

from so101_perception.arm_calibration import MIN_POSES, check_poses
from so101_perception.arm_poses import (
    JOINT_LIMITS,
    PRESET_POSES,
    check_within_limits,
    interpolate,
    load_poses,
)

PATH_STEP = 0.05
TABLE_CLEARANCE = 0.04
IMAGE_MARGIN = 25


def test_the_presets_are_enough_poses_and_stay_within_the_joint_limits():
    assert len(PRESET_POSES) >= 10
    assert len(PRESET_POSES) >= MIN_POSES + 2
    check_within_limits(PRESET_POSES)


def test_the_presets_are_valid_for_the_solver_and_inside_the_simulated_camera_view(arm):
    poses = []
    for joints in PRESET_POSES:
        arm.place(joints)
        pose = arm.gripper_pose()
        u, v = arm.pixel_of(pose[:3, 3])
        assert IMAGE_MARGIN < u < arm.sim.camera_width - IMAGE_MARGIN
        assert IMAGE_MARGIN < v < arm.sim.camera_height - IMAGE_MARGIN
        poses.append(pose)
    check_poses(np.array(poses), fit_offset=True)


def test_the_arm_never_touches_the_table_or_itself_on_the_way_through_the_presets(arm):
    start = np.zeros(5)
    for target in PRESET_POSES:
        for joints in interpolate(start, target, PATH_STEP):
            arm.place(joints)
            assert arm.gripper_pose()[2, 3] > TABLE_CLEARANCE
            assert not arm.touching()
        start = target


def test_a_pose_beyond_a_joint_limit_is_refused_with_its_number_and_joint():
    poses = PRESET_POSES.copy()
    poses[3, 2] = JOINT_LIMITS[2, 1]
    with pytest.raises(ValueError, match="pose 4 puts elbow_flex too close to its limit"):
        check_within_limits(poses)


def test_a_pose_with_a_missing_angle_is_refused():
    poses = PRESET_POSES.copy()
    poses[0, 0] = np.nan
    with pytest.raises(ValueError, match="pose 1 puts shoulder_pan too close to its limit"):
        check_within_limits(poses)


def test_an_interpolation_ends_on_the_target_with_steps_no_larger_than_asked():
    start, end = np.zeros(5), np.array([1.0, -0.2, 0.0, 0.5, 0.0])
    path = interpolate(start, end, 0.1)
    assert np.allclose(path[-1], end)
    assert np.abs(np.diff(np.vstack([start, path]), axis=0)).max() <= 0.1 + 1e-9
    assert len(interpolate(end, end, 0.1)) == 1


def test_a_poses_file_is_loaded_and_a_malformed_one_is_refused(tmp_path):
    good = tmp_path / "poses.json"
    good.write_text("[[0, 0, 0, 0, 0], [0.1, 0.2, 0.3, 0.4, 0.5]]")
    assert load_poses(good).shape == (2, 5)
    bad = tmp_path / "bad.json"
    bad.write_text("[[0, 0, 0]]")
    with pytest.raises(ValueError, match="5 joint angles"):
        load_poses(bad)
