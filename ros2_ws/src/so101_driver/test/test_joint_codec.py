import pytest

from so101_driver.joint_codec import (
    GRIPPER_RANGE_RAD,
    JOINT_NAMES,
    command_to_action,
    gripper_pct_to_rad,
    observation_to_positions,
)


def full_observation(gripper_pct: float) -> dict:
    observation = {f"{name}.pos": index * 0.1 for index, name in enumerate(JOINT_NAMES)}
    observation["gripper.pos"] = gripper_pct
    return observation


def test_gripper_closed_maps_to_lower_urdf_limit():
    assert gripper_pct_to_rad(0.0) == pytest.approx(GRIPPER_RANGE_RAD[0])


def test_gripper_open_maps_to_upper_urdf_limit():
    assert gripper_pct_to_rad(100.0) == pytest.approx(GRIPPER_RANGE_RAD[1])


def test_gripper_percentage_outside_range_is_clipped():
    assert gripper_pct_to_rad(-20.0) == pytest.approx(GRIPPER_RANGE_RAD[0])
    assert gripper_pct_to_rad(250.0) == pytest.approx(GRIPPER_RANGE_RAD[1])


def test_positions_follow_joint_names_order():
    positions = observation_to_positions(full_observation(0.0))
    assert positions[:5] == pytest.approx([0.0, 0.1, 0.2, 0.3, 0.4])


def test_positions_convert_only_the_gripper():
    positions = observation_to_positions(full_observation(100.0))
    assert positions[4] == pytest.approx(0.4)
    assert positions[5] == pytest.approx(GRIPPER_RANGE_RAD[1])


def test_command_is_matched_by_name_not_by_index():
    action = command_to_action(["elbow_flex", "shoulder_pan"], [0.5, -0.5])
    assert action == {"elbow_flex.pos": 0.5, "shoulder_pan.pos": -0.5}


def test_command_ignores_unknown_joint_names():
    assert command_to_action(["unknown", "gripper"], [1.0, 40.0]) == {"gripper.pos": 40.0}


def test_command_keeps_gripper_in_percent():
    assert command_to_action(["gripper"], [75.0]) == {"gripper.pos": 75.0}


def test_command_drops_values_that_are_not_finite():
    action = command_to_action(["shoulder_pan", "elbow_flex", "wrist_flex"], [float("nan"), float("inf"), 0.2])
    assert action == {"wrist_flex.pos": 0.2}
