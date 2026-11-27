import math

import fake_arm
import pytest
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

TICKS_PER_RADIAN = 4095 / (2 * math.pi)
MIDDLE_TICK = (fake_arm.RANGE_MIN + fake_arm.RANGE_MAX) / 2


def follower(tmp_path):
    fake_arm.write_calibration(tmp_path)
    config = SO101FollowerConfig(port="/dev/fake", id="fake_arm", calibration_dir=tmp_path, use_radians=True)
    return SO101Follower(config)


@pytest.fixture
def arm(tmp_path, monkeypatch):
    bus = fake_arm.install(monkeypatch, fake_arm.calibrated_bus())
    robot = follower(tmp_path)
    robot.connect(calibrate=False)
    yield robot, bus
    if robot.is_connected:
        robot.disconnect()


def test_connect_accepts_a_matching_calibration(arm):
    robot, _ = arm
    assert robot.is_calibrated


def test_configure_puts_every_servo_in_position_mode(arm):
    _, bus = arm
    modes = {motor_id: motor["operating_mode"] for motor_id, motor in bus.registers.items()}
    assert set(modes.values()) == {0}


def test_configure_limits_the_gripper_torque(arm):
    _, bus = arm
    assert bus.registers[6]["max_torque_limit"] == 500
    assert "max_torque_limit" not in bus.registers[1]


def test_observation_reports_the_six_joints_in_radians(arm):
    robot, _ = arm
    observation = robot.get_observation()
    assert sorted(observation) == sorted(f"{name}.pos" for name in fake_arm.CALIBRATION)
    expected = (2048 - MIDDLE_TICK) / TICKS_PER_RADIAN
    assert observation["shoulder_pan.pos"] == pytest.approx(expected, abs=1e-3)


def test_observation_reports_the_gripper_as_a_percentage(arm):
    robot, _ = arm
    span = fake_arm.RANGE_MAX - fake_arm.RANGE_MIN
    assert robot.get_observation()["gripper.pos"] == pytest.approx((2048 - fake_arm.RANGE_MIN) / span * 100, abs=0.1)


def test_action_writes_goal_ticks_of_each_joint(arm):
    robot, bus = arm
    robot.send_action({"shoulder_pan.pos": 0.5, "elbow_flex.pos": -0.25, "gripper.pos": 50.0})
    assert bus.registers[1]["goal_position"] == pytest.approx(MIDDLE_TICK + 0.5 * TICKS_PER_RADIAN, abs=1)
    assert bus.registers[3]["goal_position"] == pytest.approx(MIDDLE_TICK - 0.25 * TICKS_PER_RADIAN, abs=1)
    assert bus.registers[6]["goal_position"] == pytest.approx(MIDDLE_TICK, abs=1)


def test_action_beyond_the_calibrated_range_is_clamped(arm):
    robot, bus = arm
    robot.send_action({"shoulder_pan.pos": 10.0, "wrist_flex.pos": -10.0, "gripper.pos": 250.0})
    assert bus.registers[1]["goal_position"] == fake_arm.RANGE_MAX
    assert bus.registers[4]["goal_position"] == fake_arm.RANGE_MIN
    assert bus.registers[6]["goal_position"] == fake_arm.RANGE_MAX


def test_action_leaves_the_other_joints_untouched(arm):
    robot, bus = arm
    robot.send_action({"shoulder_pan.pos": 0.5})
    assert bus.registers[2]["goal_position"] == 2048


def test_powered_servo_reaches_its_goal(arm):
    robot, bus = arm
    bus.set_torque([1], True)
    robot.send_action({"shoulder_pan.pos": 0.5})
    assert robot.get_observation()["shoulder_pan.pos"] == pytest.approx(0.5, abs=1e-2)


def test_disconnect_releases_the_torque_and_the_port(arm):
    robot, bus = arm
    robot.disconnect()
    assert bus.closed
    assert {motor["torque_enable"] for motor in bus.registers.values()} == {0}


def test_silent_servo_fails_the_handshake_with_its_id(tmp_path, monkeypatch):
    bus = fake_arm.install(monkeypatch, fake_arm.calibrated_bus(silent=(3,)))
    with pytest.raises(RuntimeError, match="Missing motor IDs:\n  - 3"):
        follower(tmp_path).connect(calibrate=False)
    assert bus.closed


def test_wrong_servo_model_fails_the_handshake(tmp_path, monkeypatch):
    bus = fake_arm.calibrated_bus()
    bus.registers[2]["model_number"] = 1190
    fake_arm.install(monkeypatch, bus)
    with pytest.raises(RuntimeError, match="incorrect model numbers"):
        follower(tmp_path).connect(calibrate=False)


def test_mixed_firmware_fails_the_handshake(tmp_path, monkeypatch):
    bus = fake_arm.calibrated_bus()
    bus.registers[5]["firmware_minor_version"] = 11
    fake_arm.install(monkeypatch, bus)
    with pytest.raises(RuntimeError, match="different firmware versions"):
        follower(tmp_path).connect(calibrate=False)


def test_motors_calibrated_for_another_arm_are_detected(tmp_path, monkeypatch):
    bus = fake_arm.calibrated_bus()
    bus.registers[1]["homing_offset"] = 300
    fake_arm.install(monkeypatch, bus)
    robot = follower(tmp_path)
    robot.connect(calibrate=False)
    assert not robot.is_calibrated
