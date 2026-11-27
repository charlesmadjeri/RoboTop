import logging
from types import SimpleNamespace

import fake_arm
import pytest
from so101_driver.driver_node import DriverNode


def connect(port="/dev/fake", enable_torque=False):
    node = SimpleNamespace(_use_sim=False, get_logger=lambda: logging.getLogger("driver"))
    return DriverNode._connect_arm(node, port, enable_torque)


@pytest.fixture
def calibration_file(tmp_path, monkeypatch):
    path = fake_arm.write_calibration(tmp_path)
    monkeypatch.setenv("SO101_CALIBRATION_FILE", str(path))
    return path


def test_empty_port_is_refused_with_the_launch_hint(calibration_file):
    with pytest.raises(RuntimeError, match="No serial port"):
        connect(port="")


def test_unset_calibration_variable_is_named(monkeypatch):
    monkeypatch.delenv("SO101_CALIBRATION_FILE", raising=False)
    with pytest.raises(RuntimeError, match="SO101_CALIBRATION_FILE is not set"):
        connect()


def test_missing_calibration_file_is_reported_with_its_path(tmp_path, monkeypatch):
    missing = tmp_path / "absent.json"
    monkeypatch.setenv("SO101_CALIBRATION_FILE", str(missing))
    with pytest.raises(RuntimeError, match="No calibration file at .*absent.json"):
        connect()


def test_unopenable_port_raises_a_connection_error(calibration_file, monkeypatch):
    fake_arm.refuse_port(monkeypatch)
    with pytest.raises(ConnectionError, match="Could not connect on port '/dev/fake'"):
        connect()


def test_silent_servo_aborts_the_start(calibration_file, monkeypatch):
    bus = fake_arm.install(monkeypatch, fake_arm.calibrated_bus(silent=(2,)))
    with pytest.raises(RuntimeError, match="Missing motor IDs"):
        connect()
    assert bus.closed


def test_calibration_of_another_arm_disconnects_and_aborts(calibration_file, monkeypatch):
    bus = fake_arm.calibrated_bus()
    bus.registers[4]["min_position_limit"] = 10
    fake_arm.install(monkeypatch, bus)
    with pytest.raises(RuntimeError, match="does not match the motors"):
        connect()
    assert bus.closed


def test_connected_arm_has_its_torque_released(calibration_file, monkeypatch):
    bus = fake_arm.install(monkeypatch, fake_arm.calibrated_bus())
    robot = connect()
    assert robot.is_connected
    assert {motor["torque_enable"] for motor in bus.registers.values()} == {0}


def test_released_arm_keeps_its_position_when_commanded(calibration_file, monkeypatch):
    bus = fake_arm.install(monkeypatch, fake_arm.calibrated_bus())
    robot = connect()
    robot.send_action({"shoulder_pan.pos": 0.5})
    assert bus.registers[1]["goal_position"] != 2048
    assert bus.registers[1]["present_position"] == 2048


def test_enabled_torque_holds_the_present_position_instead_of_the_stale_goal(calibration_file, monkeypatch):
    bus = fake_arm.install(monkeypatch, fake_arm.calibrated_bus())
    bus.registers[1]["present_position"] = 2300
    bus.registers[1]["goal_position"] = 1000
    robot = connect(enable_torque=True)
    assert robot.is_connected
    assert {motor["torque_enable"] for motor in bus.registers.values()} == {1}
    assert bus.registers[1]["goal_position"] == 2300
