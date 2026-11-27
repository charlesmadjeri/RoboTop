import json
from pathlib import Path

from fake_bus import FakeBus
from lerobot.motors import SerialMotorsBus
from so101_driver.joint_codec import JOINT_NAMES

RANGE_MIN = 1000
RANGE_MAX = 3000
CALIBRATION = {
    name: {"id": motor_id, "drive_mode": 0, "homing_offset": 0, "range_min": RANGE_MIN, "range_max": RANGE_MAX}
    for motor_id, name in enumerate(JOINT_NAMES, start=1)
}


def write_calibration(directory, name="fake_arm"):
    path = Path(directory) / f"{name}.json"
    path.write_text(json.dumps(CALIBRATION))
    return path


def calibrated_bus(bus_class=FakeBus, **kwargs):
    registers = {
        motor["id"]: {
            "homing_offset": motor["homing_offset"],
            "min_position_limit": motor["range_min"],
            "max_position_limit": motor["range_max"],
        }
        for motor in CALIBRATION.values()
    }
    return bus_class(registers=registers, **kwargs)


def install(monkeypatch, bus):
    monkeypatch.setattr(SerialMotorsBus, "_open_bus", lambda self: bus)
    return bus


def refuse_port(monkeypatch):
    def fail(self):
        raise OSError("no such port")

    monkeypatch.setattr(SerialMotorsBus, "_open_bus", fail)
