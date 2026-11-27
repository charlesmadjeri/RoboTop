from so101_driver.torque import set_torque


class RecordingBus:
    def __init__(self):
        self.calls = []

    def disable_torque(self):
        self.calls.append("disable")

    def enable_torque(self):
        self.calls.append("enable")

    def sync_read(self, name):
        self.calls.append(f"read {name}")
        return {"shoulder_pan": 0.3}

    def sync_write(self, name, values):
        self.calls.append(f"write {name} {values}")


class Robot:
    def __init__(self):
        self.bus = RecordingBus()


def test_torque_stays_off_by_default():
    robot = Robot()
    set_torque(robot, False)
    assert robot.bus.calls == ["disable"]


def test_enabling_the_torque_first_holds_the_present_position():
    robot = Robot()
    set_torque(robot, True)
    assert robot.bus.calls == [
        "disable",
        "read Present_Position",
        "write Goal_Position {'shoulder_pan': 0.3}",
        "enable",
    ]
