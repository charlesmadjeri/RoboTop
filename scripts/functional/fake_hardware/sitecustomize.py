import json
import os
import sys
from pathlib import Path

if os.environ.get("SO101_FAKE_BUS") == "1" and Path(sys.argv[0]).name == "driver":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "ros2_ws" / "src" / "so101_driver" / "test"))
    import fake_arm
    from fake_bus import FakeBus
    from lerobot.motors import SerialMotorsBus

    class RecordingBus(FakeBus):
        def _store(self, motor_id, name, value):
            super()._store(motor_id, name, value)
            if name == "goal_position":
                with open(os.environ["SO101_FAKE_BUS_LOG"], "a") as log:
                    log.write(json.dumps({"id": motor_id, "name": name, "value": int(value)}) + "\n")

    _bus = fake_arm.calibrated_bus(bus_class=RecordingBus)
    SerialMotorsBus._open_bus = lambda self: _bus
