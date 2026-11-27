import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FAKE_HARDWARE = ROOT / "scripts" / "functional" / "fake_hardware"
DRIVER = ROOT / "ros2_ws" / "src" / "so101_driver"
DRIVER_TESTS = DRIVER / "test"

SIM = ["use_sim:=true", "camera_index:=-1"]


def fake_real(out, env, camera_index):
    sys.path[:0] = [str(DRIVER_TESTS), str(DRIVER)]
    import fake_arm

    calibration = fake_arm.write_calibration(out)
    env["SO101_FAKE_BUS"] = "1"
    env["SO101_FAKE_BUS_LOG"] = str(out / "fake_bus.log")
    env["SO101_CALIBRATION_FILE"] = str(calibration)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(FAKE_HARDWARE), env.get("PYTHONPATH")]))
    return ["use_sim:=false", "port:=/dev/fake", f"camera_index:={camera_index}"]
