import json
import math

import images
import rclpy
from probe import Probe

JOINT = "shoulder_pan"
TARGET = 0.5
MIN_JOINT_RATE = 20.0
MIN_IMAGE_RATE = 1.0
TICK_TOLERANCE = 2


def goal_ticks(log, motor_id):
    if not log.is_file():
        return []
    records = (json.loads(line) for line in log.read_text().splitlines())
    return [record["value"] for record in records if record["id"] == motor_id]


def expected_ticks(fake_arm, radians):
    middle = (fake_arm.RANGE_MIN + fake_arm.RANGE_MAX) / 2
    return middle + radians * 4095 / (2 * math.pi)


def check_camera(probe, alive):
    probe.wait_until(lambda: probe.frame is not None and probe.info, 30, "virtual camera frame and camera info", alive)
    rates = probe.rates(4)
    if rates["image"] < MIN_IMAGE_RATE:
        raise RuntimeError(f"/external_cam/image_raw at {rates['image']:.1f} Hz, expected at least {MIN_IMAGE_RATE}")
    if images.is_blank(probe.frame):
        raise RuntimeError("virtual camera frame is a uniform image")
    print(f"virtual camera ok: image {rates['image']:.1f} Hz, {probe.frame.shape[1]}x{probe.frame.shape[0]}")


def scenario(fake_arm, log, alive, with_camera):
    rclpy.init()
    probe = Probe()
    try:
        probe.wait_until(lambda: probe.joints, 60, "joint states", alive)
        if len(probe.joints) != 6 or not all(math.isfinite(value) for value in probe.joints.values()):
            raise RuntimeError(f"joint states are not 6 finite values: {probe.joints}")
        rate = probe.rates(4)["joints"]
        if rate < MIN_JOINT_RATE:
            raise RuntimeError(f"/joint_states at {rate:.1f} Hz, expected at least {MIN_JOINT_RATE}")
        rest = probe.joints[JOINT]
        print(f"joint states ok: {rate:.1f} Hz, {JOINT} at rest {rest:.3f} rad")
        if with_camera:
            check_camera(probe, alive)
        else:
            print("virtual camera skipped: no camera index given")
        for _ in range(10):
            probe.command(JOINT, TARGET)
            probe.spin_for(0.2)
            alive()
        sent = goal_ticks(log, 1)
        wanted = expected_ticks(fake_arm, TARGET)
        if not sent or abs(sent[-1] - wanted) > TICK_TOLERANCE:
            raise RuntimeError(f"goal ticks of {JOINT} are {sent[-3:]}, expected {wanted:.0f}")
        print(f"command path ok: goal {sent[-1]} ticks for {TARGET} rad")
        if abs(probe.joints[JOINT] - rest) > 0.01:
            raise RuntimeError(f"{JOINT} moved with the torque disabled: {rest:.3f} to {probe.joints[JOINT]:.3f}")
        print("torque disabled: the arm stays where it is")
    finally:
        probe.destroy_node()
        rclpy.shutdown()
