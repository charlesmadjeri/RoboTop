#!/usr/bin/env python3
import argparse
import math
import os
import random
import sys
import time
from pathlib import Path

import backends
import display
import images
import overlay
import screens
from processes import Managed

ROOT = Path(__file__).resolve().parents[2]

JOINT = "shoulder_pan"
MIN_JOINT_RATE = 20.0
MIN_IMAGE_RATE = 1.0


def start_stack(out, env, launch_arguments):
    command = ["ros2", "launch", "--noninteractive", "so101_bringup", "so101.launch.py", *launch_arguments]
    return Managed("stack", command, out / "launch.log", env)


def start_rviz(out, env):
    command = ["rviz2", "-d", "so101.rviz", "--ros-args", "--log-level", "rviz2:=warn"]
    return Managed("rviz2", command, out / "rviz.log", env, cwd=ROOT / "ros2_ws")


def check_topics(probe, alive):
    probe.wait_until(lambda: probe.joints and probe.frame is not None and probe.info, 90, "joint states, camera frame and camera info", alive)
    if len(probe.joints) != 6 or not all(math.isfinite(value) for value in probe.joints.values()):
        raise RuntimeError(f"joint states are not 6 finite values: {probe.joints}")
    rates = probe.rates(4)
    if rates["joints"] < MIN_JOINT_RATE:
        raise RuntimeError(f"/joint_states at {rates['joints']:.1f} Hz, expected at least {MIN_JOINT_RATE}")
    if rates["image"] < MIN_IMAGE_RATE:
        raise RuntimeError(f"/external_cam/image_raw at {rates['image']:.1f} Hz, expected at least {MIN_IMAGE_RATE}")
    print(f"topics ok: joints {rates['joints']:.1f} Hz, image {rates['image']:.1f} Hz, {probe.frame.shape[1]}x{probe.frame.shape[0]}")


def wait_rviz_drawn(probe, shown, out, timeout=60):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        probe.spin_for(1.0)
        screens.grab(shown, out / "rviz_probe.png")
        if not images.is_blank(images.read(out / "rviz_probe.png")):
            return
    raise TimeoutError("RViz window stayed blank")


def snapshot(probe, shown, out, label):
    probe.spin_for(2.0)
    screens.grab(shown, out / f"rviz_{label}.png")
    images.save_rgb(probe.frame, out / f"camera_{label}.png")
    return images.read(out / f"rviz_{label}.png"), probe.frame.copy()


def scenario(out, shown, alive, target):
    import rclpy
    from probe import Probe

    rclpy.init()
    probe = Probe()
    try:
        check_topics(probe, alive)
        if images.is_blank(probe.frame):
            raise RuntimeError("sim camera frame is a uniform image")
        wait_rviz_drawn(probe, shown, out)
        rviz_before, camera_before = snapshot(probe, shown, out, "before")
        start = probe.joints[JOINT]
        probe.move_to(JOINT, target, 0.05, 30, alive)
        rviz_after, camera_after = snapshot(probe, shown, out, "after")
        moved = abs(probe.joints[JOINT] - start)
        diffs = {"rviz": images.mean_abs_diff(rviz_before, rviz_after), "camera": images.mean_abs_diff(camera_before, camera_after)}
        print(f"{JOINT} moved {moved:.2f} rad, image diff rviz {diffs['rviz']:.2f}, camera {diffs['camera']:.2f}")
        if moved < 0.3:
            raise RuntimeError(f"{JOINT} moved only {moved:.2f} rad")
        for name, before, after in (("RViz", rviz_before, rviz_after), ("camera", camera_before, camera_after)):
            if not images.has_changed(before, after):
                raise RuntimeError(f"{name} image did not change after the move")
    finally:
        probe.destroy_node()
        rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts")
    parser.add_argument("--target", type=float, default=0.5, help="shoulder_pan target in radians")
    parser.add_argument("--backend", choices=("sim", "fake-real"), default="sim")
    parser.add_argument("--camera-index", type=int, default=-1, help="fake-real: USB camera index, -1 for none")
    args = parser.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    for stale in [*out.glob("*.png"), *(out / name for name in ("launch.log", "rviz.log", "fake_bus.log"))]:
        stale.unlink(missing_ok=True)
    overlay.load()
    xvfb, shown = display.start()
    env = {
        **os.environ,
        "DISPLAY": shown,
        "QT_QPA_PLATFORM": "xcb",
        "LIBGL_ALWAYS_SOFTWARE": "1",
        "ROS_DOMAIN_ID": str(random.randint(1, 100)),
    }
    env.setdefault("MUJOCO_GL", "egl")
    started = []

    def alive():
        for process in started:
            process.check()

    try:
        if args.backend == "sim":
            launch_arguments = backends.SIM
        else:
            launch_arguments = backends.fake_real(out, env, args.camera_index)
        os.environ.update(env)
        started.append(start_stack(out, env, launch_arguments))
        if args.backend == "sim":
            started.append(start_rviz(out, env))
            scenario(out, shown, alive, args.target)
        else:
            import fake_arm
            import hardware

            hardware.scenario(fake_arm, out / "fake_bus.log", alive, args.camera_index >= 0)
    except Exception as error:
        print(f"FAIL: {error}")
        try:
            screens.grab(shown, out / "rviz_failure.png")
        except Exception:
            pass
        return 1
    finally:
        for process in reversed(started):
            process.stop()
        xvfb.terminate()
    print("functional test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
