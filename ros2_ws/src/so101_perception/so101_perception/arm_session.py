from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from so101_perception.arm_poses import interpolate

COMMAND_RATE_HZ = 20.0
ARRIVAL_TOLERANCE_RAD = 0.04
ARRIVAL_TIMEOUT_S = 6.0
ARRIVAL_POLL_S = 0.2
EXPOSURE_WAIT_S = 0.5
FRAMES_PER_POSE = 3
MIN_DETECTIONS_PER_POSE = 2


class Arm(Protocol):
    def joint_positions(self) -> np.ndarray | None: ...
    def command(self, joints: np.ndarray) -> None: ...
    def gripper_pose(self) -> np.ndarray | None: ...
    def wait(self, seconds: float) -> None: ...


class Camera(Protocol):
    def frame(self) -> np.ndarray | None: ...


@dataclass(frozen=True)
class Samples:
    pixels: np.ndarray
    gripper_poses: np.ndarray
    kept: list[int]
    skipped: list[int]


def travel_to(arm: Arm, target: np.ndarray, speed: float) -> None:
    start = arm.joint_positions()
    if start is None:
        raise RuntimeError("no /joint_states received: is the driver running?")
    for waypoint in interpolate(start, target, speed / COMMAND_RATE_HZ):
        arm.command(waypoint)
        arm.wait(1.0 / COMMAND_RATE_HZ)


def wait_for_arrival(arm: Arm, target: np.ndarray) -> None:
    waited = 0.0
    while waited <= ARRIVAL_TIMEOUT_S:
        positions = arm.joint_positions()
        if positions is not None and np.abs(positions - target).max() <= ARRIVAL_TOLERANCE_RAD:
            arm.wait(EXPOSURE_WAIT_S)
            return
        arm.wait(ARRIVAL_POLL_S)
        waited += ARRIVAL_POLL_S
    raise RuntimeError("the arm did not reach the commanded pose, stop and check it")


def cube_pixel(camera: Camera, detect: Callable[[np.ndarray], tuple[float, float] | None]) -> np.ndarray | None:
    found = []
    for _ in range(FRAMES_PER_POSE):
        frame = camera.frame()
        pixel = None if frame is None else detect(frame)
        if pixel is not None:
            found.append(pixel)
    if len(found) < MIN_DETECTIONS_PER_POSE:
        return None
    return np.median(np.array(found, dtype=float), axis=0)


def collect_samples(
    arm: Arm,
    camera: Camera,
    detect: Callable[[np.ndarray], tuple[float, float] | None],
    poses: np.ndarray,
    speed: float,
    report: Callable[[str], None] = print,
) -> Samples:
    pixels, gripper_poses, kept, skipped = [], [], [], []
    for number, target in enumerate(poses, start=1):
        travel_to(arm, target, speed)
        wait_for_arrival(arm, target)
        gripper_pose = arm.gripper_pose()
        pixel = None if gripper_pose is None else cube_pixel(camera, detect)
        if pixel is None:
            skipped.append(number)
            report(f"pose {number}/{len(poses)}: cube or gripper frame not found, skipped")
            continue
        pixels.append(pixel)
        gripper_poses.append(gripper_pose)
        kept.append(number)
        report(f"pose {number}/{len(poses)}: cube at pixel ({pixel[0]:.0f}, {pixel[1]:.0f})")
    return Samples(np.array(pixels), np.array(gripper_poses), kept, skipped)
