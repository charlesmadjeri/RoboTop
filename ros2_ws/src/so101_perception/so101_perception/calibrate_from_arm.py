import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rclpy

from so101_perception.arm_calibration import ArmCalibration, solve_from_arm
from so101_perception.arm_poses import PRESET_POSES, check_within_limits, load_poses
from so101_perception.arm_session import Samples, collect_samples
from so101_perception.arm_validation import leave_one_out_errors
from so101_perception.camera_pose import intrinsics_from_horizontal_fov
from so101_perception.detection import find_cube
from so101_perception.pose_file import save_pose
from so101_perception.ros_rig import RosArm, RosCamera, read_camera_matrix

FALLBACK_FOV_DEG = 75.0
MAX_HELD_OUT_CM = 1.0
DEFAULT_SPEED_RAD_S = 0.4
MAX_SPEED_RAD_S = 1.0
READY_TIMEOUT_S = 5.0
GRIPPER_SETTLE_S = 2.0


@dataclass(frozen=True)
class Evaluation:
    calibration: ArmCalibration
    held_out_cm: np.ndarray | None


def speed_in_range(text: str) -> float:
    speed = float(text)
    if not 0.0 < speed <= MAX_SPEED_RAD_S:
        raise argparse.ArgumentTypeError(f"must be above 0 and at most {MAX_SPEED_RAD_S} rad/s")
    return speed


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Compute the camera pose in the robot frame by moving the cube, held in the gripper, through preset arm poses"
    )
    parser.add_argument("--output", type=Path, default=Path("camera_pose.json"))
    parser.add_argument("--poses", type=Path, help="a JSON list of poses of 5 joint angles in radians, replaces the presets")
    parser.add_argument("--speed", type=speed_in_range, default=DEFAULT_SPEED_RAD_S, help="largest joint speed, in rad/s")
    parser.add_argument("--fov-deg", type=float, help="horizontal field of view, replaces the K of /external_cam/camera_info")
    parser.add_argument("--grip", type=float, nargs=2, metavar=("OPEN", "CLOSED"), help="gripper percentages: open it, wait for the cube, close it")
    parser.add_argument("--max-error-cm", type=float, default=MAX_HELD_OUT_CM, help="largest median held-out error accepted")
    parser.add_argument("--force", action="store_true", help="write the pose even when the held-out error is too large")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and stop, the arm does not move")
    return parser.parse_args(argv)


def describe(poses: np.ndarray, speed: float) -> str:
    travel = np.abs(np.diff(poses, axis=0)).max(axis=1)
    lines = [f"{len(poses)} poses, joint angles in radians, largest joint speed {speed} rad/s"]
    lines += [f"  {number}: {np.round(pose, 2).tolist()}" for number, pose in enumerate(poses, start=1)]
    lines.append(f"longest move between poses {travel.max():.2f} rad, about {travel.sum() / speed:.0f} s of motion in all")
    return "\n".join(lines)


def evaluate(samples: Samples, K: np.ndarray) -> Evaluation:
    calibration = solve_from_arm(samples.pixels, samples.gripper_poses, K)
    try:
        held_out_cm = leave_one_out_errors(samples.pixels, samples.gripper_poses, K) * 100.0
    except ValueError:
        held_out_cm = None
    return Evaluation(calibration, held_out_cm)


def refusal(evaluation: Evaluation, max_error_cm: float) -> str | None:
    if evaluation.held_out_cm is None:
        return "too few poses agree to check the result on held-out poses"
    median = float(np.median(evaluation.held_out_cm))
    if median > max_error_cm:
        return f"the median held-out error is {median:.1f} cm, above {max_error_cm:.1f} cm"
    return None


def summary(samples: Samples, evaluation: Evaluation) -> str:
    calibration = evaluation.calibration
    lines = []
    for number, residual, inlier in zip(samples.kept, calibration.residuals_px, calibration.inliers):
        lines.append(f"  pose {number}: {residual:.1f} px{'' if inlier else '  dropped as a wrong detection'}")
    if samples.skipped:
        lines.append(f"  skipped, not detected: {samples.skipped}")
    if evaluation.held_out_cm is not None:
        lines.append(
            f"held-out error: median {np.median(evaluation.held_out_cm):.1f} cm, worst {evaluation.held_out_cm.max():.1f} cm"
        )
    lines.append(f"cube center from the gripper frame: {np.round(calibration.grasp_offset * 1000, 1).tolist()} mm")
    return "\n".join(lines)


def camera_matrix(args, from_topic: np.ndarray | None, frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    if args.fov_deg is not None:
        return intrinsics_from_horizontal_fov(width, height, args.fov_deg)
    if from_topic is not None:
        return from_topic
    print(f"no camera_info received, using a {FALLBACK_FOV_DEG} degree horizontal field of view")
    return intrinsics_from_horizontal_fov(width, height, FALLBACK_FOV_DEG)


def set_gripper(arm: RosArm, percent: float) -> None:
    arm.command_gripper(percent)
    arm.wait(GRIPPER_SETTLE_S)


def run_on_robot(args, poses: np.ndarray) -> None:
    rclpy.init()
    node = rclpy.create_node("calibrate_from_arm")
    try:
        arm, camera = RosArm(node), RosCamera(node)
        if not arm.wait_until_ready(READY_TIMEOUT_S):
            sys.exit("no /joint_states received: is the driver running?")
        frame = camera.frame()
        if frame is None:
            sys.exit("no frame on /external_cam/image_raw: is the camera running?")
        K = camera_matrix(args, read_camera_matrix(node, READY_TIMEOUT_S), frame)
        first_move = np.abs(arm.joint_positions() - poses[0]).max()
        print(f"the first move goes from the current pose to pose 1: largest joint travel {first_move:.2f} rad")
        input("The arm is going to move. Clear the workspace, keep the power switch in reach and press Enter: ")
        if args.grip:
            set_gripper(arm, args.grip[0])
            input("Put the cube between the open jaws and press Enter: ")
            set_gripper(arm, args.grip[1])
        samples = collect_samples(arm, camera, find_cube, poses, args.speed)
    except KeyboardInterrupt:
        sys.exit("stopped, the arm stays where it is")
    except RuntimeError as problem:
        sys.exit(str(problem))
    finally:
        node.destroy_node()
        rclpy.shutdown()
    try:
        evaluation = evaluate(samples, K)
    except ValueError as problem:
        sys.exit(str(problem))
    print(summary(samples, evaluation))
    problem = refusal(evaluation, args.max_error_cm)
    if problem and not args.force:
        sys.exit(f"{problem}: nothing written, use --force to write it anyway")
    save_pose(args.output, evaluation.calibration.robot_from_camera)
    print(f"camera pose saved to {args.output}")


def main(argv=None):
    args = parse_args(argv)
    try:
        poses = load_poses(args.poses) if args.poses else PRESET_POSES
        check_within_limits(poses)
    except ValueError as problem:
        sys.exit(str(problem))
    print(describe(poses, args.speed))
    if not args.dry_run:
        run_on_robot(args, poses)


if __name__ == "__main__":
    main()
