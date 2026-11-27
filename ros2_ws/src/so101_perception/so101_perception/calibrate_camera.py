import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from so101_perception.camera_pose import intrinsics_from_horizontal_fov, robot_from_marker, solve_camera_pose
from so101_perception.marker_detection import find_marker_corners
from so101_perception.pose_file import save_pose

DEFAULT_FOV_DEG = 70.0


class MarkerNotFound(ValueError):
    pass


def calibrate(rgb: np.ndarray, K: np.ndarray, marker_id: int, size: float, marker_pose: np.ndarray):
    corners = find_marker_corners(rgb, marker_id)
    if corners is None:
        raise MarkerNotFound(f"marker {marker_id} not found in the image")
    return solve_camera_pose(corners, size, marker_pose, K)


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Compute the camera pose in the robot frame from one ArUco marker")
    parser.add_argument("--image", type=Path, required=True, help="camera frame showing the marker")
    parser.add_argument("--marker-size", type=float, required=True, help="side of the black square, in meters")
    parser.add_argument("--marker-xyz", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"),
                        help="marker center in the robot frame, in meters")
    parser.add_argument("--marker-yaw", type=float, default=0.0, help="degrees, 0 when the top of the marker points to +y")
    parser.add_argument("--marker-id", type=int, default=0)
    parser.add_argument("--fov-deg", type=float, default=DEFAULT_FOV_DEG, help="horizontal field of view, estimates K")
    parser.add_argument("--intrinsics", type=float, nargs=4, metavar=("FX", "FY", "CX", "CY"), help="measured K, replaces --fov-deg")
    parser.add_argument("--output", type=Path, default=Path("camera_pose.json"))
    return parser.parse_args(argv)


def intrinsics_of(args, width: int, height: int) -> np.ndarray:
    if args.intrinsics is None:
        return intrinsics_from_horizontal_fov(width, height, args.fov_deg)
    fx, fy, cx, cy = args.intrinsics
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def main(argv=None):
    args = parse_args(argv)
    bgr = cv2.imread(str(args.image))
    if bgr is None:
        sys.exit(f"cannot read the image {args.image}")
    height, width = bgr.shape[:2]
    marker_pose = robot_from_marker(tuple(args.marker_xyz), args.marker_yaw)
    try:
        pose, error = calibrate(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), intrinsics_of(args, width, height),
                                args.marker_id, args.marker_size, marker_pose)
    except MarkerNotFound as problem:
        sys.exit(str(problem))
    save_pose(args.output, pose)
    print(f"camera pose saved to {args.output}, reprojection error {error:.2f} px")


if __name__ == "__main__":
    main()
