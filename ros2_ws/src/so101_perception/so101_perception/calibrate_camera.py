import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from so101_perception.camera_pose import intrinsics_from_horizontal_fov, solve_camera_pose
from so101_perception.detection import find_cube
from so101_perception.pose_file import save_pose
from so101_perception.projection import CUBE_CENTER_HEIGHT, project_to_plane

DEFAULT_FOV_DEG = 70.0


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Compute the camera pose in the robot frame from the red cube laid at known places")
    parser.add_argument("--sample", action="append", nargs=3, required=True, metavar=("IMAGE", "X", "Y"),
                        help="a camera frame with the cube alone in it, and the cube center in the robot frame, in meters")
    parser.add_argument("--cube-height", type=float, default=CUBE_CENTER_HEIGHT, help="height of the cube center above the table, in meters")
    parser.add_argument("--fov-deg", type=float, default=DEFAULT_FOV_DEG, help="horizontal field of view, estimates K")
    parser.add_argument("--intrinsics", type=float, nargs=4, metavar=("FX", "FY", "CX", "CY"), help="measured K, replaces --fov-deg")
    parser.add_argument("--output", type=Path, default=Path("camera_pose.json"))
    return parser.parse_args(argv)


def intrinsics_of(args, width: int, height: int) -> np.ndarray:
    if args.intrinsics is None:
        return intrinsics_from_horizontal_fov(width, height, args.fov_deg)
    fx, fy, cx, cy = args.intrinsics
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def read_cube_pixel(path: str) -> tuple[tuple[float, float], tuple[int, int]]:
    bgr = cv2.imread(path)
    if bgr is None:
        sys.exit(f"cannot read the image {path}")
    pixel = find_cube(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    if pixel is None:
        sys.exit(f"cube not found in {path}")
    return pixel, bgr.shape[1::-1]


def main(argv=None):
    args = parse_args(argv)
    found = [read_cube_pixel(path) for path, _, _ in args.sample]
    pixels = np.array([pixel for pixel, _ in found])
    places = np.array([[float(x), float(y)] for _, x, y in args.sample])
    width, height = found[0][1]
    K = intrinsics_of(args, width, height)
    try:
        pose, error = solve_camera_pose(pixels, places, args.cube_height, K)
    except ValueError as problem:
        sys.exit(str(problem))
    save_pose(args.output, pose)
    offsets = [np.linalg.norm(project_to_plane(u, v, K, pose, args.cube_height)[:2] - place) for (u, v), place in zip(pixels, places)]
    worst = int(np.argmax(offsets))
    print(f"camera pose saved to {args.output}, reprojection error {error:.2f} px")
    print(f"cube places given against found: worst {offsets[worst] * 100:.1f} cm for {args.sample[worst][0]}")


if __name__ == "__main__":
    main()
