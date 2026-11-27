# Camera pose calibration

The projection of `docs/perception-projection.md` needs the pose of the camera in the robot frame (`world`, the base of
the arm, table at `z = 0`). On the real arm nobody knows it, so it is measured once from one ArUco marker lying flat
at a known place. In simulation the driver already knows it, and no calibration is needed.

## Procedure

1. Print an ArUco marker of the `DICT_4X4_50` dictionary (id 0 by default) and measure the side of its black square.
2. Lay it flat on the table, in view of the camera, and measure the robot-frame position of its center `x y z` in meters.
   Put the top edge of the marker toward `+y` for a yaw of 0, and turn it by the yaw angle (degrees, counterclockwise seen from above) if it is turned.
3. Save one camera frame, with the camera topic running:
   `PYTHONPATH=ros2_ws/src/so101_perception pixi run python -m so101_perception.grab_frame --output frame.png`.
4. Compute the pose:

```
PYTHONPATH=ros2_ws/src/so101_perception pixi run python -m so101_perception.calibrate_camera \
  --image frame.png --marker-size 0.08 --marker-xyz 0.25 0.05 0.0 --marker-yaw 0 --output camera_pose.json
```

The package is skipped by colcon (`COLCON_IGNORE`) until the node is wired, so the commands run as modules. Once it is built,
`ros2 run so101_perception calibrate_camera` and `grab_frame` do the same.

The command prints `camera pose saved to camera_pose.json, reprojection error <n> px`. The file holds
`{"robot_from_camera": <4x4>}`, the pose of the optical frame (z forward, y down) in the robot frame, the matrix that
`projection.py` takes as `robot_from_camera`. `so101_perception.pose_file.load_pose` reads it back.

## Check it

- The reprojection error stays under about 1 px. A larger value means a wrong marker size, a blurred frame or a wrong `K`.
- Move the marker, calibrate again and compare the two files: the camera center must not move by more than a few millimeters.
- Project the marker center pixel on the plane `z = 0` with `project_to_plane`: it must give back the measured `x y`.

## Method

`marker_detection.py` finds the four corners with `cv2.aruco` (sub pixel refinement on). `camera_pose.py` solves the pose of
the marker in the camera with `cv2.solvePnP` (`SOLVEPNP_IPPE_SQUARE`, made for a flat square), then composes
`robot_from_camera = robot_from_marker @ inverse(camera_from_marker)`.

## Intrinsics

`K` is needed too. Without a measured one, `--fov-deg` (default 70, the driver default) estimates it from the horizontal
field of view with a centered principal point and square pixels. `--intrinsics FX FY CX CY` replaces it with measured values.

## Limits

- Calibrating the real `K` and the lens distortion is not included. The estimate from the field of view is as good as the stated angle, and the distortion is ignored.
- One marker gives a pose accurate to about 5 mm and 0.4 degree on a synthetic 640x480 view (measured in the tests). A real frame will do worse: lighting, print size and blur are not modeled.
- The frame comes from a saved image: there is no live capture inside `calibrate_camera`, `grab_frame` does it beforehand.
- The marker must lie flat: the method trusts the given `z` and yaw.

## Tests

`test_camera_pose.py` recovers the pose from exact corners (1e-6 m). `test_calibrate_camera.py` draws a marker through a known
pose, runs detection and solve, and checks the result within 1 cm and 1 degree, plus the command, its error messages and the
file round trip. `test_grab_frame.py` checks the conversion of an image message.
