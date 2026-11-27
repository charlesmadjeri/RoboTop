# Camera pose calibration

The projection of `docs/perception-projection.md` needs the pose of the camera in the robot frame (`world`, the base of
the arm, table at `z = 0`). On the real arm nobody knows it, so it is measured once from the red cube itself, laid at
a few measured places on the table. No marker is needed. In simulation the driver already knows the pose, and no
calibration is needed.

## Procedure

1. Choose the camera mode first. The UGREEN CM678 has a 75 degree horizontal field of view in its 16:9 modes (1280x720
   or 1920x1080). A 4:3 mode such as 640x480 is often a crop of the sensor, with a narrower angle, so the 75 degrees no
   longer holds there. Use the same mode for the calibration and for the run.
2. Lay the red cube on the table, in view of the camera and alone in the frame (the arm out of view).
3. Measure the `x y` of the cube center in the robot frame, in meters, from the base of the arm. The axes are those of
   the `world` frame in RViz (red x, green y).
4. Save the frame with the camera topic running:
   `grab_frame --output cube1.png`
5. Repeat for at least 4 places, 6 is better. Spread them over the table, over the whole area the arm works in: places
   on a line are refused.
6. Compute the pose:

```
calibrate_camera --fov-deg 75 --output camera_pose.json \
  --sample cube1.png 0.20 0.15 --sample cube2.png 0.25 0.05 --sample cube3.png 0.10 0.20 \
  --sample cube4.png 0.15 0.10 --sample cube5.png 0.30 0.15 --sample cube6.png 0.20 0.25
```

Before the package is built, run the commands as modules:
`PYTHONPATH=ros2_ws/src/so101_perception pixi run python -m so101_perception.calibrate_camera ...`.

The command prints `camera pose saved to camera_pose.json, reprojection error <n> px`, then the place whose given
position is the farthest from where the pose puts the cube. The file holds `{"robot_from_camera": <4x4>}`, the pose of
the optical frame (z forward, y down) in the robot frame, the matrix that `projection.py` takes as `robot_from_camera`.

## Check it

- The reprojection error stays under about 1 px, and the worst place under about 1 cm. A larger worst place names the
  frame whose position was most likely measured wrong: measure it again.
- Lay the cube at a new place that was not used, run the node with `camera_pose_file:=camera_pose.json` and compare
  `/object_position` with a ruler.

## Method

`detection.find_cube` gives the pixel of the cube center in each frame. The centers lie on the plane `z = 0.025` (the
cube center height), so they are a planar set of points of known place: `cv2.solvePnP` (`SOLVEPNP_IPPE`, made for a
planar set, then a Levenberg-Marquardt refinement) gives the pose of the table plane in the camera, and
`robot_from_camera = plane_pose @ inverse(camera_from_plane)`.

## Accuracy

On six placements of the simulated cube, rendered by the simulator, the pose is recovered to 0.6 cm and 0.3 degree
(`test_sim_cube_calibration.py`). A real frame will do worse: the measure of each place by hand is good to a few
millimeters at best.

## Intrinsics

`K` is needed too. Without a measured one, `--fov-deg` (default 70, the driver default) estimates it from the horizontal
field of view with a centered principal point and square pixels. For the CM678, pass `--fov-deg 75` here and
`camera_fov_deg:=75` to the driver (`--profile camx` sets it): the two values must be the same or the positions are scaled wrongly.
`--intrinsics FX FY CX CY` replaces the estimate with measured values.

## Limits

- Calibrating the real `K` and the lens distortion is not included. The estimate from the field of view is as good as the stated angle, and the distortion is ignored. A wide angle lens shows larger errors near the image edges.
- The detection of the cube is tuned on simulated colors. If the cube is not found in a real frame, the HSV ranges of `detection.py` need tuning first.
- The frame comes from a saved image: there is no live capture inside `calibrate_camera`, `grab_frame` does it beforehand.
- The method trusts the given places and the cube height (`--cube-height`, 0.025 m): a cube that does not rest on the table gives a wrong pose.

## Tests

`test_camera_pose.py` recovers the pose from exact pixels (1e-6 m), with noise, with four places, and refuses fewer places
or places on a line. `test_calibrate_camera.py` runs the command on rendered cube views and checks the file, the measured
intrinsics, the report of a wrongly measured place and the error messages. `test_sim_cube_calibration.py` calibrates from
the simulator's own renders.
