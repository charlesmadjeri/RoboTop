# Camera pose calibration

The projection of `docs/perception-projection.md` needs the pose of the camera in the robot frame (`world`, the base of
the arm, table at `z = 0`). On the real arm nobody knows it, so it is measured from the red cube. No marker is needed.
In simulation the driver already knows the pose, and no calibration is needed.

- **From the arm (default).** The arm holds the cube and moves it through preset poses. Its kinematics give the 3D
  point, the camera gives the pixel. Nothing is measured by hand, and a moved camera is calibrated again in a minute.
- **With a ruler (fallback).** The cube lies at a few measured places on the table. Use it when the arm run is
  refused (held-out error above 1 cm).

Both methods write the same `camera_pose.json`, and `--profile camx` loads it.

## Calibrate from the arm

`calibrate_from_arm` moves the arm through 12 preset poses (`arm_poses.py`), spread over the workspace with the wrist
turned differently at each. At each pose it waits for the arm to stop, reads the TF `world -> gripper_frame_link`,
takes 3 frames and keeps the median pixel of the cube. It then solves, checks on held-out poses and writes the file.

1. Run the driver, the `robot_state_publisher` of `so101_description` (it gives the TF of the gripper frame) and the
   camera, with the camera mode of the run.
2. `calibrate_from_arm --dry-run` prints the poses, the longest move and the duration. The arm does not move.
3. Put the cube in the gripper. By hand before the run, or let the command do it: `--grip OPEN CLOSED` opens the jaws
   (percent), waits for the cube, and closes them. The values depend on the cube and the jaws, choose them with care.
4. `calibrate_from_arm --fov-deg 75 --output camera_pose.json`. Without `--fov-deg` it uses the K of
   `/external_cam/camera_info`, and 75 degrees if there is none. It prints the largest travel of the first move, and
   waits for Enter: clear the workspace and keep the power switch in reach. The joints move at most 0.4 rad/s
   (`--speed`, at most 1.0), in steps of 20 Hz. The driver must run with `--torque`, or the arm ignores the commands. A pose counts as reached when the joints stop within 0.1 rad of it, so a little sag under gravity is accepted. Otherwise, after 6 s, the command stops and names the joint and its gap. The arm holds its current pose before the jaws move.
5. Read the report: the residual of each pose in pixels, the poses dropped as wrong detections or skipped because the
   cube was not found, the median and worst held-out error, and the offset of the cube center from the gripper frame.
   A pose dropped or skipped is not a failure. Below 9 kept poses the held-out check is impossible and nothing is
   written.
6. The file is written when the median held-out error is at most 1 cm (`--max-error-cm`). Above it, nothing is written
   unless `--force` is given. Then fall back to the ruler method.
7. Moved the camera: run it again. Nothing else changes.

The presets are chosen for the simulated camera view. On a real scene, poses where the cube is out of the picture are
skipped. If too many are, give the command your own poses with `--poses poses.json`, a list of 5 joint angles in
radians (`shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`). Poses closer than 0.1 rad to a joint
limit are refused. The gripper is never in a pose, only in `--grip`.

In the simulator, with the cube moved to the gripper at the 12 poses and exact kinematics
(`test_sim_arm_calibration.py`), all 12 poses are detected, the median held-out error is 0.2 cm, the table points are
located within 2 mm and the camera pose is 5 mm and 0.4 degree off. The real arm has not been run yet, and its
kinematics are less exact. With 6 mm to 1.5 cm of noise on the arm positions the held-out error is 0.5 cm to 2 cm, and
at 3 cm no pose is found.

## From the arm kinematics

The ruler method needs measured places and a new measure each time the camera moves. The arm can replace the ruler: it
knows where its gripper is (`gripper_frame_link` of the URDF, from the joint states). With the cube held in the gripper,
each arm pose gives a 3D point in the robot frame and, from the camera, its pixel. `so101_perception/arm_calibration.py`
solves the camera pose from at least 8 such pairs. It has no ROS import: `calibrate_from_arm` (section above) collects
the pairs.

- The cube center is not the gripper frame: a fixed offset in the gripper frame (3 values) is fitted with the pose. It
  absorbs the grasp and the part of the cube hidden by the jaws. It is only observable when the gripper orientation
  changes between poses, so a wrist that never rotates is refused, unless the offset is fixed (`fit_offset=False`).
- The poses must not lie on one plane: lift the cube to different heights.
- A first solve with a robust loss drops the poses whose detection disagrees (a red decoy, the arm hiding the cube). The
  fit then runs again on the kept poses. If the median residual stays above 30 px, nothing is returned.
- `leave_one_out_errors` fits without each kept pose in turn and returns how far its predicted cube lies from its pixel
  ray, in meters. It needs at least 9 kept poses.

On synthetic poses (12 poses, 1 px detection noise), the cubes on the table are located within 1 mm to 3 mm. Noise on
the arm positions limits it: 3 mm of random noise on each gripper position gives 5 mm on average and 8 mm at worst on
the table points, 1 cm gives about 2 cm. The camera pose itself moves more than the table points (1.7 cm for 3 mm of
noise, because the workspace is small next to the camera distance), and that error partly cancels where the poses
were sampled. Judge a calibration by the located positions, not by the camera pose. The accuracy of the real SO-101
kinematics is not measured yet.

## The cube

The solve does not need the cube size: the offset between the cube center and the gripper frame is fitted. The size
matters in three other places.

- **`cube_height`.** The node projects the cube pixel on the plane at the height of the cube center, half the side:
  0.025 m for a 5 cm cube (the default), 0.03 m for a 6 cm cube. A wrong value shifts the cube on the table by the
  height error times the tangent of the ray angle from the vertical: 5 mm of error gives 5 mm at 45 degrees. Pass
  `cube_height:=0.03` to the node, and `--cube-height 0.03` to `calibrate_camera`.
- **The detected pixel.** It is the centroid of the visible blob, and a tall cube seen from the side shows its side
  faces, so the pixel sits above the cube center. The error grows with the height of the cube.
- **The grasp.** A cube close to the opening of the jaws is hard to put in and out.

A smaller cube is better: a 3 to 4 cm cube cuts the centroid error in proportion and leaves room in the jaws. Keep the
cube low during the run too: the poses calibrate best around where the cube will be used, near the table. They only
need a spread of 1 cm in height, because a calibration from poses on one plane is refused.

## Intrinsics

`K` is needed too. Without a measured one, `--fov-deg` (default 70, the driver default) estimates it from the horizontal
field of view with a centered principal point and square pixels. For the CM678, pass `--fov-deg 75` here and
`camera_fov_deg:=75` to the driver (`--profile camx` sets it): the two values must be the same or the positions are scaled wrongly.
`--intrinsics FX FY CX CY` replaces the estimate with measured values.

## Fallback: with a ruler

The cube lies on the table at a few places whose `x y` is measured from the base of the arm.

### Procedure

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

### Check it

- The reprojection error stays under about 1 px, and the worst place under about 1 cm. A larger worst place names the
  frame whose position was most likely measured wrong: measure it again.
- Lay the cube at a new place that was not used, run the node with `camera_pose_file:=camera_pose.json` and compare
  `/object_position` with a ruler.

### Method

`detection.find_cube` gives the pixel of the cube center in each frame. The centers lie on the plane `z = 0.025` (the
cube center height), so they are a planar set of points of known place: `cv2.solvePnP` (`SOLVEPNP_IPPE`, made for a
planar set, then a Levenberg-Marquardt refinement) gives the pose of the table plane in the camera, and
`robot_from_camera = plane_pose @ inverse(camera_from_plane)`.

### Accuracy

On six placements of the simulated cube, rendered by the simulator, the pose is recovered to 0.6 cm and 0.3 degree
(`test_sim_cube_calibration.py`). A real frame will do worse: the measure of each place by hand is good to a few
millimeters at best.

## Limits

- Calibrating the real `K` and the lens distortion is not included. The estimate from the field of view is as good as the stated angle, and the distortion is ignored. A wide angle lens shows larger errors near the image edges.
- The detection of the cube is tuned on simulated colors. If the cube is not found in a real frame, the HSV ranges of `detection.py` need tuning first.
- The frame comes from a saved image: there is no live capture inside `calibrate_camera`, `grab_frame` does it beforehand.
- The method trusts the given places and the cube height (`--cube-height`, 0.025 m by default, 0.03 m for a 6 cm cube): a cube that does not rest on the table gives a wrong pose.

## Tests

`test_camera_pose.py` recovers the pose from exact pixels (1e-6 m), with noise, with four places, and refuses fewer places
or places on a line. `test_calibrate_camera.py` runs the command on rendered cube views and checks the file, the measured
intrinsics, the report of a wrongly measured place and the error messages. `test_sim_cube_calibration.py` calibrates from
the simulator's own renders. `test_arm_calibration.py` and `test_arm_validation.py` solve from synthetic arm poses: exact
recovery of the pose and the offset, noise, wrong detections dropped, and the refusals (too few poses, one height, a fixed
orientation, pixels that fit no pose). `test_arm_poses.py` checks the presets against the simulator: within the limits,
inside the camera view, clear of the table, no contact on the way between them. `test_arm_session.py` runs the moves and
the sampling on a fake arm, `test_ros_rig.py` runs the ROS side against a fake driver, `test_calibrate_from_arm.py` the
refusals and the dry run, and `test_sim_arm_calibration.py` the whole sequence on the simulated arm.
