# driver_node

`so101_driver/driver_node.py` is the only node that talks to the arm. It runs either the MuJoCo
simulation (`SO101Sim`) or the real arm (`SO101Follower`), chosen by the `use_sim` parameter.
Nothing else in the stack knows which one is running.

`robot_state_publisher` builds the TF of each joint from `/joint_states`. Until the driver publishes
it, RViz shows the arm "in pieces". The first version of the driver publishing it is by Baptiste Rachez.

## Topics

| Topic | Direction | Type | Content |
| --- | --- | --- | --- |
| `/joint_states` | published, 30 Hz | `sensor_msgs/JointState` | 6 joints in radians, in the order of `JOINT_NAMES` |
| `/joint_command` | subscribed | `sensor_msgs/JointState` | position targets by joint name, arm joints in radians, gripper in 0-100 % |
| `/external_cam/image_raw`, `/external_cam/camera_info` | published, 10 Hz | `Image`, `CameraInfo` | sim: provided by `_start_sim_camera`. Real arm: the USB camera read through LeRobot `OpenCVCamera`, published by `frame_publisher.py`, rgb8 |
| `/tf_static` | published once, sim only | `tf2_msgs/TFMessage` | `world` to `external_cam_optical_frame`, the true pose of the simulated camera (`sim_camera_pose.py`). On the real arm the pose comes from `calibrate_camera` instead, see `docs/camera-calibration.md` |

## How it works

```
/joint_command --> _cb_joint_command --> pending action (merged)
                                              |
                  50 Hz _control_step: send_action (if any), then get_observation (once)
                                              |
                                        cached observation
                                              |
                  30 Hz _publish_joint_states --> /joint_states
```

- **One backend read per control tick.** `get_observation()` and `send_action()` each advance the
  MuJoCo physics by 5 steps of 0.002 s. If the read ran in the 30 Hz publisher and the send in the
  50 Hz loop, the simulation speed would depend on whether a command was ever received: 0.3 sim
  second per second idle, 0.8 once commanded (measured), so a trajectory would run 2.7 times faster
  after the first command. With one read per control tick the speed is constant, and the real arm
  reads its serial bus once per tick.
- **Publishing is decoupled from reading.** The 30 Hz publisher only formats the cached observation.
  30 Hz is above the 20 Hz the subject requires.
- **Commands are stored, then applied.** The subscriber callback only records the targets, so it
  returns immediately. The control tick sends them once. Targets persist in the backend (MuJoCo
  `ctrl`, servo goal position), so there is no need to send them again on every tick.
- **Partial commands merge.** A message with only `shoulder_pan` does not make the other joints
  forget their target.
- **Joints are matched by name**, never by position in the message, so a partial or reordered command
  is safe. Unknown names are ignored, and a NaN or infinite position is dropped before it can reach
  the motors.

## Units

- Arm joints are in radians on both backends (the real arm is created with `use_radians=True`).
- Both backends report the gripper as 0-100 % (0 closed, 100 open). The URDF, and therefore RViz, needs
  radians, so `/joint_states` converts it linearly onto the URDF range (-0.174533 to 1.74533 rad).
- `/joint_command` keeps the gripper in 0-100 %, as the subject specifies, and passes it to the backend
  unchanged. The gripper is therefore in radians on `/joint_states` and in percent on `/joint_command`.

The pure conversions live in `so101_driver/joint_codec.py` (no ROS import), which keeps them testable
without a robot and keeps `driver_node.py` under the 200-line file limit.

## Camera

On the real arm, the USB camera is published on `/external_cam/image_raw` at 10 Hz (rgb8), with its
`camera_info`. The simulation publishes the same topics from its own camera, through the same
`FramePublisher` (`so101_driver/frame_publisher.py`).

- `camera_index`: USB camera index of the real arm, default `0`. `-1` disables the camera. Real arm only.
  Get the index from `lerobot-find-cameras opencv`.
- `camera_width`: image width, default `640`.
- `camera_height`: image height, default `480`.
- `camera_fov_deg`: horizontal field of view used to estimate the intrinsics, default `70`.

A USB hiccup skips one frame and does not stop the stream.

## Check it

```bash
pixi run launch                       # terminal 1
ros2 topic hz /joint_states           # terminal 2: about 30 Hz
ros2 topic echo --once /joint_states
ros2 topic pub --once /joint_command sensor_msgs/msg/JointState \
  "{name: ['shoulder_pan', 'gripper'], position: [0.5, 100.0]}"
pixi run rviz2                        # the arm is assembled and follows the command
```

Unit tests of the conversions:

```bash
cd ros2_ws/src/so101_driver
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest test
```

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` avoids a crash of the ROS `launch_testing` pytest plugin with recent pytest.

## Known limits

- Real arm: the torque is disabled in `_connect_arm` on purpose, so commands move nothing until you
  remove that line. The real-arm path was not run when this driver was written.
- Commands are not clipped to the joint limits of the URDF by the driver.
- An exception raised by the backend read (for example a USB hiccup on the real arm) is not caught and
  ends the node. Whether to retry or to stop is still to be decided.
- The simulated camera renders from its own thread while the control timer steps the physics. This is
  the shortcut already documented in `_start_sim_camera`.
- The real arm camera intrinsics are an estimate from the field of view (`camera_fov_deg`), with no
  distortion. A real calibration is needed for accurate pixel to 3D.
