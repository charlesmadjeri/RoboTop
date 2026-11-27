# perception_node

Finds the cube and the drop box in the camera image and publishes where they are in the robot frame (`world`).
Nothing in the node touches the simulator: it reads the camera topics only, so the sim and the real arm run the same code.

## Topics

| Topic | Direction | Type | Content |
| --- | --- | --- | --- |
| `/external_cam/image_raw` | subscribed | `sensor_msgs/Image` | rgb8 frame, 10 Hz |
| `/external_cam/camera_info` | subscribed | `sensor_msgs/CameraInfo` | `K` of the camera |
| `/object_position` | published, per frame | `geometry_msgs/PoseStamped` | cube center, frame `world`, stamp of the image, `z` = cube plane height |
| `/drop_box_position` | published, per frame | `geometry_msgs/PoseStamped` | drop box base, frame `world`, stamp of the image, `z = 0` |

A target that is not seen is not published: the last message is never repeated. The orientation is the identity, only
the position is measured. `so101_description` already moves the drop box marker of RViz from `/drop_box_position`.

## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `camera_pose_file` | empty | JSON file of `calibrate_camera`. Empty reads the TF `world` to `external_cam_optical_frame` |
| `cube_height` | 0.025 | Plane height used to project the cube pixel, see `docs/perception-projection.md` |
| `box_height` | 0.025 | Plane height used to project the drop box pixel |

Without a pose (file missing at start stops the node, TF not yet there only warns every 5 s) nothing is published.
In simulation the driver publishes the TF, so the default works. On the real arm, calibrate first and pass the file:

```
ros2 launch so101_bringup so101.launch.py use_sim:=false port:=<port> camera_pose_file:=camera_pose.json
```

`pixi run launch` does not forward `camera_pose_file`, so call `ros2 launch` after `pixi run build` (inside `pixi shell`).

## How it works

1. `ObjectPublisher` converts each image to rgb8 pixels (`image_message.py`) and takes `K` from `camera_info`.
2. `pipeline.locate` runs `detection.find_cube` and `find_drop_box`, then `projection.project_to_plane` on each pixel.
3. The drop box point is moved to `z = 0`: the plane height only serves to project its blob, its base is on the table.
4. The camera pose comes from `camera_pose_source.py`: the file when given, else the TF buffer.

## Check it

In simulation:

```
pixi run launch
ros2 topic echo --once /drop_box_position    # about x 0.05, y 0.25, z 0
ros2 topic echo --once /object_position      # the cube, it moves at every start (random spawn)
pixi run rviz2                               # the two Pose displays follow the targets
```

Measured on 30 random cube placements of the sim (render, detect, project, compared with `get_object_position()`):
cube error 0.33 cm on average, 0.88 cm at most, no miss. Drop box error 0.12 cm.

## Known limits

- The HSV thresholds of `detection.py` are tuned on the simulated colors only. A real camera frame needs its own tuning.
- The real `K` is an estimate from the field of view, and the pose from one marker is a few millimeters accurate at best.
- The cube must rest on the table: a lifted cube is projected too far from the camera.
- A frame where the arm hides the cube publishes nothing for the cube.
