# From a detected pixel to a 3D point

`so101_perception/projection.py` turns the pixel `(u, v)` of `detection.py` into a point in the robot frame
(`world`, the base of the arm, table at `z = 0`). It has no ROS import: `K` and the camera pose are arguments.

## Method

1. Back-project the pixel to a ray in the optical frame: `d = K^-1 (u, v, 1)`, normalized.
2. Rotate it into the robot frame with the camera pose `T` (optical frame to robot frame): `d_robot = R d`, from the camera center `o = t`.
3. Intersect with the horizontal plane `z = height`: `p = o + s d_robot`, `s = (height - o_z) / d_robot_z`.
4. Return `None` when the ray is parallel to the plane or when `s <= 0` (the plane is behind the camera).

## Assumption

A single camera gives a ray, not a point. The assumption of our own: **the object rests on the table**, so its
height is known and the ray is cut by a horizontal plane. A known object size would also work, but it needs
a precise blob size, which the tilted view and the holes of the cube do not give.

The plane is not the table (`z = 0`). The detector returns the centroid of the blob the camera sees, which sits
at the height of the visible mass, not at the base.

| Target | Plane height | Why |
|---|---|---|
| Cube | `CUBE_CENTER_HEIGHT` = 0.025 m | Center of the 5 cm cube |
| Drop box | `DROP_BOX_BLOB_HEIGHT` = 0.025 m | Mid height of the 5 cm walls, which dominate the blob |

Measured on the simulated scenes (distance on the table between the projected detection and the ground truth):

| Plane height | Cube | Drop box |
|---|---|---|
| 0.000 m (table) | 2.0 to 2.2 cm | 1.7 to 2.0 cm |
| 0.015 m | 0.9 to 1.0 cm | 0.4 to 0.8 cm |
| 0.025 m | 0.2 to 0.3 cm | 0.1 to 0.5 cm |

Projecting on the table would push every detection about 2 cm farther from the camera along its ray.
The drop box is a position on the table: only the `x, y` of its point matter, its base is at `z = 0`.

## Limits

- The camera pose comes from the caller. The error above is measured with the exact simulated pose; any pose error moves the point by about the same distance.
- `K` of the real camera is estimated from a field of view (see `docs/DRIVER.md`), so it adds its own error on the real arm.
- The heights are tuned on the simulated render. Real lighting and a real box change the blob, so they must be checked on a real frame.
- An object lifted off the table (in the gripper) is no longer on its plane, and the point is wrong.

## Tests

`test/test_projection.py` checks the geometry on a known camera pose (principal point, pixel offset, round trip of a projected point, rejected rays)
and the detections of the saved renders against the ground truth: the true pixel of `expected.json` projected on the same plane for the cube,
the known base `(0.05, 0.25)` for the drop box. `test/data/camera.json` holds `K` and the optical pose of the simulated camera
(`get_camera_extrinsics()` times `diag(1, -1, -1, 1)`, MuJoCo cameras look along -z with y up).
