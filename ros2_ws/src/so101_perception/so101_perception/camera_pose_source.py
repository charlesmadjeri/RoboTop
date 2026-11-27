from typing import Callable

import numpy as np
from geometry_msgs.msg import Transform
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener

from so101_perception.pose_file import load_pose

ROBOT_FRAME = "world"
CAMERA_FRAME = "external_cam_optical_frame"


def matrix_from_transform(transform: Transform) -> np.ndarray:
    x, y, z, w = (getattr(transform.rotation, axis) for axis in "xyzw")
    pose = np.eye(4)
    pose[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    pose[:3, 3] = [transform.translation.x, transform.translation.y, transform.translation.z]
    return pose


class TfCameraPose:
    def __init__(self, buffer: Buffer, robot_frame: str = ROBOT_FRAME, camera_frame: str = CAMERA_FRAME):
        self._buffer = buffer
        self._robot_frame = robot_frame
        self._camera_frame = camera_frame

    def __call__(self) -> np.ndarray | None:
        try:
            stamped = self._buffer.lookup_transform(self._robot_frame, self._camera_frame, Time())
        except TransformException:
            return None
        return matrix_from_transform(stamped.transform)


def make_camera_pose_source(node, pose_file: str) -> Callable[[], np.ndarray | None]:
    if pose_file:
        pose = load_pose(pose_file)
        return lambda: pose
    buffer = Buffer()
    TransformListener(buffer, node)
    return TfCameraPose(buffer)
