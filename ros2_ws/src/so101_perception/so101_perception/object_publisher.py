from typing import Callable

import numpy as np
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CameraInfo, Image

from so101_perception.camera_pose_source import ROBOT_FRAME
from so101_perception.hold_policy import DEFAULT_HOLD_FRAMES, HoldPolicy
from so101_perception.image_message import rgb_from_image
from so101_perception.pipeline import locate

OBJECT_TOPIC = "/object_position"
DROP_BOX_TOPIC = "/drop_box_position"
POSE_WARNING_PERIOD_S = 5.0


def pose_message(position: np.ndarray, stamp) -> PoseStamped:
    message = PoseStamped()
    message.header.stamp = stamp
    message.header.frame_id = ROBOT_FRAME
    message.pose.position.x, message.pose.position.y, message.pose.position.z = (float(value) for value in position)
    message.pose.orientation.w = 1.0
    return message


class ObjectPublisher:
    def __init__(
        self,
        node,
        camera_pose: Callable[[], np.ndarray | None],
        cube_height: float,
        box_height: float,
        hold_frames: int = DEFAULT_HOLD_FRAMES,
    ):
        self._node = node
        self._camera_pose = camera_pose
        self._cube_height = cube_height
        self._box_height = box_height
        self._K = None
        self._object_publisher = node.create_publisher(PoseStamped, OBJECT_TOPIC, 10)
        self._drop_box_publisher = node.create_publisher(PoseStamped, DROP_BOX_TOPIC, 10)
        self._cube_hold = HoldPolicy[PoseStamped](hold_frames)
        self._box_hold = HoldPolicy[PoseStamped](hold_frames)

    def on_camera_info(self, info: CameraInfo) -> None:
        self._K = np.array(info.k).reshape(3, 3)

    def on_image(self, image: Image) -> None:
        pose = self._camera_pose()
        if pose is None:
            self._node.get_logger().warning(
                "camera pose unknown: nothing published", throttle_duration_sec=POSE_WARNING_PERIOD_S
            )
            return
        if self._K is None:
            return
        cube, box = locate(rgb_from_image(image), self._K, pose, self._cube_height, self._box_height)
        self._publish(self._cube_hold, self._object_publisher, cube, image.header.stamp)
        self._publish(self._box_hold, self._drop_box_publisher, box, image.header.stamp)

    @staticmethod
    def _publish(policy: HoldPolicy[PoseStamped], publisher, position: np.ndarray | None, stamp) -> None:
        reading = policy.update(None if position is None else pose_message(position, stamp))
        if reading.value is not None:
            publisher.publish(reading.value)
