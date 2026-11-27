#!/usr/bin/env python3
import rclpy
import rclpy.node
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import CameraInfo, Image

from so101_perception.camera_pose_source import make_camera_pose_source
from so101_perception.object_publisher import ObjectPublisher
from so101_perception.projection import CUBE_CENTER_HEIGHT, DROP_BOX_BLOB_HEIGHT


class PerceptionNode(rclpy.node.Node):
    def __init__(self):
        super().__init__("perception_node")
        pose_file: str = self.declare_parameter("camera_pose_file", "").value
        cube_height: float = self.declare_parameter("cube_height", CUBE_CENTER_HEIGHT).value
        box_height: float = self.declare_parameter("box_height", DROP_BOX_BLOB_HEIGHT).value
        publisher = ObjectPublisher(self, make_camera_pose_source(self, pose_file), cube_height, box_height)
        self.create_subscription(CameraInfo, "/external_cam/camera_info", publisher.on_camera_info, 1)
        self.create_subscription(Image, "/external_cam/image_raw", publisher.on_image, 1)
        source = pose_file or "the TF world -> external_cam_optical_frame"
        self.get_logger().info(f"Perception node ready, camera pose from {source}.")


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
