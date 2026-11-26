#!/usr/bin/env python3
"""perception_node.py — golden ball detection and 3D registration.

TODO:
  - Subscribe to /external_cam/image_raw and /external_cam/camera_info
  - Publish /object_position (5-10 Hz)
  - Get the cube and drop box pixels (u, v) from so101_perception.detection
  - Project pixel -> 3D via K^{-1} and the camera->world TF
  - Intersect with the z=0 plane (table) to estimate the depth
  - Get cam_K / cam_T from the driver (params or CameraInfo)
"""
import threading

import cv2
import rclpy
import rclpy.node
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rcl_interfaces.srv import GetParameters
from sensor_msgs.msg import CameraInfo, Image


class PerceptionNode(rclpy.node.Node):
    """Detects the golden ball and publishes its 3D position."""

    def __init__(self):
        raise NotImplementedError("TO DO")

    def _cb_camera_info(self, msg):
        raise NotImplementedError("TO DO")

    def _cb_image(self, msg):
        raise NotImplementedError("TO DO")

    def _detect_object(self, cv_image):
        raise NotImplementedError("TO DO")

    def _project_to_3d(self, u, v):
        raise NotImplementedError("TO DO")

    def _publish_result(self):
        raise NotImplementedError("TO DO")

    def _fetch_cam_params(self):
        raise NotImplementedError("TO DO")


def main():
    raise NotImplementedError("TO DO")
