import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, JointState


class Probe(Node):
    def __init__(self):
        super().__init__("functional_probe")
        self.joints = None
        self.frame = None
        self.info = None
        self.counts = {"joints": 0, "image": 0}
        self.create_subscription(JointState, "/joint_states", self._on_joints, 10)
        self.create_subscription(Image, "/external_cam/image_raw", self._on_image, 1)
        self.create_subscription(CameraInfo, "/external_cam/camera_info", self._on_info, 10)
        self._command = self.create_publisher(JointState, "/joint_command", 10)

    def _on_joints(self, msg):
        self.joints = dict(zip(msg.name, msg.position))
        self.counts["joints"] += 1

    def _on_image(self, msg):
        if msg.encoding != "rgb8":
            raise RuntimeError(f"camera encoding is {msg.encoding}, expected rgb8")
        self.frame = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3).copy()
        self.counts["image"] += 1

    def _on_info(self, msg):
        self.info = msg

    def spin_for(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    def wait_until(self, condition, timeout, what, alive=lambda: None):
        end = time.monotonic() + timeout
        while not condition():
            alive()
            if time.monotonic() > end:
                raise TimeoutError(f"timeout after {timeout:.0f} s waiting for {what}")
            rclpy.spin_once(self, timeout_sec=0.05)

    def rates(self, seconds):
        before = dict(self.counts)
        self.spin_for(seconds)
        return {key: (self.counts[key] - before[key]) / seconds for key in before}

    def command(self, name, position):
        msg = JointState()
        msg.name = [name]
        msg.position = [float(position)]
        self._command.publish(msg)

    def move_to(self, name, target, tolerance, timeout, alive=lambda: None):
        def reached():
            self.command(name, target)
            self.spin_for(0.1)
            return self.joints is not None and abs(self.joints.get(name, math.inf) - target) <= tolerance

        self.wait_until(reached, timeout, f"{name} to reach {target} rad", alive)
