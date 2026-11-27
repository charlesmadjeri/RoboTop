import time

import numpy as np
import rclpy
from sensor_msgs.msg import CameraInfo, Image, JointState
from tf2_ros import Buffer, TransformListener

from so101_perception.arm_poses import ARM_JOINTS
from so101_perception.camera_pose_source import TfCameraPose
from so101_perception.image_message import rgb_from_image

GRIPPER_FRAME = "gripper_frame_link"
GRIPPER_JOINT = "gripper"
IMAGE_TOPIC = "/external_cam/image_raw"
CAMERA_INFO_TOPIC = "/external_cam/camera_info"
JOINT_STATES_TOPIC = "/joint_states"
JOINT_COMMAND_TOPIC = "/joint_command"
SPIN_STEP_S = 0.05


def spin_until(node, done, timeout: float) -> bool:
    end = time.monotonic() + timeout
    while not done() and time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=SPIN_STEP_S)
    return done()


class RosArm:
    def __init__(self, node):
        self._node = node
        self._positions = None
        buffer = Buffer()
        TransformListener(buffer, node)
        self.gripper_pose = TfCameraPose(buffer, camera_frame=GRIPPER_FRAME)
        self._publisher = node.create_publisher(JointState, JOINT_COMMAND_TOPIC, 10)
        node.create_subscription(JointState, JOINT_STATES_TOPIC, self._on_joint_states, 10)

    def _on_joint_states(self, message: JointState) -> None:
        by_name = dict(zip(message.name, message.position))
        if all(name in by_name for name in ARM_JOINTS):
            self._positions = np.array([by_name[name] for name in ARM_JOINTS])

    def wait_until_ready(self, timeout: float) -> bool:
        return spin_until(self._node, lambda: self._positions is not None, timeout)

    def joint_positions(self) -> np.ndarray | None:
        return self._positions

    def command(self, joints: np.ndarray) -> None:
        self._publisher.publish(JointState(name=list(ARM_JOINTS), position=[float(value) for value in joints]))

    def command_gripper(self, percent: float) -> None:
        self._publisher.publish(JointState(name=[GRIPPER_JOINT], position=[float(percent)]))

    def wait(self, seconds: float) -> None:
        spin_until(self._node, lambda: False, seconds)


class RosCamera:
    FRAME_TIMEOUT_S = 3.0

    def __init__(self, node):
        self._node = node
        self._latest = None
        node.create_subscription(Image, IMAGE_TOPIC, self._on_image, 1)

    def _on_image(self, message: Image) -> None:
        self._latest = message

    def frame(self) -> np.ndarray | None:
        self._latest = None
        if not spin_until(self._node, lambda: self._latest is not None, self.FRAME_TIMEOUT_S):
            return None
        return rgb_from_image(self._latest)


def read_camera_matrix(node, timeout: float) -> np.ndarray | None:
    received = []
    subscription = node.create_subscription(CameraInfo, CAMERA_INFO_TOPIC, received.append, 1)
    arrived = spin_until(node, lambda: bool(received), timeout)
    node.destroy_subscription(subscription)
    return np.array(received[0].k).reshape(3, 3) if arrived else None
