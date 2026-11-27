import threading

import numpy as np
import pytest
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import CameraInfo, Image, JointState
from tf2_ros import StaticTransformBroadcaster

from so101_perception.arm_poses import ARM_JOINTS
from so101_perception.ros_rig import GRIPPER_FRAME, RosArm, RosCamera, read_camera_matrix, spin_until

DOMAIN_ID = 97
JOINT_NAMES = [*ARM_JOINTS, "gripper"]
START = [0.1, 0.2, 0.3, 0.4, 0.5]
GRIPPER_AT = (0.2, 0.1, 0.15)
INFO_K = [900.0, 0.0, 640.0, 0.0, 900.0, 360.0, 0.0, 0.0, 1.0]


class FakeDriver(rclpy.node.Node):
    def __init__(self):
        super().__init__("fake_driver")
        self.positions = [*START, 0.0]
        self.received = []
        self.frame_number = 0
        self._states = self.create_publisher(JointState, "/joint_states", 10)
        self._images = self.create_publisher(Image, "/external_cam/image_raw", 1)
        self._info = self.create_publisher(CameraInfo, "/external_cam/camera_info", 1)
        self.create_subscription(JointState, "/joint_command", self._on_command, 10)
        self.create_timer(0.02, self._publish)
        transform = TransformStamped()
        transform.header.frame_id, transform.child_frame_id = "world", GRIPPER_FRAME
        transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z = GRIPPER_AT
        transform.transform.rotation.w = 1.0
        self._tf = StaticTransformBroadcaster(self)
        self._tf.sendTransform(transform)

    def _on_command(self, message):
        self.received.append(message)
        for name, position in zip(message.name, message.position):
            self.positions[JOINT_NAMES.index(name)] = position

    def _publish(self):
        self._states.publish(JointState(name=JOINT_NAMES, position=list(self.positions)))
        self.frame_number += 1
        image = Image(height=2, width=3, encoding="rgb8", step=9)
        image.data = bytes([self.frame_number % 256]) * 18
        self._images.publish(image)
        self._info.publish(CameraInfo(k=INFO_K))


@pytest.fixture(scope="module")
def rig():
    rclpy.init(domain_id=DOMAIN_ID)
    driver = FakeDriver()
    executor = SingleThreadedExecutor()
    executor.add_node(driver)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    node = rclpy.create_node("rig_under_test")
    yield node, driver
    executor.shutdown()
    thread.join(timeout=2)
    node.destroy_node()
    driver.destroy_node()
    rclpy.shutdown()


def test_the_arm_reads_the_five_arm_joints_from_the_joint_states(rig):
    node, _ = rig
    arm = RosArm(node)
    assert arm.joint_positions() is None
    assert arm.wait_until_ready(5.0)
    assert np.allclose(arm.joint_positions(), START)


def test_a_command_names_only_the_arm_joints_and_comes_back_in_the_joint_states(rig):
    node, driver = rig
    arm = RosArm(node)
    arm.wait_until_ready(5.0)
    target = np.array([0.5, -0.2, 0.1, 0.0, 0.3])
    arm.command(target)
    assert spin_until(node, lambda: np.allclose(arm.joint_positions(), target), 5.0)
    assert driver.received[-1].name == list(ARM_JOINTS)


def test_the_gripper_is_commanded_in_percent_by_name_alone(rig):
    node, driver = rig
    arm = RosArm(node)
    arm.command_gripper(35.0)
    assert spin_until(node, lambda: any(message.name == ["gripper"] for message in driver.received), 5.0)
    assert [message for message in driver.received if message.name == ["gripper"]][-1].position.tolist() == [35.0]


def test_the_gripper_pose_is_the_transform_from_the_robot_frame_to_the_gripper_frame(rig):
    node, _ = rig
    arm = RosArm(node)
    assert spin_until(node, lambda: arm.gripper_pose() is not None, 5.0)
    assert np.allclose(arm.gripper_pose()[:3, 3], GRIPPER_AT)


def test_each_camera_frame_is_a_fresh_rgb_image(rig):
    node, _ = rig
    camera = RosCamera(node)
    first, second = camera.frame(), camera.frame()
    assert first.shape == (2, 3, 3) and second.shape == (2, 3, 3)
    assert first[0, 0, 0] != second[0, 0, 0]


def test_the_camera_matrix_is_read_from_the_camera_info(rig):
    node, _ = rig
    assert np.allclose(read_camera_matrix(node, 5.0), np.array(INFO_K).reshape(3, 3))
