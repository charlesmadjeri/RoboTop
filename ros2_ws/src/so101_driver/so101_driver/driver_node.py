#!/usr/bin/env python3
"""driver_node.py - ROS2 driver of the SO-ARM101.

The only node that talks to the backend: SO101Sim (MuJoCo) or SO101Follower
(real arm), chosen by the `use_sim` parameter. The rest of the stack never
knows which one runs.

Todo List:
  - [DONE] Publish /joint_states in RADIANS (>= 20 Hz)
  - [DONE] Subscribe to /joint_command (radians, gripper 0-100 %)
  - [DONE] Real arm: publish the USB camera on /external_cam/image_raw
    (the simulated camera is provided: _start_sim_camera)
"""
import os
from pathlib import Path

import rclpy
import rclpy.node
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import JointState
from so101_sim import SO101Sim
from tf2_ros import StaticTransformBroadcaster

from so101_driver.camera_info import build_camera_info, pinhole_intrinsics
from so101_driver.frame_publisher import FramePublisher
from so101_driver.joint_codec import JOINT_NAMES, command_to_action, observation_to_positions
from so101_driver.sim_camera_pose import camera_transform
from so101_driver.usb_camera import open_usb_camera

CONTROL_HZ = 50.0
JOINT_STATE_HZ = 30.0
CAMERA_HZ = 10.0
CAMERA_FRAME = "external_cam_optical_frame"
ROBOT_FRAME = "world"


def _calibration_file() -> Path:
    """Calibration file of the real arm, from SO101_CALIBRATION_FILE (provided)."""
    path = os.environ.get("SO101_CALIBRATION_FILE")
    if not path:
        raise RuntimeError("SO101_CALIBRATION_FILE is not set: point it to the calibration file of your arm (see README, Real arm)")
    return Path(path).expanduser()


class DriverNode(rclpy.node.Node):
    """Driver SO-ARM101: /joint_command -> backend -> /joint_states."""

    def __init__(self):
        super().__init__("so101_driver")
        self._use_sim: bool = self.declare_parameter("use_sim", True).value
        port: str = self.declare_parameter("port", "").value
        self.get_logger().info("Mode: simulation" if self._use_sim else f"Mode: hardware (port={port})")

        self._robot = self._connect_arm(port)
        self.get_logger().info(f"Robot connected: {type(self._robot).__name__}")

        self._pending_action: dict[str, float] = {}
        self._observation = self._robot.get_observation()

        self._joint_state_pub = self.create_publisher(JointState, "/joint_states", 10)
        self._joint_cmd_sub = self.create_subscription(JointState, "/joint_command", self._cb_joint_command, 10)

        self.create_timer(1.0 / CONTROL_HZ, self._control_step)
        self.create_timer(1.0 / JOINT_STATE_HZ, self._publish_joint_states)

        if self._use_sim:
            self._start_sim_camera()
        else:
            self._start_usb_camera()

        self.get_logger().info("Driver node ready.")

    def _cb_joint_command(self, msg: JointState):
        self._pending_action.update(command_to_action(list(msg.name), list(msg.position)))

    def _control_step(self):
        if self._pending_action:
            self._robot.send_action(self._pending_action)
            self._pending_action = {}
        self._observation = self._robot.get_observation()

    def _publish_joint_states(self):
        msg = JointState(name=list(JOINT_NAMES), position=observation_to_positions(self._observation))
        msg.header.stamp = self.get_clock().now().to_msg()
        self._joint_state_pub.publish(msg)

    def _connect_arm(self, port: str) -> SO101Sim | SO101Follower:
        """Creates and connects the backend chosen by use_sim (provided)."""
        if self._use_sim:
            robot = SO101Sim()
            robot.connect()
            return robot

        if not port:
            raise RuntimeError("No serial port: pass port:=<port> (pixi run launch --backend real --port <port>)")
        calibration_file = _calibration_file()
        if not calibration_file.is_file():
            raise RuntimeError(f"No calibration file at {calibration_file}: calibrate it with `lerobot-calibrate` first")
        config = SO101FollowerConfig(
            port=port,
            id=calibration_file.stem,
            calibration_dir=calibration_file.parent,
            use_radians=True,
        )
        robot = SO101Follower(config)
        robot.connect(calibrate=False)
        if not robot.is_calibrated:
            robot.disconnect()
            raise RuntimeError(
                f"{calibration_file} does not match the motors: wrong file for this arm? "
                "Re-run `lerobot-calibrate`."
            )
        robot.bus.disable_torque()  # Comment out this line to control the robot.
        self.get_logger().info(
            "Torque disabled: arm can be moved freely by hand. Remove this part to control the arm."
        )
        return robot

    def _start_sim_camera(self):
        """Publishes the MuJoCo camera at CAMERA_HZ (provided): /external_cam/image_raw + camera_info."""
        camera_info = build_camera_info(
            self._robot.get_camera_intrinsics().flatten(),
            self._robot.camera_width,
            self._robot.camera_height,
            CAMERA_FRAME,
        )
        # shortcut: renders while your timers step the physics; fine for a camera image.
        self._camera_publisher = FramePublisher(self, self._robot.render_camera, camera_info, CAMERA_HZ)
        self._camera_publisher.start()
        self._camera_pose_broadcaster = StaticTransformBroadcaster(self)
        stamp = self.get_clock().now().to_msg()
        self._camera_pose_broadcaster.sendTransform(
            camera_transform(self._robot.get_camera_extrinsics(), ROBOT_FRAME, CAMERA_FRAME, stamp)
        )

    def _start_usb_camera(self):
        """Publishes the USB camera of the real arm on the same topics as the simulated one."""
        index: int = self.declare_parameter("camera_index", 0).value
        width: int = self.declare_parameter("camera_width", 640).value
        height: int = self.declare_parameter("camera_height", 480).value
        fov_deg: float = self.declare_parameter("camera_fov_deg", 70.0).value
        if index < 0:
            self.get_logger().info("USB camera disabled (camera_index < 0)")
            return
        try:
            camera = open_usb_camera(index, width, height)
        except Exception:
            self._robot.disconnect()
            raise
        camera_info = build_camera_info(pinhole_intrinsics(width, height, fov_deg), width, height, CAMERA_FRAME)
        self._camera_publisher = FramePublisher(self, camera.read_latest, camera_info, CAMERA_HZ, on_stop=camera.disconnect)
        self._camera_publisher.start()
        self.get_logger().info(f"USB camera {index} published ({width}x{height}, fov {fov_deg} deg)")

    def destroy_node(self):
        """Disconnects the backend (provided). On the real arm this disables the torque."""
        if getattr(self, "_camera_publisher", None) is not None:
            self._camera_publisher.stop()
        if self._robot.is_connected:
            self._robot.disconnect()
            self.get_logger().info("Robot disconnected.")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DriverNode()
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
