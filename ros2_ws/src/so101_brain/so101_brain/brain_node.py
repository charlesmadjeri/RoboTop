#!/usr/bin/env python3
from pathlib import Path

import rclpy
import rclpy.node
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped, Vector3Stamped
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import JointState

from so101_brain.arm_kinematics import ArmKinematics
from so101_brain.joint_interpolator import JointInterpolator

ARM_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]
JOINT_NAMES = ARM_JOINT_NAMES + ["gripper"]
BASE_FRAME = "base_link"
REACH_TOLERANCE_M = 0.01
CONTROL_PERIOD_S = 0.02


def default_urdf_path() -> str:
    share = Path(get_package_share_directory("so101_description"))
    return str(share / "urdf" / "so101.urdf")


class BrainNode(rclpy.node.Node):

    def __init__(self):
        super().__init__("so101_brain")
        urdf_path = self.declare_parameter("urdf_path", "").value or default_urdf_path()
        self._step_size_m = self.declare_parameter("step_size_m", 0.01).value
        self._max_vector = self.declare_parameter("max_vector_component", 3.0).value
        max_joint_step = self.declare_parameter("max_joint_step_rad", 0.015).value

        self._kinematics = ArmKinematics(urdf_path)
        self._interpolator = JointInterpolator(len(ARM_JOINT_NAMES), max_joint_step)
        self._current_q = [0.0] * len(ARM_JOINT_NAMES)
        self._teleop_active = False

        self._pose_pub = self.create_publisher(PoseStamped, "/end_effector_pose", 10)
        self._command_pub = self.create_publisher(JointState, "/joint_command", 10)
        self.create_subscription(JointState, "/joint_states", self._cb_joint_states, 10)
        self.create_subscription(Vector3Stamped, "/instruction_vector", self._cb_instruction_vector, 1)
        self.create_subscription(PoseStamped, "/go_to_target", self._cb_go_to_target, 1)
        self.create_timer(CONTROL_PERIOD_S, self._control_loop)

        self.get_logger().info("Brain node ready.")

    def _cb_joint_states(self, msg: JointState):
        positions = dict(zip(msg.name, msg.position))
        if not all(name in positions for name in ARM_JOINT_NAMES):
            return
        self._current_q = [float(positions[name]) for name in ARM_JOINT_NAMES]
        if not self._interpolator.ready:
            self._interpolator.start(self._current_q)
        self._publish_end_effector_pose()

    def _publish_end_effector_pose(self):
        position, quat = self._kinematics.pose(self._current_q)
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = BASE_FRAME
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = (float(v) for v in position)
        msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = (
            float(v) for v in quat
        )
        self._pose_pub.publish(msg)

    def _send_joint_command(self, q: list[float], gripper_pct: float | None = None):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINT_NAMES if gripper_pct is not None else ARM_JOINT_NAMES
        msg.position = list(q) + ([float(gripper_pct)] if gripper_pct is not None else [])
        self._command_pub.publish(msg)

    def _clamp(self, value: float) -> float:
        return max(-self._max_vector, min(self._max_vector, float(value)))

    def _cb_instruction_vector(self, msg: Vector3Stamped):
        if not self._interpolator.ready:
            return
        vector = [self._clamp(msg.vector.x), self._clamp(msg.vector.y), self._clamp(msg.vector.z)]
        if not any(vector):
            if self._teleop_active:
                self._teleop_active = False
                self._interpolator.hold()
            return
        self._teleop_active = True
        commanded = self._interpolator.commanded
        origin = self._kinematics.position(commanded)
        target = [float(origin[i] + vector[i] * self._step_size_m) for i in range(3)]
        try:
            q_sol = self._kinematics.solve(target, commanded)
        except ValueError as error:
            self.get_logger().warn(f"IK skipped: {error}")
            return
        if self._kinematics.error(q_sol, target) < REACH_TOLERANCE_M:
            self._interpolator.target = q_sol

    def _cb_go_to_target(self, msg: PoseStamped) -> bool:
        frame = msg.header.frame_id
        if frame not in ("", BASE_FRAME):
            self.get_logger().warn(f"go_to_target refused: header.frame_id must be {BASE_FRAME}, got {frame}")
            return False
        p = msg.pose.position
        target = [p.x, p.y, p.z]
        try:
            q_sol = self._kinematics.solve(target, self._current_q)
        except ValueError as error:
            self.get_logger().warn(f"go_to_target refused: IK failed: {error}")
            return False
        error_cm = self._kinematics.error(q_sol, target) * 100.0
        if error_cm >= REACH_TOLERANCE_M * 100.0:
            self.get_logger().warn(f"go_to_target refused: target unreachable (IK error {error_cm:.2f} cm)")
            return False
        self._interpolator.target = q_sol
        self.get_logger().info(f"go_to_target accepted (IK error {error_cm:.2f} cm)")
        return True

    def _control_loop(self):
        if not self._interpolator.ready:
            return
        q = self._interpolator.step()
        if q is not None:
            self._send_joint_command(q)


def main(args=None):
    rclpy.init(args=args)
    node = BrainNode()
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
