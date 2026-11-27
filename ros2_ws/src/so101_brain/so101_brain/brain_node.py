#!/usr/bin/env python3
"""brain_node.py - intelligence of the arm: kinematics, then pick & place.

Never talks to the backend: reads /joint_states, commands through /joint_command.
"""
from pathlib import Path
import numpy as np
import rclpy
import rclpy.node
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped, Vector3Stamped
from ikpy.chain import Chain
from rclpy.executors import ExternalShutdownException
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState
from std_msgs.msg import Int32

try:
    from so101_interfaces.srv import GoToTarget
    HAS_GOTO_SRV = True
except ImportError:
    GoToTarget = None
    HAS_GOTO_SRV = False

ARM_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]
JOINT_NAMES = ARM_JOINT_NAMES + ["gripper"]


class BrainNode(rclpy.node.Node):
    """Brain of the arm: kinematics, then pick & place."""

    def __init__(self):
        super().__init__("so101_brain")

        # 1. Chargement de la chaîne cinématique URDF (7 links : OriginLink + 5 actifs + gripper_frame_joint)
        urdf_path = Path(get_package_share_directory("so101_description")) / "urdf" / "so101.urdf"
        self._chain = Chain.from_urdf_file(
            str(urdf_path),
            base_elements=["base_link"],
            active_links_mask=[False, True, True, True, True, True, False],
        )

        # 2. Paramètres dynamiques (modifiables à chaud via ros2 param set)
        self.declare_parameter("max_speed_rad_s", 2.0)  # Vitesse max moteurs (rad/s)
        self.declare_parameter("step_size_m", 0.002)    # 2 mm par tick -> 4 cm/s normal, 12 cm/s boost à 20 Hz

        # 3. État des articulations et variables d'interpolation
        self._current_q = [0.0] * 5
        self._target_q = [0.0] * 5
        self._commanded_q = [0.0] * 5
        self._initialized_q = False

        # État de la pince (0-100 %)
        self._target_gripper_pct = 0.0
        self._last_sent_gripper_pct = None

        # 4. Publishers & Subscribers (QoS=1 sur le teleop pour éviter la file d'attente)
        self._tcp_pose_pub = self.create_publisher(PoseStamped, "/end_effector_pose", 10)
        self._joint_cmd_pub = self.create_publisher(JointState, "/joint_command", 10)

        self._joint_state_sub = self.create_subscription(
            JointState, "/joint_states", self._cb_joint_states, 10
        )
        self._instruction_sub = self.create_subscription(
            Vector3Stamped, "/instruction_vector", self._cb_instruction_vector, 1
        )
        self._grab_sub = self.create_subscription(
            Int32, "/instruction_grab", self._cb_instruction_grab, 1
        )

        # 5. Service de déplacement absolu
        if HAS_GOTO_SRV:
            self._goto_srv = self.create_service(GoToTarget, "/go_to_target", self._cb_go_to_target)

        # 6. Timer d'interpolation douce à 50 Hz (20 ms)
        self._interp_timer = self.create_timer(0.02, self._control_loop)

        self.get_logger().info("Brain node ready.")

    def _to_ikpy_q(self, q_5: list[float]) -> list[float]:
        """Ajoute les 2 joints fixes (OriginLink et gripper_frame_joint) autour des 5 angles moteurs."""
        return [0.0] + list(q_5) + [0.0]

    def _cb_joint_states(self, msg: JointState):
        name_to_pos = dict(zip(msg.name, msg.position))
        if all(name in name_to_pos for name in ARM_JOINT_NAMES):
            self._current_q = [float(name_to_pos[name]) for name in ARM_JOINT_NAMES]
            if not self._initialized_q:
                self._target_q = list(self._current_q)
                self._commanded_q = list(self._current_q)
                self._initialized_q = True
            self._publish_end_effector_pose()

    def _forward_kinematics(self, q: list[float]) -> np.ndarray:
        """Retourne la matrice homogène 4x4 du TCP (gripper_frame_link)."""
        return self._chain.forward_kinematics(self._to_ikpy_q(q))

    def _publish_end_effector_pose(self):
        tf_matrix = self._forward_kinematics(self._current_q)
        pos = tf_matrix[:3, 3]
        quat = Rotation.from_matrix(tf_matrix[:3, :3]).as_quat()  # [x, y, z, w]

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.pose.position.x = float(pos[0])
        msg.pose.position.y = float(pos[1])
        msg.pose.position.z = float(pos[2])
        msg.pose.orientation.x = float(quat[0])
        msg.pose.orientation.y = float(quat[1])
        msg.pose.orientation.z = float(quat[2])
        msg.pose.orientation.w = float(quat[3])
        self._tcp_pose_pub.publish(msg)

    def _solve_ik(self, target_position: list[float]) -> list[float]:
        """Calcule les 5 angles moteurs (rad) pour atteindre target_position = [x, y, z]."""
        init_q = self._to_ikpy_q(self._current_q)
        for i, link in enumerate(self._chain.links):
            low, high = link.bounds
            if low is not None and init_q[i] < low:
                init_q[i] = float(low)
            if high is not None and init_q[i] > high:
                init_q[i] = float(high)

        ik_full = self._chain.inverse_kinematics(
            target_position=target_position,
            initial_position=init_q,
        )
        return [float(angle) for angle in ik_full[1:6]]

    def _send_joint_command(self, q: list[float], gripper_pct: float | None = None):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        if gripper_pct is not None:
            msg.name = JOINT_NAMES
            msg.position = list(q) + [float(gripper_pct)]
        else:
            msg.name = ARM_JOINT_NAMES
            msg.position = list(q)
        self._joint_cmd_pub.publish(msg)

    def _cb_instruction_grab(self, msg: Int32):
        """Met à jour l'ouverture cible de la pince (0 à 100 %)."""
        self._target_gripper_pct = float(max(0, min(100, msg.data)))

    def _cb_instruction_vector(self, msg: Vector3Stamped):
        """Déplace la cible TCP selon le vecteur envoyé par teleoperator_node."""
        if not self._initialized_q:
            return

        vx, vy, vz = msg.vector.x, msg.vector.y, msg.vector.z

        # Arrêt immédiat au relâchement des touches
        if vx == 0.0 and vy == 0.0 and vz == 0.0:
            self._target_q = list(self._commanded_q)
            return

        step_m = float(self.get_parameter("step_size_m").value)
        current_tcp = self._forward_kinematics(self._commanded_q)[:3, 3]
        new_target = [
            float(current_tcp[0] + vx * step_m),
            float(current_tcp[1] + vy * step_m),
            float(current_tcp[2] + vz * step_m),
        ]

        try:
            q_sol = self._solve_ik(new_target)
            fk_pos = self._forward_kinematics(q_sol)[:3, 3]
            if np.linalg.norm(fk_pos - np.array(new_target)) < 0.01:
                self._target_q = q_sol
        except ValueError as e:
            self.get_logger().warn(f"IK ignorée (hors limites) : {e}")

    def _cb_go_to_target(self, request, response):
        p = request.target_pose.pose.position
        target = [p.x, p.y, p.z]
        try:
            q_sol = self._solve_ik(target)
            fk_pos = self._forward_kinematics(q_sol)[:3, 3]
            err_cm = float(np.linalg.norm(fk_pos - np.array(target)) * 100.0)

            if err_cm < 1.0:
                self._target_q = q_sol
                response.success = True
            else:
                response.success = False
            response.message = f"IK solved (error: {err_cm:.2f} cm)"
        except Exception as e:
            response.success = False
            response.message = f"IK failed: {e}"
        return response

    def _control_loop(self):
        """Interpole progressivement _commanded_q vers _target_q à 50 Hz et pilote la pince."""
        if not self._initialized_q:
            return

        speed = float(self.get_parameter("max_speed_rad_s").value)
        max_step = speed * 0.02

        diffs = [self._target_q[i] - self._commanded_q[i] for i in range(5)]
        max_diff = max(abs(d) for d in diffs)
        arm_moved = False

        if max_diff > 1e-4:
            scale = min(1.0, max_step / max_diff)
            for i in range(5):
                self._commanded_q[i] += diffs[i] * scale
            arm_moved = True

        gripper_changed = (
            self._last_sent_gripper_pct is None
            or abs(self._target_gripper_pct - self._last_sent_gripper_pct) > 0.1
        )

        # On ne publie sur /joint_command que si le bras bouge OU si la valeur de la pince a changé
        if arm_moved or gripper_changed:
            self._send_joint_command(
                self._commanded_q,
                gripper_pct=self._target_gripper_pct if gripper_changed else None,
            )
            if gripper_changed:
                self._last_sent_gripper_pct = self._target_gripper_pct


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