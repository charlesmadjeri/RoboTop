#!/usr/bin/env python3
import os

import rclpy
import rclpy.node
from rclpy.executors import ExternalShutdownException
from geometry_msgs.msg import Vector3Stamped

try:
    from keyboard_layout import AXIS_KEYS, describe_keys, resolve_layout
    from pynput_keys import PynputKeys
    from terminal_keys import TerminalKeys
except ImportError:
    from teleoperator.keyboard_layout import AXIS_KEYS, describe_keys, resolve_layout
    from teleoperator.pynput_keys import PynputKeys
    from teleoperator.terminal_keys import TerminalKeys

INSTRUCTION_HZ = 20.0
STEP = 1.0
BOOST_STEP = 3.0  # used while a shift key is held


def input_source_name(requested):
    if requested in ("pynput", "terminal"):
        return requested
    return "terminal" if os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland" else "pynput"


class TeleoperatorNode(rclpy.node.Node):

    def __init__(self):
        super().__init__('teleoperator_node')
        self.declare_parameter('keyboard_layout', 'auto')
        self.declare_parameter('input', 'auto')

        self._instruction_pub = self.create_publisher(
            Vector3Stamped, "/instruction_vector", 10)

        layout, origin = resolve_layout(self.get_parameter('keyboard_layout').value)
        self._keymap = AXIS_KEYS[layout]
        source = input_source_name(self.get_parameter('input').value)
        self._keys = TerminalKeys(self._keymap) if source == 'terminal' else PynputKeys(self._keymap)
        self._keys.start()
        self._first_key_logged = False

        self.create_timer(1.0 / INSTRUCTION_HZ, self._publish_instructions)
        self.get_logger().info(
            f"Layout {layout} ({origin}), keys from {source}. {describe_keys(layout)}. "
            "Shift = boost. Release = back to zero.")

    def _current_vector(self):
        names, boost = self._keys.snapshot()
        step = BOOST_STEP if boost else STEP
        vec = [0.0, 0.0, 0.0]
        for name in names:
            axis, sign = self._keymap[name]
            vec[axis] += sign * step  # opposite keys cancel out
        return vec

    def _publish_instructions(self):
        x, y, z = self._current_vector()
        if (x or y or z) and not self._first_key_logged:
            self._first_key_logged = True
            self.get_logger().info("First key received, the keyboard input works.")
        msg = Vector3Stamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.vector.x = x
        msg.vector.y = y
        msg.vector.z = z
        self._instruction_pub.publish(msg)

    def destroy_node(self):
        self._keys.stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TeleoperatorNode()
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
