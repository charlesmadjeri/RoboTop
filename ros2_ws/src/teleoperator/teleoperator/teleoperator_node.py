#!/usr/bin/env python3
import threading

import rclpy
import rclpy.node
from rclpy.executors import ExternalShutdownException
from geometry_msgs.msg import Vector3Stamped
from std_msgs.msg import Int32
from pynput import keyboard

INSTRUCTION_HZ = 20.0
STEP = 1.0
BOOST_STEP = 3.0  # used while LSHIFT is held

CLAW_STEP = 1      # percent added/removed per tick while P or M is held
CLAW_MIN = 0
CLAW_MAX = 100
CLAW_START = 0     # initial claw value (percent)

# key -> (axis index, sign)
VECTOR_KEYS = {
    'z': (0, +1.0),  # +x forward
    's': (0, -1.0),  # -x backward
    'q': (1, +1.0),  # +y left
    'd': (1, -1.0),  # -y right
    'a': (2, +1.0),  # +z up
    'e': (2, -1.0),  # -z down
}

# key -> sign
CLAW_KEYS = {
    'p': +1,  # open claw
    'm': -1,  # close claw
}

ALL_KEYS = set(VECTOR_KEYS) | set(CLAW_KEYS)


class TeleoperatorNode(rclpy.node.Node):

    def __init__(self):
        super().__init__('teleoperator_node')

        self._instruction_pub = self.create_publisher(
            Vector3Stamped, "/instruction_vector", 10)
        self._claw_pub = self.create_publisher(
            Int32, "/instruction_grab", 10)

        # State shared between the pynput thread and the ROS timer
        self._lock = threading.Lock()
        self._pressed = set()
        self._boost = False

        # Claw is an absolute percentage: it holds its value when no key is pressed
        self._claw = CLAW_START

        self._listener = keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release)
        self._listener.start()

        self.create_timer(1.0 / INSTRUCTION_HZ, self._publish_instructions)
        self.get_logger().info(
            "ZQSD = x/y, A/E = z, P/M = open/close claw, "
            "hold LSHIFT for boost. Release = back to zero (claw keeps its value).")

    # ---- pynput callbacks (run in the listener thread) ----
    @staticmethod
    def _key_name(key):
        # Normalise to lowercase so Shift+Z and Z are the same key,
        # even if Shift is released before the letter.
        char = getattr(key, 'char', None)
        return char.lower() if char else None

    def _on_press(self, key):
        with self._lock:
            if key in (keyboard.Key.shift_l, keyboard.Key.shift):
                self._boost = True
                return
            name = self._key_name(key)
            if name in ALL_KEYS:
                self._pressed.add(name)

    def _on_release(self, key):
        with self._lock:
            if key in (keyboard.Key.shift_l, keyboard.Key.shift):
                self._boost = False
                return
            name = self._key_name(key)
            self._pressed.discard(name)

    # ---- ROS timer ----
    def _current_vector(self):
        vec = [0.0, 0.0, 0.0]
        with self._lock:
            step = BOOST_STEP if self._boost else STEP
            for name in self._pressed:
                if name in VECTOR_KEYS:
                    axis, sign = VECTOR_KEYS[name]
                    vec[axis] += sign * step  # opposite keys cancel out
        return vec

    def _update_claw(self):
        with self._lock:
            delta = sum(CLAW_KEYS[n] for n in self._pressed if n in CLAW_KEYS)
        self._claw = max(CLAW_MIN, min(CLAW_MAX, self._claw + delta * CLAW_STEP))
        return self._claw

    def _publish_instructions(self):
        x, y, z = self._current_vector()
        msg = Vector3Stamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.vector.x = x
        msg.vector.y = y
        msg.vector.z = z
        self._instruction_pub.publish(msg)

        claw_msg = Int32()
        claw_msg.data = int(self._update_claw())
        self._claw_pub.publish(claw_msg)

    def destroy_node(self):
        self._listener.stop()
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