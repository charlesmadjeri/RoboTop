import argparse
import sys
from pathlib import Path

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

from so101_perception.image_message import rgb_from_image

TOPIC = "/external_cam/image_raw"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Save one frame of the camera topic as a PNG")
    parser.add_argument("--output", type=Path, default=Path("frame.png"))
    parser.add_argument("--topic", default=TOPIC)
    parser.add_argument("--timeout", type=float, default=10.0, help="seconds to wait for a frame")
    args = parser.parse_args(argv)
    rclpy.init()
    node = Node("grab_frame")
    frames = []
    node.create_subscription(Image, args.topic, frames.append, 1)
    end = node.get_clock().now().nanoseconds + int(args.timeout * 1e9)
    while not frames and node.get_clock().now().nanoseconds < end:
        rclpy.spin_once(node, timeout_sec=0.2)
    node.destroy_node()
    rclpy.shutdown()
    if not frames:
        sys.exit(f"no frame on {args.topic} after {args.timeout} s")
    cv2.imwrite(str(args.output), cv2.cvtColor(rgb_from_image(frames[0]), cv2.COLOR_RGB2BGR))
    print(f"frame saved to {args.output}")


if __name__ == "__main__":
    main()
