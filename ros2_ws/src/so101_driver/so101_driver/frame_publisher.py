import threading
import time
from collections.abc import Callable

from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, Image


class FramePublisher:
    """Publishes frames from read_frame on /external_cam/image_raw and /external_cam/camera_info at hz."""

    def __init__(self, node, read_frame: Callable, camera_info: CameraInfo, hz: float, on_stop: Callable | None = None):
        self._node = node
        self._read_frame = read_frame
        self._camera_info = camera_info
        self._period = 1.0 / hz
        self._on_stop = on_stop
        self._image_pub = node.create_publisher(Image, "/external_cam/image_raw", 1)
        self._camera_info_pub = node.create_publisher(CameraInfo, "/external_cam/camera_info", 1)
        self._stop = threading.Event()
        # Own thread: a render or a USB read must not delay /joint_states.
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self._on_stop is not None:
            self._on_stop()

    def _loop(self):
        bridge = CvBridge()
        next_frame = time.monotonic()
        while not self._stop.wait(max(0.0, next_frame - time.monotonic())):
            next_frame += self._period
            if not self._publish_frame(bridge):
                return

    def _publish_frame(self, bridge: CvBridge) -> bool:
        try:
            frame = self._read_frame()
        except (TimeoutError, RuntimeError):
            self._node.get_logger().warning("Camera gave no fresh frame, skipped", throttle_duration_sec=5.0)
            return True
        image = bridge.cv2_to_imgmsg(frame, encoding="rgb8")
        image.header.stamp = self._node.get_clock().now().to_msg()
        image.header.frame_id = self._camera_info.header.frame_id
        self._camera_info.header.stamp = image.header.stamp
        try:
            self._image_pub.publish(image)
            self._camera_info_pub.publish(self._camera_info)
        except Exception:  # ROS context already shut down (Ctrl-C)
            return False
        return True
