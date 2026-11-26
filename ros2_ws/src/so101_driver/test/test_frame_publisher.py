import threading

import numpy as np
from builtin_interfaces.msg import Time
from sensor_msgs.msg import CameraInfo

from so101_driver.frame_publisher import FramePublisher


class FakeLogger:
    def __init__(self):
        self.warnings = []

    def warning(self, message, **kwargs):
        self.warnings.append(message)


class FakeClock:
    def now(self):
        return self

    def to_msg(self):
        return Time(sec=12, nanosec=34)


class FakePublisher:
    def __init__(self, enough, count):
        self.messages = []
        self._enough = enough
        self._count = count

    def publish(self, message):
        self.messages.append(message)
        if len(self.messages) >= self._count:
            self._enough.set()


class FakeNode:
    def __init__(self, enough, count):
        self.publishers = {}
        self._enough = enough
        self._count = count
        self.logger = FakeLogger()

    def create_publisher(self, message_type, topic, depth):
        self.publishers[topic] = FakePublisher(self._enough, self._count)
        return self.publishers[topic]

    def get_clock(self):
        return FakeClock()

    def get_logger(self):
        return self.logger


class FakeCamera:
    def __init__(self, outcomes):
        self._outcomes = list(outcomes)
        self.is_connected = True

    def read_latest(self):
        outcome = self._outcomes.pop(0) if len(self._outcomes) > 1 else self._outcomes[0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def disconnect(self):
        self.is_connected = False


def frame():
    return np.zeros((4, 6, 3), dtype=np.uint8)


def run(outcomes, count=1):
    enough = threading.Event()
    node = FakeNode(enough, count)
    camera = FakeCamera(outcomes)
    info = CameraInfo(width=6, height=4)
    info.header.frame_id = "external_cam_optical_frame"
    publisher = FramePublisher(node, camera.read_latest, info, hz=200.0, on_stop=camera.disconnect)
    publisher.start()
    assert enough.wait(timeout=5.0)
    publisher.stop()
    return node, camera


def test_frame_is_published_as_rgb8_in_the_camera_frame():
    node, _ = run([frame()])
    image = node.publishers["/external_cam/image_raw"].messages[0]
    assert image.encoding == "rgb8"
    assert (image.width, image.height) == (6, 4)
    assert image.header.frame_id == "external_cam_optical_frame"


def test_image_and_camera_info_share_the_same_stamp():
    node, _ = run([frame()])
    image = node.publishers["/external_cam/image_raw"].messages[0]
    info = node.publishers["/external_cam/camera_info"].messages[0]
    assert image.header.stamp == info.header.stamp == Time(sec=12, nanosec=34)


def test_stale_frame_is_skipped_without_stopping_the_stream():
    node, _ = run([TimeoutError("old"), RuntimeError("none yet"), frame()])
    assert len(node.publishers["/external_cam/image_raw"].messages) >= 1
    assert len(node.logger.warnings) == 2


def test_stop_runs_the_stop_callback():
    _, camera = run([frame()])
    assert camera.is_connected is False
