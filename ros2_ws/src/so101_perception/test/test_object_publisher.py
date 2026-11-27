import numpy as np
import pytest
from builtin_interfaces.msg import Time
from scene_images import DROP_BOX_BASE_XY, K, ROBOT_FROM_CAMERA, load
from sensor_msgs.msg import CameraInfo, Image

from so101_perception.object_publisher import DROP_BOX_TOPIC, OBJECT_TOPIC, ObjectPublisher
from so101_perception.pipeline import locate
from so101_perception.projection import CUBE_CENTER_HEIGHT, DROP_BOX_BLOB_HEIGHT

STAMP = Time(sec=12, nanosec=34)


class FakeLogger:
    def __init__(self):
        self.warnings = []

    def warning(self, message, **kwargs):
        self.warnings.append(message)


class FakePublisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class FakeNode:
    def __init__(self):
        self.publishers = {}
        self.logger = FakeLogger()

    def create_publisher(self, message_type, topic, depth):
        self.publishers[topic] = FakePublisher()
        return self.publishers[topic]

    def get_logger(self):
        return self.logger


def image_of(scene):
    pixels = load(scene)
    message = Image(height=pixels.shape[0], width=pixels.shape[1], encoding="rgb8", step=pixels.shape[1] * 3)
    message.data = pixels.tobytes()
    message.header.stamp = STAMP
    return message


def camera_info():
    return CameraInfo(k=K.flatten().tolist())


def start(pose=ROBOT_FROM_CAMERA, known_camera=True):
    node = FakeNode()
    holder = {"pose": pose}
    publisher = ObjectPublisher(node, lambda: holder["pose"], CUBE_CENTER_HEIGHT, DROP_BOX_BLOB_HEIGHT)
    if known_camera:
        publisher.on_camera_info(camera_info())
    return node, publisher, holder


def test_both_targets_are_published_in_the_robot_frame_with_the_image_stamp():
    node, publisher, _ = start()
    publisher.on_image(image_of("both"))
    cube, box = locate(load("both"), K, ROBOT_FROM_CAMERA)
    for topic, expected in ((OBJECT_TOPIC, cube), (DROP_BOX_TOPIC, box)):
        [message] = node.publishers[topic].messages
        assert message.header.frame_id == "world"
        assert message.header.stamp == STAMP
        position = message.pose.position
        assert (position.x, position.y, position.z) == pytest.approx(expected)
        assert message.pose.orientation.w == 1.0
    assert np.linalg.norm(np.array(DROP_BOX_BASE_XY) - box[:2]) < 0.01


def test_only_the_target_in_view_is_published():
    node, publisher, _ = start()
    publisher.on_image(image_of("cube_only"))
    assert len(node.publishers[OBJECT_TOPIC].messages) == 1
    assert node.publishers[DROP_BOX_TOPIC].messages == []


def test_empty_scene_publishes_nothing():
    node, publisher, _ = start()
    publisher.on_image(image_of("empty"))
    assert node.publishers[OBJECT_TOPIC].messages == node.publishers[DROP_BOX_TOPIC].messages == []


def test_unknown_camera_pose_publishes_nothing_and_warns():
    node, publisher, _ = start(pose=None)
    publisher.on_image(image_of("both"))
    assert node.publishers[OBJECT_TOPIC].messages == []
    assert len(node.logger.warnings) == 1


def test_publishing_starts_once_the_camera_pose_is_known():
    node, publisher, holder = start(pose=None)
    publisher.on_image(image_of("both"))
    holder["pose"] = ROBOT_FROM_CAMERA
    publisher.on_image(image_of("both"))
    assert len(node.publishers[OBJECT_TOPIC].messages) == 1


def test_image_before_the_camera_info_publishes_nothing():
    node, publisher, _ = start(known_camera=False)
    publisher.on_image(image_of("both"))
    assert node.publishers[OBJECT_TOPIC].messages == []
    assert node.logger.warnings == []


def stamped(image, sec):
    image.header.stamp = Time(sec=sec, nanosec=0)
    return image


def test_a_cube_that_disappears_is_republished_with_its_old_stamp_for_five_frames():
    node, publisher, _ = start()
    publisher.on_image(stamped(image_of("both"), 1))
    for sec in range(2, 7):
        publisher.on_image(stamped(image_of("empty"), sec))
    messages = node.publishers[OBJECT_TOPIC].messages
    assert len(messages) == 6
    assert {message.header.stamp.sec for message in messages} == {1}
    assert messages[0].pose.position == messages[-1].pose.position


def test_nothing_is_published_from_the_sixth_frame_without_the_cube():
    node, publisher, _ = start()
    publisher.on_image(image_of("both"))
    for _ in range(6):
        publisher.on_image(image_of("empty"))
    assert len(node.publishers[OBJECT_TOPIC].messages) == 6
    assert len(node.publishers[DROP_BOX_TOPIC].messages) == 6


def test_seeing_the_cube_again_publishes_it_with_the_new_stamp():
    node, publisher, _ = start()
    publisher.on_image(stamped(image_of("both"), 1))
    publisher.on_image(stamped(image_of("empty"), 2))
    publisher.on_image(stamped(image_of("both_other_pose"), 3))
    last = node.publishers[OBJECT_TOPIC].messages[-1]
    assert last.header.stamp.sec == 3


def test_the_cube_and_the_box_are_held_independently():
    node, publisher, _ = start()
    publisher.on_image(stamped(image_of("both"), 1))
    for sec in range(2, 9):
        publisher.on_image(stamped(image_of("box_only"), sec))
    cube_stamps = [message.header.stamp.sec for message in node.publishers[OBJECT_TOPIC].messages]
    box_stamps = [message.header.stamp.sec for message in node.publishers[DROP_BOX_TOPIC].messages]
    assert cube_stamps == [1] * 6
    assert box_stamps == [1, *range(2, 9)]


def test_the_hold_length_is_a_parameter():
    node = FakeNode()
    publisher = ObjectPublisher(
        node, lambda: ROBOT_FROM_CAMERA, CUBE_CENTER_HEIGHT, DROP_BOX_BLOB_HEIGHT, hold_frames=1
    )
    publisher.on_camera_info(camera_info())
    publisher.on_image(image_of("both"))
    for _ in range(3):
        publisher.on_image(image_of("empty"))
    assert len(node.publishers[OBJECT_TOPIC].messages) == 2


def test_frames_without_a_camera_pose_do_not_age_the_held_cube():
    node, publisher, holder = start()
    publisher.on_image(stamped(image_of("both"), 1))
    holder["pose"] = None
    for _ in range(10):
        publisher.on_image(image_of("empty"))
    holder["pose"] = ROBOT_FROM_CAMERA
    publisher.on_image(image_of("empty"))
    assert len(node.publishers[OBJECT_TOPIC].messages) == 2
