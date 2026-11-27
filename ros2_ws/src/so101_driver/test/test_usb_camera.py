import logging
from types import SimpleNamespace

import pytest
from so101_driver import driver_node, usb_camera
from so101_driver.driver_node import DriverNode


class FakeCamera:
    def __init__(self):
        self.disconnected = False

    def read_latest(self):
        return None

    def disconnect(self):
        self.disconnected = True


class RecordingPublisher:
    created = []

    def __init__(self, node, read_frame, camera_info, hz, on_stop=None):
        self.read_frame = read_frame
        self.on_stop = on_stop
        self.started = False
        self.created.append(self)

    def start(self):
        self.started = True


@pytest.fixture(autouse=True)
def publishers(monkeypatch):
    RecordingPublisher.created = []
    monkeypatch.setattr(driver_node, "FramePublisher", RecordingPublisher)
    return RecordingPublisher.created


def driver(**parameters):
    robot = SimpleNamespace(disconnected=False)
    robot.disconnect = lambda: setattr(robot, "disconnected", True)
    return SimpleNamespace(
        _robot=robot,
        declare_parameter=lambda name, default: SimpleNamespace(value=parameters.get(name, default)),
        get_logger=lambda: logging.getLogger("driver"),
    )


def test_negative_index_disables_the_camera(monkeypatch, publishers):
    monkeypatch.setattr(driver_node, "open_usb_camera", lambda *args: pytest.fail("camera opened"))
    DriverNode._start_usb_camera(driver(camera_index=-1))
    assert publishers == []


def test_camera_is_opened_with_the_requested_index_and_size(monkeypatch):
    opened = []
    monkeypatch.setattr(driver_node, "open_usb_camera", lambda *args: opened.append(args) or FakeCamera())
    DriverNode._start_usb_camera(driver(camera_index=3, camera_width=320, camera_height=240))
    assert opened == [(3, 320, 240, "")]


def test_frames_come_from_the_camera_and_it_is_released_on_stop(monkeypatch, publishers):
    camera = FakeCamera()
    monkeypatch.setattr(driver_node, "open_usb_camera", lambda *args: camera)
    DriverNode._start_usb_camera(driver(camera_index=0))
    publisher = publishers[0]
    assert publisher.started
    assert publisher.read_frame == camera.read_latest
    publisher.on_stop()
    assert camera.disconnected


def test_unopenable_camera_releases_the_arm_and_raises(monkeypatch):
    def unavailable(*args):
        raise ConnectionError("no camera")

    monkeypatch.setattr(driver_node, "open_usb_camera", unavailable)
    node = driver(camera_index=0)
    with pytest.raises(ConnectionError, match="no camera"):
        DriverNode._start_usb_camera(node)
    assert node._robot.disconnected


def test_usb_camera_is_configured_from_the_index_and_connected(monkeypatch):
    seen = {}

    class RecordingCamera:
        def __init__(self, config):
            seen["config"] = config

        def connect(self):
            seen["connected"] = True

    monkeypatch.setattr(usb_camera, "OpenCVCamera", RecordingCamera)
    usb_camera.open_usb_camera(2, 640, 480)
    config = seen["config"]
    assert (config.index_or_path, config.width, config.height) == (2, 640, 480)
    assert seen["connected"]


class FourccCamera:
    configs = []

    def __init__(self, config):
        self.configs.append(config)

    def connect(self):
        self.connected = True


def open_with_format(monkeypatch, *arguments):
    FourccCamera.configs = []
    monkeypatch.setattr(usb_camera, "OpenCVCamera", FourccCamera)
    camera = usb_camera.open_usb_camera(*arguments)
    return camera, FourccCamera.configs[0]


def test_pixel_format_is_passed_to_the_camera(monkeypatch):
    _, config = open_with_format(monkeypatch, 0, 1280, 720, "MJPG")
    assert config.fourcc == "MJPG"
    assert (config.width, config.height, config.fps) == (1280, 720, usb_camera.CAMERA_FPS)


def test_no_pixel_format_keeps_the_automatic_detection(monkeypatch):
    _, config = open_with_format(monkeypatch, 0, 640, 480)
    assert config.fourcc is None


def test_camera_is_connected_before_it_is_returned(monkeypatch):
    camera, _ = open_with_format(monkeypatch, 0, 640, 480, "")
    assert camera.connected
