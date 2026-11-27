from so101_driver import usb_camera


class RecordingCamera:
    configs = []

    def __init__(self, config):
        self.configs.append(config)

    def connect(self):
        self.connected = True


def open_with(monkeypatch, *arguments):
    RecordingCamera.configs = []
    monkeypatch.setattr(usb_camera, "OpenCVCamera", RecordingCamera)
    camera = usb_camera.open_usb_camera(*arguments)
    return camera, RecordingCamera.configs[0]


def test_pixel_format_is_passed_to_the_camera(monkeypatch):
    _, config = open_with(monkeypatch, 0, 1280, 720, "MJPG")
    assert config.fourcc == "MJPG"
    assert (config.width, config.height, config.fps) == (1280, 720, usb_camera.CAMERA_FPS)


def test_no_pixel_format_keeps_the_automatic_detection(monkeypatch):
    _, config = open_with(monkeypatch, 0, 640, 480)
    assert config.fourcc is None


def test_camera_is_connected_before_it_is_returned(monkeypatch):
    camera, _ = open_with(monkeypatch, 0, 640, 480, "")
    assert camera.connected
