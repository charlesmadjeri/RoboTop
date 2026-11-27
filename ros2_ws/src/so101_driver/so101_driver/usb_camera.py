from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

CAMERA_FPS = 30


def open_usb_camera(index: int, width: int, height: int, fourcc: str = "") -> OpenCVCamera:
    config = OpenCVCameraConfig(index_or_path=index, fps=CAMERA_FPS, width=width, height=height, fourcc=fourcc or None)
    camera = OpenCVCamera(config)
    camera.connect()
    return camera
