from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

CAMERA_FPS = 30


def open_usb_camera(index: int, width: int, height: int) -> OpenCVCamera:
    camera = OpenCVCamera(OpenCVCameraConfig(index_or_path=index, fps=CAMERA_FPS, width=width, height=height))
    camera.connect()
    return camera
