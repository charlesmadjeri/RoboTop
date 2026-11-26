import math

from sensor_msgs.msg import CameraInfo

DISTORTION_MODEL = "plumb_bob"
IDENTITY_ROTATION = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]


def pinhole_intrinsics(width: int, height: int, fov_deg: float) -> list[float]:
    focal = (width / 2) / math.tan(math.radians(fov_deg) / 2)
    return [focal, 0.0, width / 2, 0.0, focal, height / 2, 0.0, 0.0, 1.0]


def build_camera_info(K: list[float], width: int, height: int, frame_id: str) -> CameraInfo:
    k = [float(value) for value in K]
    msg = CameraInfo()
    msg.header.frame_id = frame_id
    msg.width = width
    msg.height = height
    msg.distortion_model = DISTORTION_MODEL
    msg.d = [0.0] * 5
    msg.k = k
    msg.r = list(IDENTITY_ROTATION)
    msg.p = [k[0], k[1], k[2], 0.0, k[3], k[4], k[5], 0.0, k[6], k[7], k[8], 0.0]
    return msg
