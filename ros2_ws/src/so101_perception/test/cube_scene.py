import json
from pathlib import Path

import cv2
import numpy as np

from so101_perception.camera_pose import intrinsics_from_horizontal_fov
from so101_perception.projection import CUBE_CENTER_HEIGHT

CAMERA = json.loads((Path(__file__).parent / "data" / "camera.json").read_text())
ROBOT_FROM_CAMERA = np.array(CAMERA["T_world_camera"])
WIDTH, HEIGHT = 1280, 720
K = intrinsics_from_horizontal_fov(WIDTH, HEIGHT, 75.0)
PLACES = [(0.20, 0.15), (0.25, 0.05), (0.10, 0.20), (0.15, 0.10), (0.30, 0.15), (0.20, 0.25)]
CUBE_HALF_SIDE = 20
RED = (255, 0, 0)


def project_places(robot_from_camera, places):
    points = np.array([[x, y, CUBE_CENTER_HEIGHT, 1.0] for x, y in places]).T
    in_camera = (np.linalg.inv(robot_from_camera) @ points)[:3]
    pixels = K @ in_camera
    return (pixels[:2] / pixels[2]).T


def render_cube_view(pixel):
    canvas = np.full((HEIGHT, WIDTH, 3), 128, np.uint8)
    u, v = (int(round(value)) for value in pixel)
    cv2.rectangle(canvas, (u - CUBE_HALF_SIDE, v - CUBE_HALF_SIDE), (u + CUBE_HALF_SIDE, v + CUBE_HALF_SIDE), RED, -1)
    return canvas


def write_view(path, view):
    cv2.imwrite(str(path), cv2.cvtColor(view, cv2.COLOR_RGB2BGR))
    return path


def pose_error(found, expected):
    translation = float(np.linalg.norm(found[:3, 3] - expected[:3, 3]))
    cosine = (np.trace(found[:3, :3].T @ expected[:3, :3]) - 1) / 2
    return translation, float(np.degrees(np.arccos(np.clip(cosine, -1, 1))))
