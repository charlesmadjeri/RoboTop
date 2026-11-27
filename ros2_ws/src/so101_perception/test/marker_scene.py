import json
from pathlib import Path

import cv2
import numpy as np

from so101_perception.camera_pose import intrinsics_from_horizontal_fov, marker_corners
from so101_perception.marker_detection import DICTIONARY

CAMERA = json.loads((Path(__file__).parent / "data" / "camera.json").read_text())
ROBOT_FROM_CAMERA = np.array(CAMERA["T_world_camera"])
WIDTH, HEIGHT = 640, 480
K = intrinsics_from_horizontal_fov(WIDTH, HEIGHT, 70.0)
MARKER_PIXELS = 200
QUIET_ZONE_PIXELS = 40


def project(robot_from_camera, points):
    in_camera = (np.linalg.inv(robot_from_camera) @ np.vstack([points.T, np.ones(len(points))]))[:3]
    pixels = K @ in_camera
    return (pixels[:2] / pixels[2]).T


def project_marker(robot_from_camera, marker_pose, size):
    corners = marker_corners(size)
    in_robot = (marker_pose @ np.vstack([corners.T, np.ones(4)]))[:3].T
    return project(robot_from_camera, in_robot)


def render_marker_view(robot_from_camera, marker_pose, size, marker_id=0):
    marker = cv2.aruco.generateImageMarker(DICTIONARY, marker_id, MARKER_PIXELS)
    padded = cv2.copyMakeBorder(marker, *[QUIET_ZONE_PIXELS] * 4, cv2.BORDER_CONSTANT, value=255)
    scale = padded.shape[0] / MARKER_PIXELS
    sheet = project_marker(robot_from_camera, marker_pose, size * scale)
    edge = padded.shape[0] - 0.5
    source = np.array([[-0.5, -0.5], [edge, -0.5], [edge, edge], [-0.5, edge]], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(source, sheet.astype(np.float32))
    canvas = np.full((HEIGHT, WIDTH), 128, np.uint8)
    cv2.warpPerspective(padded, transform, (WIDTH, HEIGHT), dst=canvas, borderMode=cv2.BORDER_TRANSPARENT)
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)


def pose_error(found, expected):
    translation = float(np.linalg.norm(found[:3, 3] - expected[:3, 3]))
    cosine = (np.trace(found[:3, :3].T @ expected[:3, :3]) - 1) / 2
    return translation, float(np.degrees(np.arccos(np.clip(cosine, -1, 1))))
