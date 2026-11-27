import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from so101_perception.detection import find_cube, find_drop_box
from so101_perception.projection import (
    CUBE_CENTER_HEIGHT,
    DROP_BOX_BLOB_HEIGHT,
    pixel_to_ray,
    project_to_plane,
)

DATA = Path(__file__).parent / "data"
EXPECTED = json.loads((DATA / "expected.json").read_text())
CAMERA = json.loads((DATA / "camera.json").read_text())
K = np.array(CAMERA["K"])
ROBOT_FROM_CAMERA = np.array(CAMERA["T_world_camera"])
DROP_BOX_BASE_XY = (0.05, 0.25)
TOLERANCE_M = 0.01

LOOKING_DOWN = np.array([[1.0, 0, 0, 0], [0, -1.0, 0, 0], [0, 0, -1.0, 1.0], [0, 0, 0, 1.0]])
LOOKING_AHEAD = np.array([[0, 0, 1.0, 0], [-1.0, 0, 0, 0], [0, -1.0, 0, 0.5], [0, 0, 0, 1.0]])
CENTER = (K[0, 2], K[1, 2])


def load(name):
    return cv2.cvtColor(cv2.imread(str(DATA / f"{name}.png")), cv2.COLOR_BGR2RGB)


def camera_pixel_of(point):
    in_camera = np.linalg.inv(ROBOT_FROM_CAMERA) @ np.append(point, 1.0)
    pixel = K @ in_camera[:3]
    return pixel[:2] / pixel[2]


def test_ray_through_the_principal_point_follows_the_optical_axis():
    assert pixel_to_ray(*CENTER, K) == pytest.approx([0, 0, 1])


def test_camera_looking_down_sees_its_axis_hit_the_plane_below():
    point = project_to_plane(*CENTER, K, LOOKING_DOWN, 0.025)
    assert point == pytest.approx([0, 0, 0.025])


def test_pixel_offset_moves_the_point_by_the_pinhole_ratio():
    depth = 1.0 - 0.025
    pixel_u = CENTER[0] + K[0, 0] * 0.1 / depth
    point = project_to_plane(pixel_u, CENTER[1], K, LOOKING_DOWN, 0.025)
    assert point == pytest.approx([0.1, 0, 0.025])


@pytest.mark.parametrize("target", [[0.25, 0.05, 0.025], [0.1, 0.2, 0.0], [0.3, -0.1, 0.05]])
def test_projecting_the_pixel_of_a_point_returns_that_point(target):
    u, v = camera_pixel_of(np.array(target))
    point = project_to_plane(u, v, K, ROBOT_FROM_CAMERA, target[2])
    assert point == pytest.approx(target, abs=1e-9)


def test_ray_parallel_to_the_plane_is_rejected():
    assert project_to_plane(*CENTER, K, LOOKING_AHEAD, 0.0) is None


def test_ray_pointing_away_from_the_plane_is_rejected():
    assert project_to_plane(CENTER[0], 0.0, K, LOOKING_AHEAD, 0.0) is None


def test_ray_pointing_toward_the_plane_hits_it_in_front_of_the_camera():
    point = project_to_plane(CENTER[0], 2 * CENTER[1], K, LOOKING_AHEAD, 0.0)
    assert point[0] > 0 and point[2] == pytest.approx(0.0)


@pytest.mark.parametrize("scene", ["both", "both_other_pose", "cube_only", "dim_light"])
def test_detected_cube_projects_within_one_centimeter_of_the_true_position(scene):
    true_pixel = EXPECTED[scene]["cube"]
    truth = project_to_plane(*true_pixel, K, ROBOT_FROM_CAMERA, CUBE_CENTER_HEIGHT)
    found = project_to_plane(*find_cube(load(scene)), K, ROBOT_FROM_CAMERA, CUBE_CENTER_HEIGHT)
    assert np.linalg.norm(found[:2] - truth[:2]) < TOLERANCE_M


@pytest.mark.parametrize("scene", ["both", "both_other_pose", "box_only", "dim_light"])
def test_detected_drop_box_projects_within_one_centimeter_of_its_base(scene):
    found = project_to_plane(*find_drop_box(load(scene)), K, ROBOT_FROM_CAMERA, DROP_BOX_BLOB_HEIGHT)
    assert np.linalg.norm(found[:2] - DROP_BOX_BASE_XY) < TOLERANCE_M
