import time

import cv2
import numpy as np
import pytest
from test_detection import BOX_TOLERANCE_PX, CUBE_TOLERANCE_PX, EXPECTED, distance, load

from so101_perception.detector import detect, detect_cube

REAL_CUBE_FULL_SCALE = (612.0, 120.0)
FULL_WIDTH = 1280
TIME_BUDGET_S = 0.1


def at_full_size(name):
    half = load(name)
    return cv2.resize(half, (FULL_WIDTH, half.shape[0] * 2), interpolation=cv2.INTER_LINEAR)


@pytest.mark.parametrize("scene", ["both", "both_other_pose", "cube_only", "box_only", "dim_light"])
def test_the_simulated_scenes_give_the_same_cube_and_box_as_the_plain_detector(scene):
    found = detect(load(scene))
    for kind, tolerance in (("cube", CUBE_TOLERANCE_PX), ("box", BOX_TOLERANCE_PX)):
        if EXPECTED[scene][kind] is None:
            assert getattr(found, kind) is None
        else:
            assert distance(getattr(found, kind), EXPECTED[scene][kind]) < tolerance


def test_an_empty_scene_has_neither_cube_nor_box():
    found = detect(load("empty"))
    assert found.cube is None and found.box is None


def test_the_cube_is_found_in_a_real_frame_of_the_real_camera():
    found = detect(load("real_cube"))
    assert distance(found.cube, (REAL_CUBE_FULL_SCALE[0] / 2, REAL_CUBE_FULL_SCALE[1] / 2)) < CUBE_TOLERANCE_PX / 2


def test_the_dark_power_adapter_of_the_real_frame_is_not_a_drop_box():
    assert detect(load("real_cube")).box is None


def test_a_full_size_frame_is_shrunk_and_the_pixel_comes_back_in_full_size_coordinates():
    found = detect(at_full_size("real_cube"))
    assert distance(found.cube, REAL_CUBE_FULL_SCALE) < CUBE_TOLERANCE_PX


def test_the_cube_alone_is_the_cube_of_the_full_detection():
    image = at_full_size("real_cube")
    assert detect_cube(image) == detect(image).cube


def test_a_full_size_frame_is_detected_within_the_time_budget():
    image = at_full_size("real_cube")
    detect(image)
    durations = []
    for _ in range(3):
        start = time.perf_counter()
        detect(image)
        durations.append(time.perf_counter() - start)
    assert min(durations) < TIME_BUDGET_S


def test_a_blue_tinted_scene_still_gives_the_cube():
    image = load("both")
    tinted = np.clip(image * np.array([0.8, 0.9, 1.15]), 0, 255).astype(np.uint8)
    assert distance(detect(tinted).cube, EXPECTED["both"]["cube"]) < CUBE_TOLERANCE_PX
