import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from so101_perception.detection import find_cube, find_drop_box

DATA = Path(__file__).parent / "data"
EXPECTED = json.loads((DATA / "expected.json").read_text())
CUBE_TOLERANCE_PX = 6.0
BOX_TOLERANCE_PX = 10.0
RED = (220, 15, 60)


def load(name):
    return cv2.cvtColor(cv2.imread(str(DATA / f"{name}.png")), cv2.COLOR_BGR2RGB)


def paint(image, top_left, size, color):
    painted = image.copy()
    x, y = top_left
    painted[y : y + size, x : x + size] = color
    return painted


def distance(found, expected):
    return float(np.hypot(found[0] - expected[0], found[1] - expected[1]))


@pytest.mark.parametrize("scene", ["both", "both_other_pose", "cube_only", "dim_light"])
def test_cube_is_found_at_its_projected_pixel(scene):
    found = find_cube(load(scene))
    assert found is not None
    assert distance(found, EXPECTED[scene]["cube"]) < CUBE_TOLERANCE_PX


@pytest.mark.parametrize("scene", ["both", "both_other_pose", "box_only", "dim_light"])
def test_drop_box_is_found_at_its_projected_pixel(scene):
    found = find_drop_box(load(scene))
    assert found is not None
    assert distance(found, EXPECTED[scene]["box"]) < BOX_TOLERANCE_PX


def test_empty_scene_has_no_cube_and_no_drop_box():
    image = load("empty")
    assert find_cube(image) is None
    assert find_drop_box(image) is None


def test_missing_drop_box_does_not_hide_the_cube():
    image = load("cube_only")
    assert find_cube(image) is not None
    assert find_drop_box(image) is None


def test_missing_cube_does_not_hide_the_drop_box():
    image = load("box_only")
    assert find_drop_box(image) is not None
    assert find_cube(image) is None


def test_red_speck_below_the_minimum_area_is_ignored():
    assert find_cube(paint(load("empty"), (40, 40), 4, RED)) is None


def test_red_wrapping_around_hue_zero_is_found_as_the_cube():
    image = paint(load("empty"), (100, 100), 25, (220, 20, 15))
    assert distance(find_cube(image), (112, 112)) < 1.0


def test_largest_red_blob_wins_over_a_smaller_one():
    image = paint(load("both"), (20, 20), 9, RED)
    assert distance(find_cube(image), EXPECTED["both"]["cube"]) < CUBE_TOLERANCE_PX


def test_uniform_gray_image_has_no_cube():
    assert find_cube(np.full((240, 320, 3), 90, np.uint8)) is None


def test_pixel_scales_with_the_image_resolution():
    image = load("both")
    doubled = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    small = find_cube(image)
    large = find_cube(doubled)
    assert distance(large, (2 * small[0], 2 * small[1])) < 2 * CUBE_TOLERANCE_PX
