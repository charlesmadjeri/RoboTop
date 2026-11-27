import cv2
import numpy as np
import pytest

from perception_bench.methods import DEFAULT_METHODS, METHODS, background_diff, hsv_fixed, hsv_normalized, lab_geometry

GREY = (110, 110, 118)
CUBE_RGB = (200, 30, 40)
BOX_RGB = (150, 90, 50)
CUBE_BOX = (150, 100, 180, 130)
BOX_BOX = (200, 150, 280, 220)


def scene(size=(240, 320), cube=True, box=True, cast=(1.0, 1.0, 1.0)):
    image = np.full((*size, 3), GREY, np.uint8)
    scale = size[0] / 240
    for present, colour, (x0, y0, x1, y1) in ((box, BOX_RGB, BOX_BOX), (cube, CUBE_RGB, CUBE_BOX)):
        if present:
            image[int(y0 * scale) : int(y1 * scale), int(x0 * scale) : int(x1 * scale)] = colour
    return np.clip(image * np.array(cast), 0, 255).astype(np.uint8)


@pytest.fixture(params=sorted(DEFAULT_METHODS))
def detect(request):
    return METHODS[request.param](scene(cube=False))


def test_every_method_finds_the_cube_and_the_box(detect):
    found = detect(scene())
    assert found["cube"] == pytest.approx((164.5, 114.5), abs=1.5)
    assert found["box"] == pytest.approx((239.5, 184.5), abs=1.5)


def test_every_method_reports_nothing_on_an_empty_scene(detect):
    found = detect(scene(cube=False, box=False))
    assert found["cube"] is None


def test_every_method_reports_a_missing_cube_while_the_box_stays(detect):
    found = detect(scene(cube=False))
    assert found["cube"] is None
    assert found["box"] is not None


def test_every_method_runs_at_another_resolution(detect):
    found = detect(scene(size=(480, 640)))
    assert found["cube"] == pytest.approx((329.0, 229.0), abs=3)


def test_white_balance_removes_a_colour_cast():
    cast = scene(cube=False, box=False, cast=(1.3, 1.0, 0.7))
    balanced = hsv_normalized.white_balance(cast).reshape(-1, 3)
    assert np.ptp(np.median(balanced, axis=0)) <= 2


def test_normalized_hsv_finds_a_cube_the_fixed_ranges_miss_under_a_strong_cast():
    green = scene(cast=(1.0, 2.5, 1.0))
    assert hsv_fixed.detect(green)["cube"] is None
    assert hsv_normalized.detect(green)["cube"] == pytest.approx((164.5, 114.5), abs=2)


def test_lab_geometry_rejects_a_red_bar_that_is_not_square():
    image = scene(cube=False)
    image[100:106, 20:140] = CUBE_RGB
    assert lab_geometry.detect(image)["cube"] is None


def test_lab_geometry_still_finds_a_cube_when_the_colour_is_off():
    image = scene(cube=False)
    image[100:130, 150:180] = (170, 70, 80)
    assert lab_geometry.detect(image)["cube"] == pytest.approx((164.5, 114.5), abs=2)


def test_background_diff_ignores_a_global_brightness_change():
    detect = background_diff.make(scene(cube=False))
    assert detect(scene(cube=False, cast=(0.6, 0.6, 0.6)))["cube"] is None


def test_background_diff_ignores_a_red_thing_that_was_already_in_the_empty_frame():
    empty = scene(cube=True)
    detect = background_diff.make(empty)
    assert detect(empty)["cube"] is None


def test_background_diff_ignores_a_change_that_is_not_red():
    detect = background_diff.make(scene(cube=False))
    image = scene(cube=False)
    cv2.rectangle(image, (20, 20), (80, 80), (30, 30, 30), -1)
    assert detect(image)["cube"] is None
