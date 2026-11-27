import cv2
import numpy as np

from so101_perception.normalization import MAX_WIDTH, equalize, shrink_to_width, white_balance


def test_a_frame_wider_than_the_limit_is_halved_and_the_scale_brings_pixels_back():
    image = np.zeros((720, 1280, 3), np.uint8)
    small, scale = shrink_to_width(image)
    assert small.shape == (360, MAX_WIDTH, 3)
    assert scale == 2.0


def test_a_frame_within_the_limit_is_left_alone():
    image = np.zeros((240, 320, 3), np.uint8)
    small, scale = shrink_to_width(image)
    assert small is image
    assert scale == 1.0


def test_a_colored_cast_over_a_uniform_scene_is_removed():
    tinted = np.full((20, 20, 3), (90, 120, 160), np.uint8)
    balanced = white_balance(tinted)
    assert np.ptp(balanced[0, 0].astype(int)) <= 1


def test_a_low_contrast_image_gets_more_contrast_in_its_brightness():
    ramp = np.tile(np.linspace(100, 140, 64), (64, 1)).astype(np.uint8)
    low_contrast = np.dstack([ramp, ramp, ramp])
    brightness_before = cv2.cvtColor(low_contrast, cv2.COLOR_RGB2HSV)[..., 2].std()
    brightness_after = cv2.cvtColor(equalize(low_contrast), cv2.COLOR_RGB2HSV)[..., 2].std()
    assert brightness_after > brightness_before
