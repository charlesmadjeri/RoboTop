from types import SimpleNamespace

import numpy as np
import pytest

from so101_perception.image_message import rgb_from_image


def message(pixels, padding=0, encoding="rgb8"):
    height, width = pixels.shape[:2]
    rows = np.pad(pixels.reshape(height, width * 3), ((0, 0), (0, padding)))
    return SimpleNamespace(encoding=encoding, height=height, width=width, step=width * 3 + padding, data=rows.tobytes())


def test_image_message_gives_back_the_same_pixels():
    pixels = np.random.default_rng(0).integers(0, 255, (4, 5, 3), dtype=np.uint8)
    assert np.array_equal(rgb_from_image(message(pixels)), pixels)


def test_row_padding_is_dropped():
    pixels = np.random.default_rng(0).integers(0, 255, (4, 5, 3), dtype=np.uint8)
    assert np.array_equal(rgb_from_image(message(pixels, padding=6)), pixels)


def test_encoding_other_than_rgb8_is_rejected():
    pixels = np.zeros((2, 2, 3), np.uint8)
    with pytest.raises(ValueError, match="bgr8"):
        rgb_from_image(message(pixels, encoding="bgr8"))
