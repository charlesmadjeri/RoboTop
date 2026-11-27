import numpy as np
import pytest

import images


def flat(value, shape=(8, 8, 3)):
    return np.full(shape, value, dtype=np.uint8)


def noise(seed):
    return np.random.default_rng(seed).integers(0, 255, (8, 8, 3), dtype=np.uint8)


def test_uniform_image_is_blank():
    assert images.is_blank(flat(40))


def test_textured_image_is_not_blank():
    assert not images.is_blank(noise(1))


def test_identical_images_have_not_changed():
    assert not images.has_changed(noise(2), noise(2))


def test_different_images_have_changed():
    assert images.has_changed(noise(3), noise(4))


def test_diff_does_not_wrap_around_on_unsigned_bytes():
    assert images.mean_abs_diff(flat(250), flat(5)) == 245.0


def test_small_change_below_threshold_is_ignored():
    changed = flat(100)
    changed[0, 0, 0] = 101
    assert not images.has_changed(flat(100), changed)


def test_diff_of_different_shapes_is_refused():
    with pytest.raises(ValueError):
        images.mean_abs_diff(flat(0, (4, 4, 3)), flat(0, (8, 8, 3)))


def test_saved_image_reads_back_with_same_pixels(tmp_path):
    rgb = noise(5)
    images.save_rgb(rgb, tmp_path / "frame.png")
    assert np.array_equal(images.read(tmp_path / "frame.png")[:, :, ::-1], rgb)
