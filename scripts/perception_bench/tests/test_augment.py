import cv2
import numpy as np
import pytest

from perception_bench.augment import PERTURBATIONS, apply

NAMES = sorted(PERTURBATIONS)


def _noisy(seed: int = 0, shape=(48, 64, 3)) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, size=shape, dtype=np.uint8)


def _saturation(rgb: np.ndarray) -> float:
    return float(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 1].mean())


@pytest.mark.parametrize("name", NAMES)
def test_keeps_shape_and_dtype(name):
    img = _noisy()
    out = PERTURBATIONS[name](img, np.random.default_rng(3))
    assert out.shape == img.shape
    assert out.dtype == np.uint8


@pytest.mark.parametrize("name", NAMES)
def test_does_not_modify_input(name):
    img = _noisy()
    before = img.copy()
    PERTURBATIONS[name](img, np.random.default_rng(4))
    assert np.array_equal(img, before)


@pytest.mark.parametrize("name", NAMES)
def test_same_seed_same_image(name):
    img = _noisy()
    a = PERTURBATIONS[name](img, np.random.default_rng(7))
    b = PERTURBATIONS[name](img, np.random.default_rng(7))
    assert np.array_equal(a, b)


@pytest.mark.parametrize("name", NAMES)
def test_different_seeds_different_images(name):
    img = _noisy()
    outs = [PERTURBATIONS[name](img, np.random.default_rng(seed)) for seed in range(5)]
    assert any(not np.array_equal(outs[0], other) for other in outs[1:])


def test_white_balance_changes_channel_means():
    img = np.full((32, 32, 3), 128, np.uint8)
    out = PERTURBATIONS["white_balance"](img, np.random.default_rng(1))
    assert not np.allclose(out.mean(axis=(0, 1)), img.mean(axis=(0, 1)), atol=0.5)


def test_desaturate_lowers_mean_saturation():
    img = _noisy()
    out = PERTURBATIONS["desaturate"](img, np.random.default_rng(2))
    assert _saturation(out) < _saturation(img)


def test_gaussian_noise_raises_variance_of_flat_image():
    img = np.full((32, 32, 3), 128, np.uint8)
    out = PERTURBATIONS["gaussian_noise"](img, np.random.default_rng(5))
    assert img.var() == 0
    assert out.var() > 1


def test_blur_lowers_variance_of_noisy_image():
    img = _noisy()
    out = PERTURBATIONS["blur"](img, np.random.default_rng(6))
    assert out.var() < img.var()


def test_apply_empty_list_returns_equal_copy():
    img = _noisy()
    out = apply(img, [], np.random.default_rng(0))
    assert np.array_equal(out, img)
    assert out is not img


def test_apply_chains_named_perturbations():
    img = _noisy()
    out = apply(img, ["blur", "vignette"], np.random.default_rng(0))
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    assert not np.array_equal(out, img)
