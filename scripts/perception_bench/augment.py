from collections.abc import Callable, Sequence

import cv2
import numpy as np

Perturbation = Callable[[np.ndarray, np.random.Generator], np.ndarray]


def white_balance(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    gains = rng.uniform(0.8, 1.25, size=3).astype(np.float32)
    return _to_uint8(rgb.astype(np.float32) * gains)


def gamma(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    exponent = rng.uniform(0.6, 1.8)
    lut = _to_uint8(255.0 * (np.arange(256, dtype=np.float32) / 255.0) ** exponent)
    return cv2.LUT(rgb, lut)


def desaturate(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    factor = rng.uniform(0.55, 0.95)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hsv[..., 1] = _to_uint8(hsv[..., 1].astype(np.float32) * factor)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def gaussian_noise(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    sigma = rng.uniform(2.0, 12.0)
    noise = rng.normal(0.0, sigma, size=rgb.shape).astype(np.float32)
    return _to_uint8(rgb.astype(np.float32) + noise)


def blur(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    size = int(rng.choice([3, 5]))
    return cv2.GaussianBlur(rgb, (size, size), 0)


def jpeg(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    quality = int(rng.integers(20, 61))
    ok, buffer = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("jpeg encoding failed")
    decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


def vignette(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    strength = rng.uniform(0.1, 0.4)
    height, width = rgb.shape[:2]
    ys = (np.arange(height, dtype=np.float32) - (height - 1) / 2) / (height / 2)
    xs = (np.arange(width, dtype=np.float32) - (width - 1) / 2) / (width / 2)
    radius_sq = ys[:, None] ** 2 + xs[None, :] ** 2
    falloff = 1.0 - strength * radius_sq / 2.0
    return _to_uint8(rgb.astype(np.float32) * falloff[..., None])


def _to_uint8(values: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(values), 0, 255).astype(np.uint8)


PERTURBATIONS: dict[str, Perturbation] = {
    "white_balance": white_balance,
    "gamma": gamma,
    "desaturate": desaturate,
    "gaussian_noise": gaussian_noise,
    "blur": blur,
    "jpeg": jpeg,
    "vignette": vignette,
}


def apply(rgb: np.ndarray, names: Sequence[str], rng: np.random.Generator) -> np.ndarray:
    out = rgb.copy()
    for name in names:
        out = PERTURBATIONS[name](out, rng)
    return out
