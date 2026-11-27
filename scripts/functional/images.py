from pathlib import Path

import cv2
import numpy as np

BLANK_SPREAD = 2.0
MOVED_DIFF = 0.2


def read(path):
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"cannot read image {path}")
    return image


def save_rgb(array, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(array, cv2.COLOR_RGB2BGR))


def spread(image):
    return float(np.std(image))


def is_blank(image, min_spread=BLANK_SPREAD):
    return spread(image) < min_spread


def mean_abs_diff(first, second):
    if first.shape != second.shape:
        raise ValueError(f"shapes differ: {first.shape} and {second.shape}")
    return float(np.mean(np.abs(first.astype(np.int16) - second.astype(np.int16))))


def has_changed(first, second, min_diff=MOVED_DIFF):
    return mean_abs_diff(first, second) >= min_diff
