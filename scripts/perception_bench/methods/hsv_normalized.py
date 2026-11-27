import cv2
import numpy as np

from so101_perception.detection import find_cube, find_drop_box

CLIP_LIMIT = 2.0
TILES = (4, 4)


def make(empty: np.ndarray):
    return detect


def detect(rgb: np.ndarray) -> dict:
    balanced = normalize(rgb)
    return {"cube": find_cube(balanced), "box": find_drop_box(balanced)}


def normalize(rgb: np.ndarray) -> np.ndarray:
    return equalize(white_balance(rgb))


def white_balance(rgb: np.ndarray) -> np.ndarray:
    floor = np.median(rgb.reshape(-1, 3), axis=0).astype(np.float32)
    gains = floor.mean() / np.maximum(floor, 1.0)
    return np.clip(rgb * gains, 0, 255).astype(np.uint8)


def equalize(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hsv[..., 2] = cv2.createCLAHE(CLIP_LIMIT, TILES).apply(hsv[..., 2])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
