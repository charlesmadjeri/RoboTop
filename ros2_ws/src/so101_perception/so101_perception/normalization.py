import cv2
import numpy as np

CLIP_LIMIT = 2.0
TILES = (4, 4)
MAX_WIDTH = 640


def white_balance(rgb: np.ndarray) -> np.ndarray:
    floor = np.median(rgb.reshape(-1, 3), axis=0).astype(np.float32)
    gains = floor.mean() / np.maximum(floor, 1.0)
    return np.clip(rgb * gains, 0, 255).astype(np.uint8)


def equalize(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hsv[..., 2] = cv2.createCLAHE(CLIP_LIMIT, TILES).apply(hsv[..., 2])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def normalize(rgb: np.ndarray) -> np.ndarray:
    return equalize(white_balance(rgb))


def shrink_to_width(rgb: np.ndarray, width: int = MAX_WIDTH) -> tuple[np.ndarray, float]:
    full_width = rgb.shape[1]
    if full_width <= width:
        return rgb, 1.0
    height = round(rgb.shape[0] * width / full_width)
    return cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA), full_width / width
