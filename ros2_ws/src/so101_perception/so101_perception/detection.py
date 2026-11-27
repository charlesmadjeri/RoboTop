import cv2
import numpy as np

Pixel = tuple[float, float]

CUBE_HSV_RANGES = (
    ((160, 120, 60), (180, 255, 255)),
    ((0, 120, 60), (4, 255, 255)),
)
BOX_HSV_RANGES = (((5, 100, 40), (20, 255, 255)),)
MIN_AREA_RATIO = 0.0008
KERNEL = np.ones((3, 3), np.uint8)


def find_cube(rgb: np.ndarray) -> Pixel | None:
    return _find(rgb, CUBE_HSV_RANGES)


def find_drop_box(rgb: np.ndarray) -> Pixel | None:
    return _find(rgb, BOX_HSV_RANGES)


def _find(rgb: np.ndarray, ranges) -> Pixel | None:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = _threshold(hsv, ranges)
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)
    closed = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, KERNEL)
    return _largest_blob_center(closed, MIN_AREA_RATIO * mask.size)


def _threshold(hsv: np.ndarray, ranges) -> np.ndarray:
    masks = [cv2.inRange(hsv, np.array(low, np.uint8), np.array(high, np.uint8)) for low, high in ranges]
    return np.bitwise_or.reduce(masks)


def _largest_blob_center(mask: np.ndarray, min_area: float) -> Pixel | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    blob = max(contours, key=cv2.contourArea)
    moments = cv2.moments(blob)
    if moments["m00"] < min_area:
        return None
    return moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]
