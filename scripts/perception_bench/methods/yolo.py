import os
from pathlib import Path

import cv2
import numpy as np

from perception_bench.truth import Point

SIZE = 320
CONFIDENCE = 0.25
PAD_VALUE = 114
CLASSES = ("cube", "box")
WEIGHTS = Path(os.environ.get("ROBOTOP_YOLO_ONNX", Path.home() / ".cache" / "robotop" / "yolo_cube_box.onnx"))


def make(empty: np.ndarray):
    if not WEIGHTS.exists():
        raise FileNotFoundError(f"{WEIGHTS} is missing, train it with perception_bench/train_yolo.py")
    net = cv2.dnn.readNetFromONNX(str(WEIGHTS))

    def detect(rgb: np.ndarray) -> dict:
        canvas, scale, pad = letterbox(rgb)
        net.setInput(cv2.dnn.blobFromImage(canvas, 1 / 255, (SIZE, SIZE)))
        return decode(net.forward(), scale, pad)

    return detect


def letterbox(rgb: np.ndarray) -> tuple[np.ndarray, float, tuple[float, float]]:
    height, width = rgb.shape[:2]
    scale = SIZE / max(height, width)
    resized = cv2.resize(rgb, (round(width * scale), round(height * scale)))
    pad = ((SIZE - resized.shape[1]) / 2, (SIZE - resized.shape[0]) / 2)
    canvas = np.full((SIZE, SIZE, 3), PAD_VALUE, np.uint8)
    canvas[int(pad[1]) : int(pad[1]) + resized.shape[0], int(pad[0]) : int(pad[0]) + resized.shape[1]] = resized
    return canvas, scale, pad


def decode(output: np.ndarray, scale: float, pad: tuple[float, float]) -> dict:
    proposals = output[0].T
    found: dict[str, Point | None] = {}
    for index, name in enumerate(CLASSES):
        scores = proposals[:, 4 + index]
        best = int(scores.argmax())
        found[name] = None
        if scores[best] >= CONFIDENCE:
            found[name] = ((proposals[best, 0] - pad[0]) / scale, (proposals[best, 1] - pad[1]) / scale)
    return found
