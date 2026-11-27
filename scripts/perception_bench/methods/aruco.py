import cv2
import numpy as np

from perception_bench import markers
from perception_bench.truth import Point


def make(empty: np.ndarray):
    dictionary = cv2.aruco.getPredefinedDictionary(markers.DICTIONARY)
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())

    def detect(rgb: np.ndarray) -> dict:
        corners, ids, _ = detector.detectMarkers(rgb)
        centres = _centres(corners, ids)
        return {"cube": centres.get(markers.CUBE_ID), "box": centres.get(markers.BOX_ID)}

    return detect


def _centres(corners, ids) -> dict[int, Point]:
    if ids is None:
        return {}
    return {int(i): tuple(c.reshape(4, 2).mean(axis=0)) for i, c in zip(ids.ravel(), corners)}
