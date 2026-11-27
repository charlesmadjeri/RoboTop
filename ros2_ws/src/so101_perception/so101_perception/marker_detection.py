import cv2
import numpy as np

DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
PARAMETERS = cv2.aruco.DetectorParameters()
PARAMETERS.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
DETECTOR = cv2.aruco.ArucoDetector(DICTIONARY, PARAMETERS)


def find_marker_corners(rgb: np.ndarray, marker_id: int) -> np.ndarray | None:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    corners, ids, _ = DETECTOR.detectMarkers(gray)
    if ids is None:
        return None
    for marker_corners, found_id in zip(corners, ids.ravel()):
        if found_id == marker_id:
            return marker_corners.reshape(4, 2)
    return None
