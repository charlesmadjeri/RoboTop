import numpy as np

from so101_perception.detection import find_cube, find_drop_box
from so101_perception.projection import CUBE_CENTER_HEIGHT, DROP_BOX_BLOB_HEIGHT, project_to_plane

DROP_BOX_BASE_HEIGHT = 0.0


def locate(
    rgb: np.ndarray,
    K: np.ndarray,
    robot_from_camera: np.ndarray,
    cube_height: float = CUBE_CENTER_HEIGHT,
    box_height: float = DROP_BOX_BLOB_HEIGHT,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    cube = project_detection(find_cube(rgb), K, robot_from_camera, cube_height)
    box = project_detection(find_drop_box(rgb), K, robot_from_camera, box_height)
    if box is not None:
        box[2] = DROP_BOX_BASE_HEIGHT
    return cube, box


def project_detection(pixel, K, robot_from_camera, height):
    if pixel is None:
        return None
    return project_to_plane(pixel[0], pixel[1], K, robot_from_camera, height)
