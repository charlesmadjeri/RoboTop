import numpy as np

CUBE_CENTER_HEIGHT = 0.025
DROP_BOX_BLOB_HEIGHT = 0.025
PARALLEL_LIMIT = 1e-9


def pixel_to_ray(u: float, v: float, K: np.ndarray) -> np.ndarray:
    direction = np.linalg.solve(K, np.array([u, v, 1.0]))
    return direction / np.linalg.norm(direction)


def project_to_plane(u: float, v: float, K: np.ndarray, robot_from_camera: np.ndarray, height: float) -> np.ndarray | None:
    origin = robot_from_camera[:3, 3]
    direction = robot_from_camera[:3, :3] @ pixel_to_ray(u, v, K)
    if abs(direction[2]) < PARALLEL_LIMIT:
        return None
    distance = (height - origin[2]) / direction[2]
    if distance <= 0:
        return None
    return origin + distance * direction
