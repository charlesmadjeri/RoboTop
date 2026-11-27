from pathlib import Path

import cv2
import mujoco
import numpy as np

DICTIONARY = cv2.aruco.DICT_4X4_50
CUBE_ID = 7
BOX_ID = 3
PIXELS = 160
MARGIN = PIXELS // 6
PLATES = (
    ("cube_marker", CUBE_ID, "pickup_object", 0.046, (0.0, 0.0, 0.0255)),
    ("box_marker", BOX_ID, "drop_box", 0.055, (-0.015, -0.015, 0.0045)),
)


def marker_image(marker_id: int) -> np.ndarray:
    dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARY)
    marker = cv2.aruco.generateImageMarker(dictionary, marker_id, PIXELS)
    return cv2.copyMakeBorder(marker, MARGIN, MARGIN, MARGIN, MARGIN, cv2.BORDER_CONSTANT, value=255)


def add_markers(spec: mujoco.MjSpec, folder: Path) -> None:
    for name, marker_id, body, size, position in PLATES:
        path = folder / f"{name}.png"
        cv2.imwrite(str(path), marker_image(marker_id))
        _attach(spec, name, str(path), body, size / 2, position)


def _attach(spec: mujoco.MjSpec, name: str, path: str, body: str, half: float, position: tuple) -> None:
    spec.add_texture(name=name, type=mujoco.mjtTexture.mjTEXTURE_2D, file=path)
    material = spec.add_material(name=name)
    material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = name
    spec.body(body).add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[half, half, 0.0004],
        pos=list(position),
        material=name,
        group=2,
        contype=0,
        conaffinity=0,
        density=0,
    )
