import json
from pathlib import Path

import cv2
import numpy as np

DATA = Path(__file__).parent / "data"
EXPECTED = json.loads((DATA / "expected.json").read_text())
CAMERA = json.loads((DATA / "camera.json").read_text())
K = np.array(CAMERA["K"])
ROBOT_FROM_CAMERA = np.array(CAMERA["T_world_camera"])
DROP_BOX_BASE_XY = (0.05, 0.25)


def load(name):
    return cv2.cvtColor(cv2.imread(str(DATA / f"{name}.png")), cv2.COLOR_BGR2RGB)
