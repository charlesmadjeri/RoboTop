import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

REPO = Path(__file__).resolve().parents[2]
PERCEPTION_PACKAGE = REPO / "ros2_ws" / "src" / "so101_perception"
if str(PERCEPTION_PACKAGE) not in sys.path:
    sys.path.insert(0, str(PERCEPTION_PACKAGE))
