from pathlib import Path

URDF_PATH = str(Path(__file__).resolve().parents[2] / "so101_description" / "urdf" / "so101.urdf")
REACHABLE_Q = [0.3, -0.4, 0.8, 0.5, 0.0]
