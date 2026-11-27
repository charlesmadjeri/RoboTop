import json
from pathlib import Path

PROFILES_DIR = Path(__file__).resolve().parent.parent / "profiles"
DRIVER_KEYS = {"camera_index", "camera_width", "camera_height", "camera_fourcc", "camera_fov_deg"}
ALLOWED = DRIVER_KEYS | {"camera_pose_file"}


def available(directory: Path = PROFILES_DIR) -> list[str]:
    return sorted(path.stem for path in directory.glob("*.json"))


def load(name: str, directory: Path = PROFILES_DIR) -> dict:
    path = directory / f"{name}.json"
    if not path.is_file():
        raise ValueError(f"profile '{name}' not found, available: {', '.join(available(directory)) or 'none'}")
    settings = json.loads(path.read_text(encoding="utf-8"))
    unknown = sorted(set(settings) - ALLOWED)
    if unknown:
        raise ValueError(f"profile '{name}' has unknown keys: {', '.join(unknown)}")
    return settings


def launch_arguments(settings: dict, overrides: dict) -> list[str]:
    merged = {**settings, **{key: value for key, value in overrides.items() if value not in (None, "")}}
    if "camera_pose_file" in merged:
        merged["camera_pose_file"] = str(PROFILES_DIR.parent / merged["camera_pose_file"])
    return [f"{key}:={value}" for key, value in merged.items()]
