import math

JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
GRIPPER = "gripper"
GRIPPER_RANGE_RAD = (-0.174533, 1.74533)


def gripper_pct_to_rad(pct: float) -> float:
    low, high = GRIPPER_RANGE_RAD
    return low + min(max(pct, 0.0), 100.0) / 100.0 * (high - low)


def observation_to_positions(observation: dict) -> list[float]:
    return [
        gripper_pct_to_rad(observation[f"{name}.pos"]) if name == GRIPPER else observation[f"{name}.pos"]
        for name in JOINT_NAMES
    ]


def command_to_action(names: list[str], positions: list[float]) -> dict:
    return {
        f"{name}.pos": float(position)
        for name, position in zip(names, positions)
        if name in JOINT_NAMES and math.isfinite(position)
    }
