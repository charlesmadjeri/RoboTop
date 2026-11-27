import numpy as np
from cube_scene import pose_error
from sim_arm import SimArm, SimRig

from so101_perception.arm_poses import PRESET_POSES
from so101_perception.arm_session import collect_samples
from so101_perception.calibrate_from_arm import evaluate, refusal
from so101_perception.detection import find_cube
from so101_perception.projection import project_to_plane

GRASP_OFFSET = np.array([0.0, 0.01, 0.02])
SPEED = 0.8
TABLE_GRID = [(x, y) for x in (0.15, 0.25, 0.35) for y in (-0.15, 0.0, 0.15)]


def table_error(arm: SimArm, found: np.ndarray) -> float:
    errors = []
    for x, y in TABLE_GRID:
        u, v = arm.pixel_of(np.array([x, y, 0.025]))
        errors.append(np.linalg.norm(project_to_plane(u, v, arm.K, found, 0.025)[:2] - [x, y]))
    return float(np.max(errors))


def test_the_command_sequence_on_the_simulated_arm_recovers_the_camera_pose(arm):
    rig = SimRig(arm, GRASP_OFFSET, start=PRESET_POSES[0])
    samples = collect_samples(rig, rig, find_cube, PRESET_POSES, SPEED, report=lambda line: None)
    assert len(samples.kept) >= 10
    evaluation = evaluate(samples, arm.K)
    translation, angle = pose_error(evaluation.calibration.robot_from_camera, arm.robot_from_camera)
    assert table_error(arm, evaluation.calibration.robot_from_camera) < 0.01
    assert translation < 0.05 and angle < 3.0
    assert refusal(evaluation, max_error_cm=1.0) is None


def test_the_moves_are_slow_steps_that_never_jump_between_poses(arm):
    rig = SimRig(arm, GRASP_OFFSET, start=PRESET_POSES[0])
    collect_samples(rig, rig, find_cube, PRESET_POSES[:3], SPEED, report=lambda line: None)
    steps = np.abs(np.diff(np.vstack(rig.commands), axis=0)).max()
    assert steps <= SPEED / 20.0 + 1e-9
