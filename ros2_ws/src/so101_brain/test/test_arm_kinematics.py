import numpy as np
import pytest
from so101_brain.arm_kinematics import ArmKinematics
from brain_fixtures import REACHABLE_Q, URDF_PATH


@pytest.fixture(scope="module")
def kinematics():
    return ArmKinematics(URDF_PATH)


def test_ik_reaches_the_position_forward_kinematics_gave(kinematics):
    target = [float(v) for v in kinematics.position(REACHABLE_Q)]
    solution = kinematics.solve(target, [0.0] * 5)
    assert kinematics.error(solution, target) < 0.01


def test_ik_returns_the_five_arm_joints(kinematics):
    target = [float(v) for v in kinematics.position(REACHABLE_Q)]
    assert len(kinematics.solve(target, [0.0] * 5)) == 5


def test_a_seed_outside_the_joint_limits_is_clamped_not_refused(kinematics):
    target = [float(v) for v in kinematics.position(REACHABLE_Q)]
    kinematics.solve(target, [10.0] * 5)


def test_an_unreachable_target_leaves_a_large_error(kinematics):
    target = [5.0, 0.0, 0.0]
    solution = kinematics.solve(target, [0.0] * 5)
    assert kinematics.error(solution, target) > 1.0


def test_pose_orientation_is_a_unit_quaternion(kinematics):
    _, quat = kinematics.pose(REACHABLE_Q)
    assert np.isclose(np.linalg.norm(quat), 1.0)


def test_pose_position_matches_forward_kinematics(kinematics):
    position, _ = kinematics.pose(REACHABLE_Q)
    assert np.allclose(position, kinematics.forward(REACHABLE_Q)[:3, 3])
