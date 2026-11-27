import logging
from types import SimpleNamespace

import pytest
from brain_fixtures import REACHABLE_Q, URDF_PATH
from geometry_msgs.msg import PoseStamped, Vector3Stamped
from so101_brain.arm_kinematics import ArmKinematics
from so101_brain.brain_node import BrainNode
from so101_brain.joint_interpolator import JointInterpolator


@pytest.fixture(scope="module")
def kinematics():
    return ArmKinematics(URDF_PATH)


@pytest.fixture
def node(kinematics):
    interpolator = JointInterpolator(5, max_step=0.015)
    interpolator.start(REACHABLE_Q)
    fake = SimpleNamespace(
        _kinematics=kinematics,
        _interpolator=interpolator,
        _current_q=list(REACHABLE_Q),
        _teleop_active=False,
        _step_size_m=0.01,
        _max_vector=3.0,
        get_logger=lambda: logging.getLogger("brain"),
    )
    fake._clamp = lambda value: BrainNode._clamp(fake, value)
    return fake


def vector(x=0.0, y=0.0, z=0.0):
    msg = Vector3Stamped()
    msg.vector.x, msg.vector.y, msg.vector.z = x, y, z
    return msg


def target_pose(position, frame_id="base_link"):
    msg = PoseStamped()
    msg.header.frame_id = frame_id
    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = (float(v) for v in position)
    return msg


def reachable_goal(kinematics):
    return kinematics.position([0.1, -0.4, 0.8, 0.5, 0.0])


def test_a_key_press_moves_the_target_one_step_along_the_axis(node, kinematics):
    BrainNode._cb_instruction_vector(node, vector(x=1.0))
    start = kinematics.position(REACHABLE_Q)
    moved = kinematics.position(node._interpolator.target)
    assert moved[0] - start[0] == pytest.approx(0.01, abs=0.005)


def test_idle_zero_vectors_do_not_cancel_a_go_to_target_move(node, kinematics):
    BrainNode._cb_go_to_target(node, target_pose(reachable_goal(kinematics)))
    target = list(node._interpolator.target)
    BrainNode._cb_instruction_vector(node, vector())
    BrainNode._cb_instruction_vector(node, vector())
    assert node._interpolator.target == target


def test_releasing_the_keys_stops_the_motion_where_it_is(node):
    BrainNode._cb_instruction_vector(node, vector(x=1.0))
    node._interpolator.step()
    BrainNode._cb_instruction_vector(node, vector())
    assert node._interpolator.target == node._interpolator.commanded


def test_oversized_vector_components_are_clamped(node, kinematics):
    BrainNode._cb_instruction_vector(node, vector(x=1000.0))
    start = kinematics.position(REACHABLE_Q)
    moved = kinematics.position(node._interpolator.target)
    assert abs(moved[0] - start[0]) < 0.05


def test_go_to_target_accepts_a_reachable_pose(node, kinematics):
    assert BrainNode._cb_go_to_target(node, target_pose(reachable_goal(kinematics))) is True
    assert node._interpolator.target != REACHABLE_Q


def test_go_to_target_refuses_an_unreachable_pose_and_keeps_the_target(node, caplog):
    with caplog.at_level(logging.WARNING, logger="brain"):
        assert BrainNode._cb_go_to_target(node, target_pose([5.0, 0.0, 0.0])) is False
    assert node._interpolator.target == REACHABLE_Q
    assert "unreachable" in caplog.text


def test_go_to_target_names_the_frame_it_refuses(node, caplog):
    with caplog.at_level(logging.WARNING, logger="brain"):
        assert BrainNode._cb_go_to_target(node, target_pose([0.2, 0.0, 0.2], frame_id="map")) is False
    assert "frame_id" in caplog.text and "map" in caplog.text
    assert node._interpolator.target == REACHABLE_Q
