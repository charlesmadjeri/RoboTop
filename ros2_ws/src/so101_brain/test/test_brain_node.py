import logging
import sys
import types
from types import SimpleNamespace

import pytest
from geometry_msgs.msg import Vector3Stamped

try:
    import so101_interfaces.srv  # noqa: F401
except ImportError:
    srv = types.ModuleType("so101_interfaces.srv")
    srv.GoToTarget = object
    sys.modules["so101_interfaces"] = types.ModuleType("so101_interfaces")
    sys.modules["so101_interfaces.srv"] = srv

from so101_brain.arm_kinematics import ArmKinematics
from so101_brain.brain_node import BrainNode
from so101_brain.joint_interpolator import JointInterpolator
from brain_fixtures import REACHABLE_Q, URDF_PATH


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


def request(position, frame_id="base_link"):
    req = SimpleNamespace(target_pose=SimpleNamespace(
        header=SimpleNamespace(frame_id=frame_id),
        pose=SimpleNamespace(position=SimpleNamespace(x=position[0], y=position[1], z=position[2])),
    ))
    return req, SimpleNamespace(success=None, message="")


def test_a_key_press_moves_the_target_one_step_along_the_axis(node, kinematics):
    BrainNode._cb_instruction_vector(node, vector(x=1.0))
    start = kinematics.position(REACHABLE_Q)
    moved = kinematics.position(node._interpolator.target)
    assert moved[0] - start[0] == pytest.approx(0.01, abs=0.005)


def test_idle_zero_vectors_do_not_cancel_a_go_to_target_move(node, kinematics):
    goal = [float(v) for v in kinematics.position([0.1, -0.4, 0.8, 0.5, 0.0])]
    req, res = request(goal)
    BrainNode._cb_go_to_target(node, req, res)
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
    goal = [float(v) for v in kinematics.position([0.1, -0.4, 0.8, 0.5, 0.0])]
    req, res = request(goal)
    assert BrainNode._cb_go_to_target(node, req, res).success is True
    assert node._interpolator.target != REACHABLE_Q


def test_go_to_target_refuses_an_unreachable_pose_and_keeps_the_target(node):
    req, res = request([5.0, 0.0, 0.0])
    assert BrainNode._cb_go_to_target(node, req, res).success is False
    assert node._interpolator.target == REACHABLE_Q


def test_go_to_target_names_the_frame_it_refuses(node):
    req, res = request([0.2, 0.0, 0.2], frame_id="map")
    result = BrainNode._cb_go_to_target(node, req, res)
    assert result.success is False
    assert "frame_id" in result.message and "map" in result.message
