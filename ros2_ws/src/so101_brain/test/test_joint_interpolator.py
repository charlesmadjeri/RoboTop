import pytest
from so101_brain.joint_interpolator import JointInterpolator


@pytest.fixture
def interpolator():
    interp = JointInterpolator(2, max_step=0.1)
    interp.start([0.0, 0.0])
    return interp


def test_a_tick_moves_each_joint_by_at_most_max_step(interpolator):
    interpolator.target = [1.0, -1.0]
    assert interpolator.step() == pytest.approx([0.1, -0.1])


def test_a_close_target_is_reached_in_one_tick(interpolator):
    interpolator.target = [0.05, 0.0]
    assert interpolator.step() == pytest.approx([0.05, 0.0])


def test_nothing_is_sent_once_the_target_is_reached(interpolator):
    interpolator.target = [0.05, 0.0]
    interpolator.step()
    assert interpolator.step() is None


def test_hold_stops_the_motion_where_it_is(interpolator):
    interpolator.target = [1.0, 0.0]
    interpolator.step()
    interpolator.hold()
    assert interpolator.step() is None
    assert interpolator.commanded == pytest.approx([0.1, 0.0])


def test_not_ready_before_start():
    assert not JointInterpolator(2, max_step=0.1).ready
