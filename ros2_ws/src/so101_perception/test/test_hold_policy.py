import pytest

from so101_perception.hold_policy import HELD, LOST, TRACKING, HoldPolicy


def test_a_detection_is_tracked_with_age_zero():
    reading = HoldPolicy[str]().update("here")
    assert (reading.state, reading.value, reading.age) == (TRACKING, "here", 0)


def test_nothing_seen_from_the_start_is_lost():
    reading = HoldPolicy[str]().update(None)
    assert (reading.state, reading.value) == (LOST, None)


def test_the_last_value_is_held_for_the_given_number_of_frames_then_lost():
    policy = HoldPolicy[str](hold_frames=5)
    policy.update("here")
    held = [policy.update(None) for _ in range(5)]
    assert [reading.state for reading in held] == [HELD] * 5
    assert [reading.age for reading in held] == [1, 2, 3, 4, 5]
    assert {reading.value for reading in held} == {"here"}
    assert policy.update(None).state == LOST


def test_a_new_detection_after_a_hold_resets_the_age():
    policy = HoldPolicy[str](hold_frames=2)
    policy.update("old")
    policy.update(None)
    assert policy.update("new").age == 0
    assert policy.update(None).value == "new"


def test_a_lost_target_stays_lost_until_seen_again():
    policy = HoldPolicy[str](hold_frames=1)
    policy.update("here")
    policy.update(None)
    for _ in range(3):
        assert policy.update(None).state == LOST
    assert policy.update("back").state == TRACKING


def test_zero_hold_frames_never_holds():
    policy = HoldPolicy[str](hold_frames=0)
    policy.update("here")
    assert policy.update(None).state == LOST


def test_a_negative_hold_is_refused():
    with pytest.raises(ValueError, match="zero or more"):
        HoldPolicy[str](hold_frames=-1)
