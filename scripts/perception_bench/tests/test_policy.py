from perception_bench.policy import HELD, LOST, TRACKING, HoldPolicy


def test_a_fresh_detection_is_published_as_tracking():
    reading = HoldPolicy().update((10.0, 20.0))
    assert (reading.state, reading.position, reading.age) == (TRACKING, (10.0, 20.0), 0)


def test_the_last_position_is_held_with_its_age_until_the_timeout():
    policy = HoldPolicy(timeout=3)
    policy.update((10.0, 20.0))
    readings = [policy.update(None) for _ in range(3)]
    assert [r.state for r in readings] == [HELD, HELD, HELD]
    assert [r.age for r in readings] == [1, 2, 3]
    assert all(r.position == (10.0, 20.0) for r in readings)


def test_the_object_is_declared_lost_after_the_timeout_and_nothing_is_published():
    policy = HoldPolicy(timeout=2)
    policy.update((10.0, 20.0))
    policy.update(None)
    policy.update(None)
    reading = policy.update(None)
    assert (reading.state, reading.position) == (LOST, None)


def test_nothing_is_held_before_the_first_detection():
    assert HoldPolicy().update(None).state == LOST


def test_a_new_detection_after_loss_tracks_again():
    policy = HoldPolicy(timeout=1)
    policy.update((1.0, 1.0))
    policy.update(None)
    policy.update(None)
    assert policy.update((5.0, 5.0)).state == TRACKING
