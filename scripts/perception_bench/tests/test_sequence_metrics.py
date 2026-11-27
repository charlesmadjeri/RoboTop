from perception_bench.sequence_metrics import score_sequence
from helpers import detection, frame, target

HERE = (100.0, 50.0)


def present():
    return frame(target())


def gone():
    return frame(target(0, 400))


def test_time_to_lost_counts_frames_from_the_first_absent_frame():
    frames = [present(), present(), gone(), gone(), gone(), gone(), gone(), gone(), gone()]
    found = [detection(HERE)] * 2 + [detection()] * 7
    result = score_sequence(frames, found, timeout=3)
    assert result["time_to_lost"] == 3
    assert result["held_frames"] == 3


def test_a_detector_that_keeps_seeing_something_is_never_declared_lost():
    frames = [present(), gone(), gone(), gone()]
    found = [detection(HERE)] * 4
    result = score_sequence(frames, found, timeout=1)
    assert result["time_to_lost"] is None
    assert result["false_positive"] == 1.0


def test_reacquire_counts_frames_between_reentry_and_the_first_accurate_detection():
    frames = [present(), gone(), present(), present(), present()]
    found = [detection(HERE), detection(), detection(), detection((300.0, 50.0)), detection(HERE)]
    assert score_sequence(frames, found)["reacquire"] == 2


def test_reacquire_is_none_when_the_object_is_never_found_again():
    frames = [present(), gone(), present(), present()]
    found = [detection(HERE), detection(), detection(), detection()]
    assert score_sequence(frames, found)["reacquire"] is None


def test_a_sequence_without_exit_has_no_loss_or_reentry_figures():
    frames = [present(), present()]
    result = score_sequence(frames, [detection(HERE)] * 2)
    assert result["time_to_lost"] is None
    assert result["reacquire"] is None
