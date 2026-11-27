import pytest

from perception_bench import metrics
from helpers import detection, frame, target


def test_detection_inside_half_the_target_width_is_a_hit():
    assert metrics.is_hit(target(400, 400), (109.0, 50.0))


def test_detection_beyond_half_the_target_width_is_a_miss():
    assert not metrics.is_hit(target(400, 400), (111.0, 50.0))


def test_hit_threshold_follows_the_full_size_not_the_visible_part():
    partly_hidden = target(visible=50, full=400)
    assert metrics.is_hit(partly_hidden, (109.0, 50.0))


def test_no_detection_is_a_miss():
    assert not metrics.is_hit(target(), None)


def test_recall_counts_only_present_frames():
    frames = [frame(target()), frame(target()), frame(target(0, 400))]
    found = [detection((100.0, 50.0)), detection((160.0, 50.0)), detection()]
    result = metrics.score(frames, found, "cube")
    assert result["recall"] == 0.5
    assert result["n_eval"] == 2
    assert result["n_absent"] == 1


def test_error_percentiles_use_detected_frames_only():
    frames = [frame(target()) for _ in range(3)]
    found = [detection((100.0, 50.0)), detection((103.0, 50.0)), detection()]
    result = metrics.score(frames, found, "cube")
    assert result["median_px"] == pytest.approx(1.5)
    assert result["p95_px"] == pytest.approx(2.85)


def test_detection_on_an_absent_target_is_a_false_positive():
    frames = [frame(target(0, 0)), frame(target(0, 0)), frame(target(0, 400))]
    found = [detection((10.0, 10.0)), detection(), detection((10.0, 10.0))]
    result = metrics.score(frames, found, "cube")
    assert result["false_positive"] == pytest.approx(2 / 3)
    assert result["recall"] is None


def test_a_few_visible_pixels_are_neither_present_nor_absent():
    frames = [frame(target(visible=5, full=400))]
    result = metrics.score(frames, [detection((100.0, 50.0))], "cube")
    assert result["n_eval"] == 0
    assert result["n_absent"] == 0


def test_box_is_scored_independently_from_the_cube():
    frames = [frame(target())]
    found = [detection(cube=(100.0, 50.0), box=(0.0, 0.0))]
    assert metrics.score(frames, found, "cube")["recall"] == 1.0
    assert metrics.score(frames, found, "box")["recall"] == 0.0
