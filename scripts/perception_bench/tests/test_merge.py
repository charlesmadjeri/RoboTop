import pytest

from perception_bench.merge import merge


def run(method, frames=None):
    return {
        "frames": frames or {"nominal": 10},
        "methods": {method: {"scenarios": {}, "latency_ms": {}}},
        "motion": {method: {}},
    }


def test_merging_two_runs_keeps_every_method():
    merged = merge([run("hsv_fixed"), run("aruco")])
    assert list(merged["methods"]) == ["hsv_fixed", "aruco"]
    assert list(merged["motion"]) == ["hsv_fixed", "aruco"]


def test_frame_counts_that_differ_are_shown_as_a_range():
    merged = merge([run("hsv_fixed", {"nominal": 10, "light_dim": 5}), run("aruco", {"nominal": 9, "light_dim": 5})])
    assert merged["frames"] == {"nominal": "9 to 10", "light_dim": 5}


def test_runs_with_different_scenarios_are_refused():
    with pytest.raises(ValueError, match="scenarios"):
        merge([run("hsv_fixed"), run("aruco", {"light_dim": 10})])


def test_a_method_present_in_two_runs_is_refused():
    with pytest.raises(ValueError, match="aruco"):
        merge([run("aruco"), run("aruco")])
