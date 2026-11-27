import pytest
from helpers import BLANK, detection, frame, target

from perception_bench import report, report_motion, run
from perception_bench.methods import METHODS

HERE = (100.0, 50.0)
BOX = (200.0, 160.0)


@pytest.fixture
def fake_method(monkeypatch):
    monkeypatch.setitem(METHODS, "fake", lambda empty: lambda rgb: detection(HERE, BOX))


def test_evaluate_scores_every_scenario_and_sequence_for_every_method(fake_method):
    static = {"nominal": [frame(target())]}
    moving = {"motion": [frame(target()), frame(target(0, 400)), frame(target())]}
    results = run.evaluate(["fake"], static, moving, BLANK)
    cells = results["methods"]["fake"]["scenarios"]
    assert set(cells) == {"nominal", "motion"}
    assert cells["nominal"]["cube"]["recall"] == 1.0
    assert cells["motion"]["cube"]["false_positive"] == 1.0
    assert set(results["motion"]["fake"]) == {"motion"}
    assert results["frames"] == {"nominal": 1, "motion": 3}
    assert set(results["methods"]["fake"]["latency_ms"]) == {"320x240", "640x480"}


def test_evaluate_output_feeds_both_report_renderers(fake_method):
    results = run.evaluate(["fake"], {"nominal": [frame(target())]}, {"motion": [frame(target())]}, BLANK)
    assert "| nominal | 100% |" in report.render(results)
    assert "### Frames to re-acquire" in report_motion.render(results["motion"])


def test_a_detector_that_never_sees_the_cube_is_scored_as_never_found(fake_method, monkeypatch):
    monkeypatch.setitem(METHODS, "blind", lambda empty: lambda rgb: detection())
    frames = [frame(target()), frame(target(0, 400)), frame(target())]
    results = run.evaluate(["blind"], {"nominal": frames[:1]}, {"motion": frames}, BLANK)
    assert results["methods"]["blind"]["scenarios"]["nominal"]["cube"]["recall"] == 0.0
    assert results["motion"]["blind"]["motion"]["reacquire"] is None
    assert "never" in report_motion.render(results["motion"])
