import numpy as np
import pytest

from perception_bench import train_data
from perception_bench.methods import yolo
from perception_bench.truth import Target, Truth


def proposals(*rows):
    out = np.zeros((1, 6, len(rows)), np.float32)
    for column, (cx, cy, cube, box) in enumerate(rows):
        out[0, :, column] = (cx, cy, 20, 20, cube, box)
    return out


def test_letterbox_scales_to_the_long_side_and_centres_the_padding():
    canvas, scale, pad = yolo.letterbox(np.zeros((480, 640, 3), np.uint8))
    assert canvas.shape == (320, 320, 3)
    assert scale == 0.5
    assert pad == (0.0, 40.0)
    assert canvas[0, 0, 0] == yolo.PAD_VALUE
    assert canvas[160, 160, 0] == 0


def test_decode_maps_the_best_proposal_back_to_the_image():
    found = yolo.decode(proposals((100, 80, 0.4, 0.0), (160, 120, 0.9, 0.0), (10, 10, 0.1, 0.8)), 0.5, (0.0, 40.0))
    assert found["cube"] == pytest.approx((320.0, 160.0))
    assert found["box"] == pytest.approx((20.0, -60.0))


def test_decode_reports_nothing_below_the_confidence():
    assert yolo.decode(proposals((100, 80, 0.2, 0.1)), 1.0, (0.0, 0.0)) == {"cube": None, "box": None}


def test_a_missing_weights_file_names_the_training_script(monkeypatch, tmp_path):
    monkeypatch.setattr(yolo, "WEIGHTS", tmp_path / "none.onnx")
    with pytest.raises(FileNotFoundError, match="train_yolo"):
        yolo.make(np.zeros((240, 320, 3), np.uint8))


def test_labels_use_the_visible_bounds_and_skip_slivers():
    truth = Truth(
        cube=Target(400, 400, (50.0, 60.0), (40, 50, 60, 70)),
        box=Target(train_data.MIN_VISIBLE_PX - 1, 900, (200.0, 100.0), (180, 80, 220, 120)),
    )
    assert train_data.label_lines(truth, 320, 240) == ["0 0.156250 0.250000 0.062500 0.083333"]
