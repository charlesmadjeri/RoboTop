import cv2
import numpy as np
import pytest

from perception_bench.real import load, score


def fixed(cube, box):
    return lambda frame: {"cube": cube, "box": box}


FRAMES = [np.zeros((48, 64, 3), np.uint8)] * 4


def test_a_detection_near_the_cube_is_a_hit():
    result = score(fixed((30.0, 20.0), None), FRAMES, (32.0, 21.0))
    assert result["cube_hits"] == 4
    assert result["cube_median_error"] < 3


def test_a_detection_far_from_the_cube_is_not_a_hit():
    assert score(fixed((5.0, 5.0), None), FRAMES, (60.0, 40.0))["cube_hits"] == 0


def test_a_missing_cube_gives_no_error():
    result = score(fixed(None, None), FRAMES, (10.0, 10.0))
    assert result["cube_hits"] == 0
    assert result["cube_median_error"] is None


def test_every_reported_box_is_counted():
    assert score(fixed(None, (1.0, 1.0)), FRAMES, (10.0, 10.0))["box_detections"] == 4


def test_load_returns_rgb_frames_in_name_order(tmp_path):
    cv2.imwrite(str(tmp_path / "b.png"), np.full((4, 4, 3), (255, 0, 0), np.uint8))
    cv2.imwrite(str(tmp_path / "a.png"), np.full((4, 4, 3), (0, 0, 255), np.uint8))
    frames = load(tmp_path)
    assert frames[0][0, 0].tolist() == [255, 0, 0]
    assert frames[1][0, 0].tolist() == [0, 0, 255]


def test_an_empty_folder_is_refused(tmp_path):
    with pytest.raises(ValueError, match="no PNG frames"):
        load(tmp_path)


def test_an_unreadable_image_names_the_file(tmp_path):
    (tmp_path / "broken.png").write_text("not an image")
    with pytest.raises(ValueError, match="broken.png"):
        load(tmp_path)
