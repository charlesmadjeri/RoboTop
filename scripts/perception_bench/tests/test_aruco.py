import sys

import numpy as np
import pytest

from perception_bench import markers, run
from perception_bench.methods import aruco
from perception_bench.render import capture
from perception_bench.scene import Scene


def paste(image, marker_id, top, left):
    tile = markers.marker_image(marker_id)
    image[top : top + tile.shape[0], left : left + tile.shape[1]] = tile[..., None]
    return top + tile.shape[0] / 2 - 0.5, left + tile.shape[1] / 2 - 0.5


def blank():
    return np.full((480, 640, 3), 120, np.uint8)


def test_the_cube_and_box_markers_give_their_centres():
    image = blank()
    cube_row, cube_col = paste(image, markers.CUBE_ID, 60, 80)
    box_row, box_col = paste(image, markers.BOX_ID, 250, 400)
    found = aruco.make(blank())(image)
    assert found["cube"] == pytest.approx((cube_col, cube_row), abs=1.0)
    assert found["box"] == pytest.approx((box_col, box_row), abs=1.0)


def test_a_frame_without_markers_gives_nothing():
    assert aruco.make(blank())(blank()) == {"cube": None, "box": None}


def test_a_marker_with_another_id_is_ignored():
    image = blank()
    paste(image, 12, 60, 80)
    assert aruco.make(blank())(image)["cube"] is None


def test_marker_methods_cannot_be_mixed_with_the_others(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run", "--methods", "aruco", "hsv_fixed"])
    with pytest.raises(SystemExit):
        run.main()


@pytest.fixture(scope="module")
def marked_scene():
    try:
        built = Scene(640, 480, with_markers=True)
    except Exception as error:
        pytest.skip(f"no offscreen OpenGL: {error}")
    yield built
    built.close()


def test_both_markers_are_found_on_the_rendered_scene_at_the_real_camera_size(marked_scene):
    marked_scene.reset()
    marked_scene.set_cube(0.22, 0.05, 0.3)
    marked_scene.forward()
    frame = capture(marked_scene)
    found = aruco.make(frame.rgb)(frame.rgb)
    for target in ("cube", "box"):
        truth = getattr(frame.truth, target)
        assert found[target] is not None
        assert np.hypot(found[target][0] - truth.centroid[0], found[target][1] - truth.centroid[1]) < 0.5 * truth.width
