import numpy as np
import pytest

from perception_bench.render import capture
from so101_perception.detection import find_cube


def place_cube(scene, x=0.22, y=0.05, yaw=0.3):
    scene.reset()
    scene.set_cube(x, y, yaw)
    scene.forward()


def test_parked_cube_is_absent_and_the_box_is_present(scene):
    scene.reset()
    truth = capture(scene).truth
    assert truth.cube.full_px == 0 and truth.cube.centroid is None
    assert truth.box.visible_px > 0


def test_placed_cube_is_fully_visible_and_matches_the_detector_centroid(scene):
    place_cube(scene)
    frame = capture(scene)
    assert frame.truth.cube.visible_fraction == 1.0
    found = find_cube(frame.rgb)
    assert np.hypot(found[0] - frame.truth.cube.centroid[0], found[1] - frame.truth.cube.centroid[1]) < 1.5


def test_decoys_change_the_picture_but_not_the_truth(scene):
    place_cube(scene)
    before = capture(scene)
    scene.place_decoy(0, (0.3, -0.1), 0.03, (1, 0, 0, 1))
    scene.forward()
    after = capture(scene)
    assert after.truth == before.truth
    assert not np.array_equal(after.rgb, before.rgb)


def test_hidden_box_is_absent_and_reset_brings_it_back(scene):
    scene.reset()
    scene.hide_box()
    scene.forward()
    assert capture(scene).truth.box.visible_px == 0
    scene.reset()
    assert capture(scene).truth.box.visible_px > 0


def test_lighting_scenarios_are_undone_by_reset(scene):
    place_cube(scene)
    nominal = capture(scene).rgb
    scene.scale_lighting(0.2, 0.2, 0.2)
    scene.forward()
    assert capture(scene).rgb.mean() < nominal.mean()
    place_cube(scene)
    assert np.array_equal(capture(scene).rgb, nominal)


def test_an_arm_in_front_of_the_cube_lowers_its_visible_fraction_but_not_its_full_size(scene):
    place_cube(scene)
    free = capture(scene).truth.cube
    rng = np.random.default_rng(0)
    low, high = scene.arm_ranges().T
    for _ in range(400):
        scene.set_arm(rng.uniform(low, high))
        scene.forward()
        truth = capture(scene).truth.cube
        if truth.visible_fraction < 0.9:
            assert truth.full_px == free.full_px
            return
    pytest.fail("no arm pose hid part of the cube")
