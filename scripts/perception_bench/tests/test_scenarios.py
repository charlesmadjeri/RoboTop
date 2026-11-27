import numpy as np
import pytest

from perception_bench import scenarios, sequences

BY_NAME = {s.name: s for s in scenarios.SCENARIOS}


def test_scenario_names_are_unique():
    assert len(BY_NAME) == len(scenarios.SCENARIOS)


def test_same_seed_gives_the_same_frames_and_another_seed_does_not(scene):
    first = scenarios.generate(BY_NAME["nominal"], scene, 2, seed=3)
    again = scenarios.generate(BY_NAME["nominal"], scene, 2, seed=3)
    other = scenarios.generate(BY_NAME["nominal"], scene, 2, seed=4)
    assert all(np.array_equal(a.rgb, b.rgb) for a, b in zip(first, again))
    assert not np.array_equal(first[0].rgb, other[0].rgb)


def test_nominal_frames_show_the_whole_cube_and_box(scene):
    for frame in scenarios.generate(BY_NAME["nominal"], scene, 3, seed=0):
        assert frame.truth.cube.visible_fraction >= 0.98
        assert frame.truth.box.visible_fraction >= 0.98


def test_camera_gap_changes_the_picture_and_keeps_the_truth(scene):
    plain = scenarios.generate(BY_NAME["nominal"], scene, 2, seed=5)
    gap = scenarios.generate(BY_NAME["gap_color"], scene, 2, seed=5)
    assert [f.truth for f in plain] == [f.truth for f in gap]
    assert not np.array_equal(plain[0].rgb, gap[0].rgb)


def test_decoy_scenarios_keep_the_cube_in_view(scene):
    frames = scenarios.generate(BY_NAME["decoys_red"], scene, 2, seed=1)
    assert len(frames) == 2
    assert all(f.truth.cube.visible_fraction >= 0.98 for f in frames)


def test_decoys_without_a_cube_leave_the_box_in_view(scene):
    frames = scenarios.generate(BY_NAME["absent_decoys"], scene, 2, seed=2)
    assert all(f.truth.cube.visible_px == 0 for f in frames)
    assert all(f.truth.box.visible_px > 0 for f in frames)


def test_empty_view_has_neither_cube_nor_box(scene):
    frames = scenarios.generate(BY_NAME["absent_empty"], scene, 2, seed=2)
    assert frames
    assert all(f.truth.cube.visible_px == 0 and f.truth.box.visible_px == 0 for f in frames)


def test_partial_occlusion_frames_fall_in_the_requested_bin(scene):
    frames = scenarios.generate(BY_NAME["occlusion_partial"], scene, 1, seed=0)
    assert frames
    assert all(0.3 <= f.truth.cube.visible_fraction < 0.9 for f in frames)


def test_motion_sequence_leaves_the_view_and_comes_back(scene):
    frames = sequences.generate(sequences.Motion("test", 0.06), scene)
    cube = [f.truth.cube for f in frames]
    assert len(frames) == len(sequences.path(0.06))
    assert cube[0].visible_px > 0 and cube[-1].visible_px > 0
    assert any(t.visible_px == 0 for t in cube)
    positions = [t.centroid for t in cube if t.centroid]
    assert positions[0] == pytest.approx(positions[-1])
