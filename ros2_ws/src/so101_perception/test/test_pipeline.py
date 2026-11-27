import numpy as np
import pytest
from scene_images import DROP_BOX_BASE_XY, EXPECTED, K, ROBOT_FROM_CAMERA, load

from so101_perception.pipeline import locate
from so101_perception.projection import CUBE_CENTER_HEIGHT, project_to_plane

TOLERANCE_M = 0.01


@pytest.mark.parametrize("scene", ["both", "both_other_pose", "cube_only", "dim_light"])
def test_cube_is_located_within_one_centimeter_at_its_center_height(scene):
    truth = project_to_plane(*EXPECTED[scene]["cube"], K, ROBOT_FROM_CAMERA, CUBE_CENTER_HEIGHT)
    cube, _ = locate(load(scene), K, ROBOT_FROM_CAMERA)
    assert np.linalg.norm(cube[:2] - truth[:2]) < TOLERANCE_M
    assert cube[2] == pytest.approx(CUBE_CENTER_HEIGHT)


@pytest.mark.parametrize("scene", ["both", "both_other_pose", "box_only", "dim_light"])
def test_drop_box_is_located_within_one_centimeter_and_reported_on_the_table(scene):
    _, box = locate(load(scene), K, ROBOT_FROM_CAMERA)
    assert np.linalg.norm(box[:2] - DROP_BOX_BASE_XY) < TOLERANCE_M
    assert box[2] == 0.0


def test_target_missing_from_the_image_is_none_and_the_other_is_kept():
    cube, box = locate(load("cube_only"), K, ROBOT_FROM_CAMERA)
    assert cube is not None and box is None
    cube, box = locate(load("box_only"), K, ROBOT_FROM_CAMERA)
    assert cube is None and box is not None


def test_empty_scene_locates_nothing():
    assert locate(load("empty"), K, ROBOT_FROM_CAMERA) == (None, None)


def test_cube_height_argument_moves_the_point_along_the_ray():
    low, _ = locate(load("both"), K, ROBOT_FROM_CAMERA, cube_height=0.0)
    high, _ = locate(load("both"), K, ROBOT_FROM_CAMERA, cube_height=CUBE_CENTER_HEIGHT)
    assert np.linalg.norm(low[:2] - high[:2]) > TOLERANCE_M
