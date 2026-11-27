import cv2
import numpy as np

from perception_bench.methods.blobs import Shape, best_blob


def mask_with(*rects, size=(240, 320)):
    mask = np.zeros(size, np.uint8)
    for x0, y0, x1, y1 in rects:
        mask[y0:y1, x0:x1] = 255
    return mask


def test_the_largest_accepted_blob_wins_and_its_centroid_is_returned():
    found = best_blob(mask_with((10, 10, 30, 30), (100, 100, 160, 160)), Shape())
    assert found == (129.5, 129.5)


def test_a_blob_below_the_minimum_area_is_ignored():
    assert best_blob(mask_with((10, 10, 14, 14)), Shape(min_area=0.0008)) is None


def test_a_blob_above_the_maximum_area_is_ignored_and_a_smaller_one_is_kept():
    big, small = (0, 0, 200, 200), (250, 10, 270, 30)
    assert best_blob(mask_with(big, small), Shape(max_area=0.1)) == (259.5, 19.5)


def test_a_thin_bar_is_rejected_by_aspect_ratio():
    assert best_blob(mask_with((10, 100, 200, 108)), Shape(max_aspect=2.0)) is None


def test_a_concave_shape_is_rejected_by_solidity():
    mask = mask_with((50, 50, 150, 70), (50, 50, 70, 150))
    assert best_blob(mask, Shape(min_solidity=0.8)) is None
    assert best_blob(mask, Shape(min_solidity=0.4)) is not None


def test_an_empty_mask_gives_nothing():
    assert best_blob(np.zeros((240, 320), np.uint8), Shape()) is None


def test_specks_are_removed_before_measuring():
    mask = mask_with((100, 100, 140, 140))
    mask[50, 50] = 255
    assert cv2.countNonZero(mask) > 0
    assert best_blob(mask, Shape(min_area=0.0001)) == (119.5, 119.5)
