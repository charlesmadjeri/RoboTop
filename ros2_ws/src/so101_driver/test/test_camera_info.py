import pytest

from so101_driver.camera_info import build_camera_info, pinhole_intrinsics


def test_principal_point_is_the_image_center():
    K = pinhole_intrinsics(640, 480, 70.0)
    assert K[2] == pytest.approx(320.0)
    assert K[5] == pytest.approx(240.0)


def test_focal_length_of_a_90_degree_field_of_view_is_half_the_width():
    K = pinhole_intrinsics(640, 480, 90.0)
    assert K[0] == pytest.approx(320.0)


def test_focal_lengths_are_equal_for_square_pixels():
    K = pinhole_intrinsics(640, 480, 70.0)
    assert K[0] == pytest.approx(K[4])


def test_projection_matrix_repeats_the_intrinsics_with_a_zero_last_column():
    K = pinhole_intrinsics(640, 480, 70.0)
    msg = build_camera_info(K, 640, 480, "external_cam_optical_frame")
    assert list(msg.p[0:3]) == pytest.approx(K[0:3])
    assert msg.p[3] == 0.0
    assert list(msg.p[4:7]) == pytest.approx(K[3:6])
    assert msg.p[7] == 0.0
    assert list(msg.p[8:11]) == pytest.approx(K[6:9])
    assert msg.p[11] == 0.0


def test_frame_id_and_distortion_model_are_set_on_the_message():
    K = pinhole_intrinsics(640, 480, 70.0)
    msg = build_camera_info(K, 640, 480, "external_cam_optical_frame")
    assert msg.header.frame_id == "external_cam_optical_frame"
    assert msg.distortion_model == "plumb_bob"
    assert list(msg.d) == [0.0] * 5
    assert (msg.width, msg.height) == (640, 480)
