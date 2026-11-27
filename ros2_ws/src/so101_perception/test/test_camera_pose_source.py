import numpy as np
import pytest
from geometry_msgs.msg import Transform, TransformStamped
from scene_images import ROBOT_FROM_CAMERA
from tf2_ros import Buffer

from so101_perception.camera_pose_source import CAMERA_FRAME, ROBOT_FRAME, TfCameraPose, make_camera_pose_source, matrix_from_transform
from so101_perception.pose_file import save_pose

HALF_TURN_ROOT = np.sqrt(0.5)


def quarter_turn_about_z():
    transform = Transform()
    transform.translation.x, transform.translation.y, transform.translation.z = 0.1, 0.2, 0.3
    transform.rotation.z, transform.rotation.w = HALF_TURN_ROOT, HALF_TURN_ROOT
    return transform


def static_buffer():
    buffer = Buffer()
    stamped = TransformStamped(transform=quarter_turn_about_z(), child_frame_id=CAMERA_FRAME)
    stamped.header.frame_id = ROBOT_FRAME
    buffer.set_transform_static(stamped, "test")
    return buffer


def test_identity_transform_gives_the_identity_matrix():
    transform = Transform()
    transform.rotation.w = 1.0
    assert matrix_from_transform(transform) == pytest.approx(np.eye(4))


def test_quarter_turn_about_z_moves_x_to_y_and_keeps_the_translation():
    pose = matrix_from_transform(quarter_turn_about_z())
    assert pose[:3, :3] @ [1, 0, 0] == pytest.approx([0, 1, 0])
    assert pose[:3, 3] == pytest.approx([0.1, 0.2, 0.3])


def test_published_camera_pose_is_read_from_the_tf_buffer():
    pose = TfCameraPose(static_buffer())()
    assert pose == pytest.approx(matrix_from_transform(quarter_turn_about_z()))


def test_camera_pose_is_none_while_the_tf_is_not_there():
    assert TfCameraPose(Buffer())() is None


def test_pose_file_takes_the_place_of_the_tf(tmp_path):
    save_pose(tmp_path / "pose.json", ROBOT_FROM_CAMERA)
    assert make_camera_pose_source(None, str(tmp_path / "pose.json"))() == pytest.approx(ROBOT_FROM_CAMERA)


def test_pose_file_that_is_missing_stops_the_node_at_start(tmp_path):
    with pytest.raises(FileNotFoundError):
        make_camera_pose_source(None, str(tmp_path / "missing.json"))
