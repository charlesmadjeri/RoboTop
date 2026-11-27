import numpy as np
import pytest
from builtin_interfaces.msg import Time
from so101_sim import SO101Sim

from so101_driver.sim_camera_pose import camera_transform, optical_pose, quaternion_from_rotation


def rotation_from_quaternion(x, y, z, w):
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def random_rotations(count):
    rng = np.random.default_rng(3)
    rotations = [np.linalg.qr(rng.normal(size=(3, 3)))[0] for _ in range(count)]
    return [rotation * np.linalg.det(rotation) for rotation in rotations]


@pytest.mark.parametrize("rotation", [np.eye(3), np.diag([1.0, -1.0, -1.0]), np.diag([-1.0, 1.0, -1.0]), np.diag([-1.0, -1.0, 1.0])])
def test_half_turns_and_identity_give_a_unit_quaternion_that_rebuilds_the_rotation(rotation):
    quaternion = quaternion_from_rotation(rotation)
    assert np.linalg.norm(quaternion) == pytest.approx(1.0)
    assert rotation_from_quaternion(*quaternion) == pytest.approx(rotation)


@pytest.mark.parametrize("rotation", random_rotations(20))
def test_any_rotation_survives_the_round_trip_through_a_quaternion(rotation):
    assert rotation_from_quaternion(*quaternion_from_rotation(rotation)) == pytest.approx(rotation, abs=1e-9)


def test_optical_axes_follow_the_mujoco_camera_that_looks_along_its_minus_z():
    mujoco = np.eye(4)
    mujoco[:3, :3] = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    optical = optical_pose(mujoco)
    assert optical[:3, :3] @ [0, 0, 1] == pytest.approx(-mujoco[:3, :3] @ [0, 0, 1])
    assert optical[:3, :3] @ [0, 1, 0] == pytest.approx(-mujoco[:3, :3] @ [0, 1, 0])
    assert optical[:3, :3] @ [1, 0, 0] == pytest.approx(mujoco[:3, :3] @ [1, 0, 0])


def test_transform_names_the_two_frames_and_carries_the_stamp_and_position():
    mujoco = np.eye(4)
    mujoco[:3, 3] = [0.45, 0.45, 0.55]
    transform = camera_transform(mujoco, "world", "external_cam_optical_frame", Time(sec=5, nanosec=6))
    assert (transform.header.frame_id, transform.child_frame_id) == ("world", "external_cam_optical_frame")
    assert transform.header.stamp == Time(sec=5, nanosec=6)
    translation = transform.transform.translation
    assert (translation.x, translation.y, translation.z) == pytest.approx((0.45, 0.45, 0.55))


def test_published_transform_projects_the_simulated_drop_box_where_the_camera_sees_it():
    sim = SO101Sim()
    sim.connect()
    transform = camera_transform(sim.get_camera_extrinsics(), "world", "external_cam_optical_frame", Time())
    rotation = transform.transform.rotation
    translation = transform.transform.translation
    robot_from_camera = np.eye(4)
    robot_from_camera[:3, :3] = rotation_from_quaternion(rotation.x, rotation.y, rotation.z, rotation.w)
    robot_from_camera[:3, 3] = [translation.x, translation.y, translation.z]
    box = np.append(sim.get_drop_box_position(), 1.0)
    in_camera = np.linalg.inv(robot_from_camera) @ box
    pixel = sim.get_camera_intrinsics() @ in_camera[:3]
    u, v = pixel[:2] / pixel[2]
    sim.disconnect()
    assert in_camera[2] > 0
    assert (u, v) == pytest.approx((211.1, 170.3), abs=1.0)
