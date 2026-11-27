import numpy as np
from builtin_interfaces.msg import Time
from geometry_msgs.msg import TransformStamped

OPTICAL_FROM_MUJOCO = np.diag([1.0, -1.0, -1.0, 1.0])


def optical_pose(mujoco_pose: np.ndarray) -> np.ndarray:
    return mujoco_pose @ OPTICAL_FROM_MUJOCO


def quaternion_from_rotation(rotation: np.ndarray) -> tuple[float, float, float, float]:
    (xx, xy, xz), (yx, yy, yz), (zx, zy, zz) = rotation
    symmetric = np.array(
        [
            [xx - yy - zz, yx + xy, zx + xz, zy - yz],
            [yx + xy, yy - xx - zz, zy + yz, xz - zx],
            [zx + xz, zy + yz, zz - xx - yy, yx - xy],
            [zy - yz, xz - zx, yx - xy, xx + yy + zz],
        ]
    )
    quaternion = np.linalg.eigh(symmetric)[1][:, -1]
    x, y, z, w = quaternion if quaternion[3] >= 0 else -quaternion
    return float(x), float(y), float(z), float(w)


def camera_transform(mujoco_pose: np.ndarray, parent_frame: str, camera_frame: str, stamp: Time) -> TransformStamped:
    pose = optical_pose(mujoco_pose)
    transform = TransformStamped()
    transform.header.stamp = stamp
    transform.header.frame_id = parent_frame
    transform.child_frame_id = camera_frame
    translation = transform.transform.translation
    translation.x, translation.y, translation.z = (float(value) for value in pose[:3, 3])
    rotation = transform.transform.rotation
    rotation.x, rotation.y, rotation.z, rotation.w = quaternion_from_rotation(pose[:3, :3])
    return transform
