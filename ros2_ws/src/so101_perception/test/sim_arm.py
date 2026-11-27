import mujoco
import numpy as np
from so101_sim import SO101Sim

from so101_perception.arm_poses import ARM_JOINTS

OPTICAL_FROM_MUJOCO = np.diag([1.0, -1.0, -1.0, 1.0])
FAR_AWAY = [5.0, 5.0, 5.0, 1.0, 0.0, 0.0, 0.0]


class SimArm:
    def __init__(self, camera_width: int = 320, camera_height: int = 240):
        self.sim = SO101Sim(seed=0, camera_width=camera_width, camera_height=camera_height)
        self.sim.connect()
        self.model, self.data = self.sim.model, self.sim.data
        self.K = self.sim.get_camera_intrinsics()
        self.robot_from_camera = self.sim.get_camera_extrinsics() @ OPTICAL_FROM_MUJOCO
        self._joint_address = [self.model.joint(name).qposadr[0] for name in ARM_JOINTS]
        self._cube_address = self.model.joint("pickup_object_free").qposadr[0]

    def place(self, joints: np.ndarray) -> None:
        self.data.qpos[self._joint_address] = joints
        self.data.qpos[self._cube_address : self._cube_address + 7] = FAR_AWAY
        mujoco.mj_forward(self.model, self.data)

    def gripper_pose(self) -> np.ndarray:
        site = self.data.site("gripperframe")
        pose = np.eye(4)
        pose[:3, :3] = site.xmat.reshape(3, 3)
        pose[:3, 3] = site.xpos
        return pose

    def pixel_of(self, point: np.ndarray) -> np.ndarray:
        in_camera = np.linalg.inv(self.robot_from_camera) @ np.append(point, 1.0)
        projected = self.K @ in_camera[:3]
        return projected[:2] / projected[2]

    def touching(self) -> bool:
        return any(
            "pickup" not in self.model.body(self.model.geom_bodyid[contact.geom1]).name
            and "pickup" not in self.model.body(self.model.geom_bodyid[contact.geom2]).name
            for contact in self.data.contact[: self.data.ncon]
        )

    def hold_cube(self, offset: np.ndarray) -> None:
        center = self.gripper_pose() @ np.append(offset, 1.0)
        self.data.qpos[self._cube_address : self._cube_address + 7] = [*center[:3], 1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(self.model, self.data)

    def close(self) -> None:
        self.sim.disconnect()


class SimRig:
    def __init__(self, arm: SimArm, grasp_offset: np.ndarray, start: np.ndarray):
        self.arm = arm
        self.grasp_offset = grasp_offset
        self.joints = np.array(start, dtype=float)
        self.commands = []
        self._hold()

    def _hold(self) -> None:
        self.arm.place(self.joints)
        self.arm.hold_cube(self.grasp_offset)

    def joint_positions(self) -> np.ndarray:
        return self.joints.copy()

    def command(self, joints: np.ndarray) -> None:
        self.commands.append(np.array(joints, dtype=float))
        self.joints = np.array(joints, dtype=float)
        self._hold()

    def gripper_pose(self) -> np.ndarray:
        return self.arm.gripper_pose()

    def wait(self, seconds: float) -> None:
        pass

    def frame(self) -> np.ndarray:
        return self.arm.sim.render_camera()
