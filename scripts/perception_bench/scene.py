from dataclasses import dataclass

import mujoco
import numpy as np

from perception_bench import REPO

SCENE_PATH = REPO / "sim" / "so101_sim" / "assets" / "so101" / "scene_tek5.xml"
CAMERA = "external_cam"
CUBE_BODY = "pickup_object"
BOX_BODY = "drop_box"
CUBE_JOINT = "pickup_object_free"
MAX_DECOYS = 4
WARM_UP_RENDERS = 2
PARK = np.array([5.0, 5.0, -1.0])
DECOY_GROUP = 4
ARM_GROUP = 5
VISIBLE_GROUPS = (1, 1, 1, 0, 1, 1)
STATIC_GROUPS = (1, 1, 1, 0, 0, 0)
LIGHT_FIELDS = ("light_diffuse", "light_ambient", "light_specular", "light_dir")


@dataclass(frozen=True)
class Lighting:
    arrays: dict
    headlight_diffuse: np.ndarray
    headlight_ambient: np.ndarray


class Scene:
    def __init__(self, width: int = 320, height: int = 240):
        spec = mujoco.MjSpec.from_file(str(SCENE_PATH))
        for index in range(MAX_DECOYS):
            body = spec.worldbody.add_body(name=f"decoy_{index}", mocap=True, pos=PARK.tolist())
            body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.02, 0.02, 0.02],
                rgba=[1, 0, 0, 1],
                group=DECOY_GROUP,
                contype=0,
                conaffinity=0,
            )
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.width, self.height = width, height
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.cube_id = self.model.body(CUBE_BODY).id
        self.box_id = self.model.body(BOX_BODY).id
        self.decoy_ids = [self.model.body(f"decoy_{i}").id for i in range(MAX_DECOYS)]
        self.cube_geoms = self._visible_geoms(self.cube_id)
        self.box_geoms = self._visible_geoms(self.box_id)
        self._move_arm_geoms_to_arm_group()
        self.cube_adr = self.model.joint(CUBE_JOINT).qposadr[0]
        self.arm_joints = self._arm_joints()
        self.home_qpos = self.data.qpos.copy()
        self._nominal = self._snapshot_lighting()
        self.cube_z = self._resting_cube_z()
        self._box_pos = self.model.body_pos[self.box_id].copy()
        self.reset()
        self._warm_up()

    def close(self) -> None:
        self.renderer.close()

    def reset(self) -> None:
        self.restore_lighting()
        self.data.qpos[:] = self.home_qpos
        self.data.qvel[:] = 0
        self.park_decoys()
        self.park_cube()
        self.model.body_pos[self.box_id] = self._box_pos
        self.forward()

    def hide_box(self) -> None:
        self.model.body_pos[self.box_id] = PARK

    def forward(self) -> None:
        mujoco.mj_forward(self.model, self.data)

    def set_cube(self, x: float, y: float, yaw: float, z: float | None = None) -> None:
        self.data.qpos[self.cube_adr : self.cube_adr + 3] = [x, y, self.cube_z if z is None else z]
        self.data.qpos[self.cube_adr + 3 : self.cube_adr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]

    def park_cube(self) -> None:
        self.set_cube(PARK[0], PARK[1], 0.0, PARK[2])

    def set_arm(self, values: np.ndarray) -> None:
        for adr, value in zip(self.arm_joints[0], values):
            self.data.qpos[adr] = value

    def arm_ranges(self) -> np.ndarray:
        return self.arm_joints[1]

    def place_decoy(self, index: int, xy, radius: float, rgba) -> None:
        geom = self.model.body_geomadr[self.decoy_ids[index]]
        self.model.geom_size[geom] = [radius, radius, radius]
        self.model.geom_rgba[geom] = rgba
        self.data.mocap_pos[self.model.body_mocapid[self.decoy_ids[index]]] = [xy[0], xy[1], radius]

    def park_decoys(self) -> None:
        for mocap in self.model.body_mocapid[self.decoy_ids]:
            self.data.mocap_pos[mocap] = PARK

    def scale_lighting(self, diffuse: float, ambient: float, headlight: float) -> None:
        self.model.light_diffuse[:] = self._nominal.arrays["light_diffuse"] * diffuse
        self.model.light_specular[:] = self._nominal.arrays["light_specular"] * diffuse
        self.model.light_ambient[:] = self._nominal.arrays["light_ambient"] * ambient
        self.model.vis.headlight.diffuse[:] = self._nominal.headlight_diffuse * headlight
        self.model.vis.headlight.ambient[:] = self._nominal.headlight_ambient * headlight

    def tint_lighting(self, gains) -> None:
        self.model.light_diffuse[:] *= np.asarray(gains)
        self.model.vis.headlight.diffuse[:] *= np.asarray(gains)

    def restore_lighting(self) -> None:
        for name in LIGHT_FIELDS:
            getattr(self.model, name)[:] = self._nominal.arrays[name]
        self.model.vis.headlight.diffuse[:] = self._nominal.headlight_diffuse
        self.model.vis.headlight.ambient[:] = self._nominal.headlight_ambient

    def set_light_direction(self, index: int, direction) -> None:
        unit = np.asarray(direction, dtype=float)
        self.model.light_dir[index] = unit / np.linalg.norm(unit)

    def _warm_up(self) -> None:
        for _ in range(WARM_UP_RENDERS):
            self.renderer.update_scene(self.data, camera=CAMERA)
            self.renderer.render()

    def _visible_geoms(self, body_id: int) -> np.ndarray:
        geoms = np.flatnonzero(self.model.geom_bodyid == body_id)
        return geoms[self.model.geom_group[geoms] <= 2]

    def _move_arm_geoms_to_arm_group(self) -> None:
        owners = set(self.decoy_ids) | {0, self.cube_id, self.box_id}
        for geom in range(self.model.ngeom):
            if self.model.geom_bodyid[geom] not in owners and self.model.geom_group[geom] <= 2:
                self.model.geom_group[geom] = ARM_GROUP

    def _arm_joints(self):
        adrs, ranges = [], []
        for joint in range(self.model.njnt):
            if self.model.jnt_type[joint] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                adrs.append(self.model.jnt_qposadr[joint])
                ranges.append(self.model.jnt_range[joint].copy())
        return np.array(adrs), np.array(ranges)

    def _snapshot_lighting(self) -> Lighting:
        arrays = {name: getattr(self.model, name).copy() for name in LIGHT_FIELDS}
        return Lighting(arrays, self.model.vis.headlight.diffuse.copy(), self.model.vis.headlight.ambient.copy())

    def _resting_cube_z(self) -> float:
        self.data.qpos[self.cube_adr : self.cube_adr + 3] = [0.2, 0.0, 0.05]
        self.data.qpos[self.cube_adr + 3 : self.cube_adr + 7] = [1, 0, 0, 0]
        for _ in range(300):
            mujoco.mj_step(self.model, self.data)
        return float(self.data.qpos[self.cube_adr + 2])
