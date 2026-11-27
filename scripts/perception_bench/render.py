import mujoco
import numpy as np

from perception_bench.scene import CAMERA, STATIC_GROUPS, VISIBLE_GROUPS, Scene
from perception_bench.truth import Frame, Target, Truth

GEOM = int(mujoco.mjtObj.mjOBJ_GEOM)


def render_rgb(scene: Scene) -> np.ndarray:
    scene.renderer.update_scene(scene.data, camera=CAMERA, scene_option=_options(VISIBLE_GROUPS))
    return scene.renderer.render().copy()


def capture(scene: Scene) -> Frame:
    rgb = render_rgb(scene)
    visible = _segmentation(scene, VISIBLE_GROUPS)
    full = _segmentation(scene, STATIC_GROUPS)
    truth = Truth(
        cube=_target(visible, full, scene.cube_geoms),
        box=_target(visible, full, scene.box_geoms),
    )
    return Frame(rgb, truth)


def _options(groups) -> mujoco.MjvOption:
    option = mujoco.MjvOption()
    option.geomgroup[:] = groups
    return option


def _segmentation(scene: Scene, groups) -> np.ndarray:
    scene.renderer.enable_segmentation_rendering()
    scene.renderer.update_scene(scene.data, camera=CAMERA, scene_option=_options(groups))
    seg = scene.renderer.render().copy()
    scene.renderer.disable_segmentation_rendering()
    return seg


def _mask(seg: np.ndarray, geoms: np.ndarray) -> np.ndarray:
    return (seg[..., 1] == GEOM) & np.isin(seg[..., 0], geoms)


def _target(visible: np.ndarray, full: np.ndarray, geoms: np.ndarray) -> Target:
    visible_mask = _mask(visible, geoms)
    count = int(visible_mask.sum())
    centroid = None
    if count:
        ys, xs = np.nonzero(visible_mask)
        centroid = (float(xs.mean()), float(ys.mean()))
    return Target(count, int(_mask(full, geoms).sum()), centroid)
