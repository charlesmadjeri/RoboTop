import numpy as np

from perception_bench.scene import MAX_DECOYS, Scene

RED_SHADES = ((1.0, 0.0, 0.0), (0.55, 0.05, 0.05), (1.0, 0.25, 0.05), (0.9, 0.2, 0.3))
BROWN_SHADES = ((0.45, 0.25, 0.1), (0.8, 0.5, 0.3), (0.6, 0.35, 0.2), (0.9, 0.6, 0.4))
MIN_GAP = 0.07


def place_red(scene: Scene, rng: np.random.Generator) -> None:
    _scatter(scene, rng, RED_SHADES, (0.012, 0.035), _workspace)


def place_brown(scene: Scene, rng: np.random.Generator) -> None:
    box = scene.data.body(scene.box_id).xpos[:2].copy()
    _scatter(scene, rng, BROWN_SHADES, (0.02, 0.05), lambda r: box + r.uniform(-0.12, 0.12, 2))


def _workspace(rng: np.random.Generator) -> np.ndarray:
    return np.array([rng.uniform(0.1, 0.4), rng.uniform(-0.3, 0.3)])


def _scatter(scene: Scene, rng: np.random.Generator, shades, radii, position) -> None:
    cube = scene.data.qpos[scene.cube_adr : scene.cube_adr + 2].copy()
    for index in range(int(rng.integers(1, MAX_DECOYS))):
        for _ in range(20):
            xy = position(rng)
            if np.linalg.norm(xy - cube) >= MIN_GAP:
                break
        shade = shades[int(rng.integers(len(shades)))]
        scene.place_decoy(index, xy, rng.uniform(*radii), (*shade, 1.0))
