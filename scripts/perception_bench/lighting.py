import numpy as np

from perception_bench.scene import Scene


def dim(scene: Scene, rng: np.random.Generator) -> None:
    level = rng.uniform(0.12, 0.4)
    scene.scale_lighting(level, level, level)


def bright(scene: Scene, rng: np.random.Generator) -> None:
    level = rng.uniform(1.8, 2.5)
    scene.scale_lighting(level, level, level)


def warm(scene: Scene, rng: np.random.Generator) -> None:
    scene.tint_lighting([rng.uniform(1.2, 1.5), 1.0, rng.uniform(0.55, 0.8)])


def cool(scene: Scene, rng: np.random.Generator) -> None:
    scene.tint_lighting([rng.uniform(0.55, 0.8), 1.0, rng.uniform(1.2, 1.5)])


def shadow(scene: Scene, rng: np.random.Generator) -> None:
    scene.scale_lighting(1.2, 0.35, 0.25)
    scene.set_light_direction(0, [rng.uniform(-0.9, 0.9), rng.uniform(-0.9, 0.9), -rng.uniform(0.25, 0.5)])


LIGHTING = {
    "light_dim": dim,
    "light_bright": bright,
    "light_warm": warm,
    "light_cool": cool,
    "light_shadow": shadow,
}
