from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from perception_bench import augment, decoys, lighting
from perception_bench.sampling import random_arm, random_cube, sample
from perception_bench.scene import Scene
from perception_bench.truth import Frame

Setup = Callable[[Scene, np.random.Generator], None]
Accept = Callable[[Frame], bool]
COLOR_GAP = ("white_balance", "gamma", "desaturate")
SENSOR_GAP = ("gaussian_noise", "blur", "jpeg", "vignette")


@dataclass(frozen=True)
class Scenario:
    name: str
    setup: Setup
    accept: Accept
    perturbations: tuple[str, ...] = ()


def clear(frame: Frame) -> bool:
    return frame.truth.cube.visible_fraction >= 0.98 and frame.truth.box.visible_fraction >= 0.98


def cube_fraction(low: float, high: float) -> Accept:
    return lambda frame: frame.truth.cube.full_px > 0 and low <= frame.truth.cube.visible_fraction < high


def nothing_to_find(frame: Frame) -> bool:
    return frame.truth.cube.visible_px == 0 and frame.truth.box.visible_px == 0


def cube_gone(frame: Frame) -> bool:
    return frame.truth.cube.visible_px == 0


def arm_and_cube(scene: Scene, rng: np.random.Generator) -> None:
    random_cube(scene, rng)
    random_arm(scene, rng)


def with_effect(effect: Setup) -> Setup:
    def setup(scene: Scene, rng: np.random.Generator) -> None:
        arm_and_cube(scene, rng)
        effect(scene, rng)

    return setup


def empty_view(scene: Scene, rng: np.random.Generator) -> None:
    scene.hide_box()
    random_arm(scene, rng)


def decoys_without_cube(scene: Scene, rng: np.random.Generator) -> None:
    random_arm(scene, rng)
    decoys.place_red(scene, rng)


SCENARIOS = (
    Scenario("nominal", arm_and_cube, clear),
    *(Scenario(name, with_effect(effect), clear) for name, effect in lighting.LIGHTING.items()),
    Scenario("gap_color", arm_and_cube, clear, COLOR_GAP),
    Scenario("gap_sensor", arm_and_cube, clear, SENSOR_GAP),
    Scenario("decoys_red", with_effect(decoys.place_red), clear),
    Scenario("decoys_brown", with_effect(decoys.place_brown), clear),
    Scenario("occlusion_partial", arm_and_cube, cube_fraction(0.3, 0.9)),
    Scenario("occlusion_heavy", arm_and_cube, cube_fraction(0.02, 0.3)),
    Scenario("absent_decoys", decoys_without_cube, cube_gone),
    Scenario("absent_empty", empty_view, nothing_to_find),
)


def generate(scenario: Scenario, scene: Scene, count: int, seed: int) -> list[Frame]:
    rng = np.random.default_rng(seed)
    frames = sample(scene, rng, count, scenario.setup, scenario.accept)
    return [Frame(augment.apply(f.rgb, scenario.perturbations, rng), f.truth) for f in frames]
