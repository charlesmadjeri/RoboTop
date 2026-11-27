import numpy as np
import pytest

from perception_bench import lighting
from perception_bench.render import render_rgb


def lit(scene, effect, seed=0):
    scene.reset()
    scene.set_cube(0.22, 0.05, 0.3)
    if effect is not None:
        effect(scene, np.random.default_rng(seed))
    scene.forward()
    return render_rgb(scene).astype(float)


def red_to_blue(image):
    return image[..., 0].mean() / image[..., 2].mean()


def test_dim_light_darkens_the_picture(scene):
    assert lit(scene, lighting.dim).mean() < 0.7 * lit(scene, None).mean()


def test_bright_light_brightens_the_picture(scene):
    assert lit(scene, lighting.bright).mean() > 1.15 * lit(scene, None).mean()


def test_warm_light_shifts_the_picture_towards_red(scene):
    assert red_to_blue(lit(scene, lighting.warm)) > red_to_blue(lit(scene, None))


def test_cool_light_shifts_the_picture_towards_blue(scene):
    assert red_to_blue(lit(scene, lighting.cool)) < red_to_blue(lit(scene, None))


def test_shadow_light_changes_the_picture_without_moving_the_cube(scene):
    assert np.abs(lit(scene, lighting.shadow) - lit(scene, None)).mean() > 1.0


@pytest.mark.parametrize("name", list(lighting.LIGHTING))
def test_a_lighting_effect_never_survives_a_reset(scene, name):
    before = lit(scene, None)
    lit(scene, lighting.LIGHTING[name])
    assert np.array_equal(lit(scene, None), before)
