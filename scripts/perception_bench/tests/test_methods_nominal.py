import numpy as np
import pytest

from perception_bench import scenarios
from perception_bench.methods import DEFAULT_METHODS, METHODS
from perception_bench.run import empty_frame

NOMINAL = next(s for s in scenarios.SCENARIOS if s.name == "nominal")


@pytest.mark.parametrize("name", sorted(DEFAULT_METHODS))
def test_every_method_finds_the_nominal_cube_and_box_within_six_pixels(scene, name):
    detect = METHODS[name](empty_frame(scene))
    for frame in scenarios.generate(NOMINAL, scene, 4, seed=0):
        found = detect(frame.rgb)
        for target in ("cube", "box"):
            truth = getattr(frame.truth, target).centroid
            assert found[target] is not None
            assert np.hypot(found[target][0] - truth[0], found[target][1] - truth[1]) < 6
