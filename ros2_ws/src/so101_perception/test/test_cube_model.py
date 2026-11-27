from pathlib import Path

import numpy as np
import so101_sim

from so101_perception.projection import CUBE_CENTER_HEIGHT

ASSETS = Path(so101_sim.__file__).parent / "assets" / "so101" / "assets"
TRIANGLE = np.dtype([("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])


def mesh_vertices_in_meters(name: str) -> np.ndarray:
    data = (ASSETS / name).read_bytes()
    count = int(np.frombuffer(data[80:84], "<u4")[0])
    triangles = np.frombuffer(data[84 : 84 + 50 * count], TRIANGLE)
    return triangles["vertices"].reshape(-1, 3).astype(float) / 1000.0


def test_the_cube_model_is_50_mm_wide_and_rests_on_the_table_with_its_center_at_the_projection_height():
    vertices = mesh_vertices_in_meters("hollow_cube_rhombus_50mm.stl")
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    assert np.allclose(high - low, 0.05, atol=1e-4)
    assert np.allclose((low + high)[:2], 0.0, atol=1e-4)
    assert low[2] == 0.0
    assert (low[2] + high[2]) / 2 == CUBE_CENTER_HEIGHT


def test_the_collision_octant_is_half_the_width_of_the_cube_model():
    visual = mesh_vertices_in_meters("hollow_cube_rhombus_50mm.stl")
    octant = mesh_vertices_in_meters("pickup_object_collision_octant.stl")
    assert np.allclose(octant.max(axis=0) - octant.min(axis=0), 0.025, atol=1e-4)
    assert np.allclose(2 * octant.max(axis=0), visual.max(axis=0) - visual.min(axis=0), atol=1e-4)
