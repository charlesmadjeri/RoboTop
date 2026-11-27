import numpy as np
from cube_scene import pose_error
from so101_sim import SO101Sim

from so101_perception.camera_pose import solve_camera_pose
from so101_perception.detection import find_cube
from so101_perception.projection import CUBE_CENTER_HEIGHT

OPTICAL_FROM_MUJOCO = np.diag([1.0, -1.0, -1.0, 1.0])


def test_cube_renders_at_six_random_places_give_back_the_simulated_camera_pose():
    pixels, places = [], []
    for seed in range(6):
        sim = SO101Sim(seed=seed)
        sim.connect()
        K = sim.get_camera_intrinsics()
        truth = sim.get_camera_extrinsics() @ OPTICAL_FROM_MUJOCO
        pixels.append(find_cube(sim.render_camera()))
        places.append(sim.get_object_position()[:2])
        sim.disconnect()
    found, error = solve_camera_pose(np.array(pixels), np.array(places), CUBE_CENTER_HEIGHT, K)
    translation, angle = pose_error(found, truth)
    assert translation < 0.015 and angle < 1.0
    assert error < 1.0
