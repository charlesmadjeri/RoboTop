import numpy as np
from ikpy.chain import Chain
from scipy.spatial.transform import Rotation

ACTIVE_LINKS_MASK = [False, True, True, True, True, True, False]


class ArmKinematics:
    def __init__(self, urdf_path: str):
        self._chain = Chain.from_urdf_file(
            urdf_path,
            base_elements=["base_link"],
            active_links_mask=ACTIVE_LINKS_MASK,
        )

    @staticmethod
    def _full(q: list[float]) -> list[float]:
        return [0.0] + list(q) + [0.0]

    def forward(self, q: list[float]) -> np.ndarray:
        return self._chain.forward_kinematics(self._full(q))

    def position(self, q: list[float]) -> np.ndarray:
        return self.forward(q)[:3, 3]

    def pose(self, q: list[float]) -> tuple[np.ndarray, np.ndarray]:
        matrix = self.forward(q)
        return matrix[:3, 3], Rotation.from_matrix(matrix[:3, :3]).as_quat()

    def solve(self, target: list[float], seed: list[float]) -> list[float]:
        initial = self._full(seed)
        for index, link in enumerate(self._chain.links):
            low, high = link.bounds
            if low is not None and initial[index] < low:
                initial[index] = float(low)
            if high is not None and initial[index] > high:
                initial[index] = float(high)
        solution = self._chain.inverse_kinematics(target_position=target, initial_position=initial)
        return [float(angle) for angle in solution[1:6]]

    def error(self, q: list[float], target: list[float]) -> float:
        return float(np.linalg.norm(self.position(q) - np.array(target)))
