class JointInterpolator:
    def __init__(self, joint_count: int, max_step: float, tolerance: float = 1e-4):
        self._count = joint_count
        self._max_step = max_step
        self._tolerance = tolerance
        self.target = [0.0] * joint_count
        self.commanded = [0.0] * joint_count
        self.ready = False

    def start(self, q: list[float]):
        self.target = list(q)
        self.commanded = list(q)
        self.ready = True

    def hold(self):
        self.target = list(self.commanded)

    def step(self) -> list[float] | None:
        moved = False
        for i in range(self._count):
            diff = self.target[i] - self.commanded[i]
            if abs(diff) > self._tolerance:
                self.commanded[i] += max(-self._max_step, min(self._max_step, diff))
                moved = True
        return list(self.commanded) if moved else None
