from dataclasses import dataclass

from perception_bench.truth import Point

TRACKING = "tracking"
HELD = "held"
LOST = "lost"


@dataclass(frozen=True)
class Reading:
    state: str
    position: Point | None
    age: int


class HoldPolicy:
    def __init__(self, timeout: int = 5):
        self.timeout = timeout
        self._last: Point | None = None
        self._age = 0

    def update(self, found: Point | None) -> Reading:
        if found is not None:
            self._last, self._age = found, 0
            return Reading(TRACKING, found, 0)
        self._age += 1
        if self._last is None or self._age > self.timeout:
            self._last = None
            return Reading(LOST, None, self._age)
        return Reading(HELD, self._last, self._age)
