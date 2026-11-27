from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")

TRACKING = "tracking"
HELD = "held"
LOST = "lost"
DEFAULT_HOLD_FRAMES = 5


@dataclass(frozen=True)
class Reading(Generic[T]):
    state: str
    value: T | None
    age: int


class HoldPolicy(Generic[T]):
    def __init__(self, hold_frames: int = DEFAULT_HOLD_FRAMES):
        if hold_frames < 0:
            raise ValueError("hold_frames must be zero or more")
        self._hold_frames = hold_frames
        self._last: T | None = None
        self._age = 0

    def update(self, found: T | None) -> Reading[T]:
        if found is not None:
            self._last, self._age = found, 0
            return Reading(TRACKING, found, 0)
        self._age += 1
        if self._last is None or self._age > self._hold_frames:
            self._last = None
            return Reading(LOST, None, self._age)
        return Reading(HELD, self._last, self._age)
