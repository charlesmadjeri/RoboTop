from collections.abc import Callable

from perception_bench import metrics
from perception_bench.policy import HELD, LOST, HoldPolicy
from perception_bench.truth import Frame


def score_sequence(frames: list[Frame], detections: list[dict], timeout: int = 5) -> dict:
    result = metrics.score(frames, detections, "cube")
    policy = HoldPolicy(timeout)
    readings = [policy.update(d["cube"]) for d in detections]
    cubes = [f.truth.cube for f in frames]
    exit_at = _first(len(frames), lambda i: metrics.is_absent(cubes[i]), 0)
    lost_at = _first(len(frames), lambda i: readings[i].state == LOST, exit_at)
    back_at = _first(len(frames), lambda i: metrics.is_present(cubes[i]), exit_at)
    hit_at = _first(len(frames), lambda i: metrics.is_hit(cubes[i], detections[i]["cube"]), back_at)
    result["time_to_lost"] = _gap(exit_at, lost_at)
    result["reacquire"] = _gap(back_at, hit_at)
    result["held_frames"] = sum(r.state == HELD for r in readings)
    return result


def _first(count: int, condition: Callable[[int], bool], start: int | None) -> int | None:
    if start is None:
        return None
    return next((i for i in range(start, count) if condition(i)), None)


def _gap(start: int | None, end: int | None) -> int | None:
    return None if start is None or end is None else end - start
