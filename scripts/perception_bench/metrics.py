from math import hypot

import numpy as np

from perception_bench.truth import Frame, Point, Target

MIN_PRESENT_PX = 10
HIT_FRACTION = 0.5


def is_present(target: Target) -> bool:
    return target.visible_px >= MIN_PRESENT_PX


def is_absent(target: Target) -> bool:
    return target.visible_px == 0


def error_px(target: Target, found: Point | None) -> float | None:
    if found is None or target.centroid is None:
        return None
    return hypot(found[0] - target.centroid[0], found[1] - target.centroid[1])


def is_hit(target: Target, found: Point | None) -> bool:
    error = error_px(target, found)
    return error is not None and error <= HIT_FRACTION * target.width


def score(frames: list[Frame], detections: list[dict], name: str) -> dict:
    present, absent, errors, hits, false_alarms = 0, 0, [], 0, 0
    for frame, detected in zip(frames, detections):
        target = getattr(frame.truth, name)
        found = detected[name]
        if is_present(target):
            present += 1
            hits += is_hit(target, found)
            error = error_px(target, found)
            if error is not None:
                errors.append(error)
        elif is_absent(target):
            absent += 1
            false_alarms += found is not None
    return {
        "recall": hits / present if present else None,
        "median_px": float(np.median(errors)) if errors else None,
        "p95_px": float(np.percentile(errors, 95)) if errors else None,
        "false_positive": false_alarms / absent if absent else None,
        "n_eval": present,
        "n_absent": absent,
    }
