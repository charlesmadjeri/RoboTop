import argparse
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from perception_bench.methods import METHODS, MARKER_METHODS

HIT_PX = 40.0
SKIPPED = MARKER_METHODS | {"background_diff"}


def score(detect: Callable, frames: list[np.ndarray], cube: tuple[float, float]) -> dict:
    errors, box_detections = [], 0
    for frame in frames:
        found = detect(frame)
        box_detections += found["box"] is not None
        if found["cube"] is not None:
            errors.append(float(np.hypot(found["cube"][0] - cube[0], found["cube"][1] - cube[1])))
    return {
        "frames": len(frames),
        "cube_hits": sum(error < HIT_PX for error in errors),
        "cube_median_error": float(np.median(errors)) if errors else None,
        "box_detections": box_detections,
    }


def load(folder: Path) -> list[np.ndarray]:
    frames = []
    for path in sorted(folder.glob("*.png")):
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"cannot read {path}")
        frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not frames:
        raise ValueError(f"no PNG frames in {folder}")
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the detectors on real frames with one cube and no box")
    parser.add_argument("frames", type=Path)
    parser.add_argument("--cube", required=True, help="cube centre in pixels, as U,V")
    args = parser.parse_args()
    cube = tuple(float(v) for v in args.cube.split(","))
    frames = load(args.frames)
    print("| Method | Cube found | Median error (px) | Box reported (should be 0) |\n|---|---|---|---|")
    for name in (m for m in METHODS if m not in SKIPPED):
        try:
            detect = METHODS[name](frames[0])
        except (FileNotFoundError, RuntimeError) as error:
            print(f"| {name} | skipped: {error} | | |")
            continue
        result = score(detect, frames, cube)
        error = result["cube_median_error"]
        print(f"| {name} | {result['cube_hits']}/{result['frames']} | {'n/a' if error is None else f'{error:.1f}'} "
              f"| {result['box_detections']}/{result['frames']} |")


if __name__ == "__main__":
    main()
