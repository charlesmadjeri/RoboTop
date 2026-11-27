import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from perception_bench import metrics, report, report_motion, scenarios, sequence_metrics, sequences
from perception_bench.methods import METHODS
from perception_bench.render import capture
from perception_bench.scene import Scene

LATENCY_FRAMES = 30
RESOLUTIONS = {"320x240": 1, "640x480": 2}


def build_frames(scene: Scene, count: int, seed: int, only: list[str] | None) -> dict:
    frames = {}
    for index, scenario in enumerate(scenarios.SCENARIOS):
        if only is None or scenario.name in only:
            frames[scenario.name] = scenarios.generate(scenario, scene, count, seed + index)
    return frames


def build_sequences(scene: Scene, only: list[str] | None) -> dict:
    return {m.name: sequences.generate(m, scene) for m in sequences.MOTIONS if only is None or m.name in only}


def empty_frame(scene: Scene) -> np.ndarray:
    scene.reset()
    return capture(scene).rgb


def latency_ms(detector, frames: list) -> dict:
    cv2.setNumThreads(1)
    sample = [f.rgb for f in frames[:LATENCY_FRAMES]]
    result = {}
    for label, factor in RESOLUTIONS.items():
        images = [cv2.resize(i, None, fx=factor, fy=factor) if factor != 1 else i for i in sample]
        start = time.perf_counter()
        for image in images:
            detector(image)
        result[label] = (time.perf_counter() - start) / len(images) * 1000
    return result


def score_frames(detector, frames: list) -> tuple[list[dict], dict]:
    found = [detector(f.rgb) for f in frames]
    return found, {target: metrics.score(frames, found, target) for target in ("cube", "box")}


def evaluate(names: list[str], static: dict, moving: dict, empty: np.ndarray) -> dict:
    out = {"frames": {k: len(v) for k, v in {**static, **moving}.items()}, "methods": {}, "motion": {}}
    for name in names:
        detector = METHODS[name](empty)
        cells = {scenario: score_frames(detector, frames)[1] for scenario, frames in static.items()}
        motion = {}
        for scenario, frames in moving.items():
            found, cells[scenario] = score_frames(detector, frames)
            motion[scenario] = sequence_metrics.score_sequence(frames, found)
        reference = next(iter(static.values()))
        out["methods"][name] = {"scenarios": cells, "latency_ms": latency_ms(detector, reference)}
        out["motion"][name] = motion
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--methods", nargs="*", default=list(METHODS))
    parser.add_argument("--scenarios", nargs="*")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    scene = Scene()
    static = build_frames(scene, args.frames, args.seed, args.scenarios)
    moving = build_sequences(scene, args.scenarios)
    results = evaluate(args.methods, static, moving, empty_frame(scene))
    scene.close()
    if args.out:
        args.out.write_text(json.dumps(results, indent=2))
    print(report.render(results))
    print(report_motion.render(results["motion"]))


if __name__ == "__main__":
    main()
