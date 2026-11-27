import argparse
from pathlib import Path

import cv2
import numpy as np

from perception_bench import augment, decoys
from perception_bench.render import capture
from perception_bench.sampling import random_arm, random_cube
from perception_bench.scene import Scene
from perception_bench.truth import Truth

CLASSES = ("cube", "box")
MIN_VISIBLE_PX = 30
NO_CUBE_SHARE = 0.12
NO_BOX_SHARE = 0.08
DECOY_SHARE = 0.4
LIGHT_SHARE = 0.6
TINT_SHARE = 0.4
EFFECT_SHARE = 0.3
LEVEL = (0.45, 1.6)
TINT = (0.85, 1.2)
VALIDATION_EVERY = 10


def randomize(scene: Scene, rng: np.random.Generator) -> None:
    random_arm(scene, rng)
    if rng.random() > NO_CUBE_SHARE:
        random_cube(scene, rng)
    if rng.random() < NO_BOX_SHARE:
        scene.hide_box()
    if rng.random() < DECOY_SHARE:
        (decoys.place_red if rng.random() < 0.5 else decoys.place_brown)(scene, rng)
    if rng.random() < LIGHT_SHARE:
        level = rng.uniform(*LEVEL)
        scene.scale_lighting(level, level, level)
    if rng.random() < TINT_SHARE:
        scene.tint_lighting([rng.uniform(*TINT), 1.0, rng.uniform(*TINT)])


def effects(rng: np.random.Generator) -> list[str]:
    return [name for name in augment.PERTURBATIONS if rng.random() < EFFECT_SHARE]


def label_lines(truth: Truth, width: int, height: int) -> list[str]:
    lines = []
    for index, name in enumerate(CLASSES):
        target = getattr(truth, name)
        if target.visible_px < MIN_VISIBLE_PX:
            continue
        x0, y0, x1, y1 = target.bounds
        centre = ((x0 + x1) / 2 / width, (y0 + y1) / 2 / height)
        lines.append(f"{index} {centre[0]:.6f} {centre[1]:.6f} {(x1 - x0) / width:.6f} {(y1 - y0) / height:.6f}")
    return lines


def write_dataset(scene: Scene, out: Path, count: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    for index in range(count):
        scene.reset()
        randomize(scene, rng)
        scene.forward()
        frame = capture(scene)
        rgb = augment.apply(frame.rgb, effects(rng), rng)
        split = "val" if index % VALIDATION_EVERY == 0 else "train"
        cv2.imwrite(str(out / "images" / split / f"{index:05d}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        lines = label_lines(frame.truth, rgb.shape[1], rgb.shape[0])
        (out / "labels" / split / f"{index:05d}.txt").write_text("\n".join(lines))
    names = "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASSES))
    (out / "data.yaml").write_text(f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n{names}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()
    scene = Scene()
    write_dataset(scene, args.out, args.count, args.seed)
    scene.close()


if __name__ == "__main__":
    main()
