# Perception benchmark

Compares cube and drop box detectors on rendered frames with exact ground truth. Every detector is a function `detect(rgb) -> {"cube": (u, v) | None, "box": (u, v) | None}` registered in `methods/__init__.py`.

## Run

From `scripts/`, with the pixi environment python:

```
python -m perception_bench.run --frames 40 --methods hsv_fixed --out results.json
```

Headless rendering uses EGL. On NixOS add `LD_LIBRARY_PATH=/run/opengl-driver/lib`, `__EGL_VENDOR_LIBRARY_DIRS=/run/opengl-driver/share/glvnd/egl_vendor.d` and `LIBGL_DRIVERS_PATH=/run/opengl-driver/lib/dri`.

Tests (the rendering ones skip when no offscreen OpenGL is available):

```
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest perception_bench/tests
```

## Methods

- `hsv_fixed`: the fixed HSV ranges of `so101_perception.detection`, the baseline.
- `hsv_normalized`: gray-world white balance and CLAHE on V, then the same HSV ranges.
- `lab_geometry`: Lab chroma distance to a reference color at growing radii, kept only if the blob is compact and roughly square.
- `background_diff`: change against the empty-scene frame, kept only where the pixel is red. The box is static and part of that frame, so its position comes from the HSV box detector.

## How it is scored

- Ground truth comes from MuJoCo segmentation rendering: the visible mask and its centroid, and the full size with the arm and decoys hidden.
- A frame counts for recall when at least 10 pixels of the target are visible. A detection is a hit within half the target width of the visible centroid.
- A frame counts for false positives when none of the target is visible. Frames in between are ignored.
- Latency is the mean time per frame on one CPU thread, at 320x240 and at 640x480 (upscaled frames).

## Scenarios

`nominal`, five lighting changes, two camera gaps (color, sensor), red and brown decoys, two occlusion levels, and two absent cases. The motion sequences slide the cube out of the view and back at two speeds, with motion blur, and score a hold policy on top of each detector: the last position is published with its age until a timeout, then the cube is declared lost.

## Not covered

Specular glare, decoy shapes other than spheres, a moving box, and the real camera. The camera gap scenarios are synthetic stand-ins until real frames exist.
