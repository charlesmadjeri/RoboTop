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
- `aruco`: `cv2.aruco` on a marker stuck on the cube top (id 7) and in the box (id 3). It runs on a scene copy that carries the markers (`markers.py`), alone, because the markers change what the other methods see.
- `yolo`: YOLOv8n fine-tuned on rendered frames, run from an ONNX file through `cv2.dnn`. Not part of the default run, since it needs weights.
- `yolo_vulkan`: the same network run by ncnn on a Vulkan GPU (AMD, Nvidia or Intel, Linux or Windows). Needs the ncnn weights and the `ncnn` wheel, see below.

## YOLO weights

The training stack (torch, ultralytics) is not a project dependency. Install it in a venv outside the repo, then:

```
python -m perception_bench.train_data --out /tmp/yolo_data --count 3000
python -m perception_bench.train_yolo --data /tmp/yolo_data/data.yaml --out ~/.cache/robotop/yolo_cube_box.onnx --runs /tmp/yolo_runs
python -m perception_bench.run --methods yolo --out yolo.json
```

`train_data` runs in the pixi environment (it renders), `train_yolo` runs in the venv. The data uses narrower lighting ranges than the test scenarios (level 0.45 to 1.6 against 0.12 to 0.4 and 1.8 to 2.5), so the dim, bright and warm scenarios are outside what it saw. `ROBOTOP_YOLO_ONNX` overrides the weights path.

## GPU inference

`yolo_vulkan` runs the exported ncnn model on any Vulkan 1.1 device, so one code path serves AMD and Nvidia on Linux and on Windows. Install the wheel in the same venv or any interpreter on `PYTHONPATH` (`pip install ncnn`); it is not a project dependency. `train_yolo` writes `yolo_cube_box_ncnn/` next to the ONNX file. `ROBOTOP_YOLO_NCNN` overrides that folder and `ROBOTOP_VULKAN_DEVICE` picks the GPU index.

```
python -m perception_bench.run --methods yolo_vulkan --out yolo_vulkan.json
```

On NixOS point the loader at the driver, e.g. `VK_ICD_FILENAMES=/run/opengl-driver/share/vulkan/icd.d/radeon_icd.x86_64.json` for AMD. Checked on Linux with the AMD Radeon 780M (RADV). Not run on Nvidia or on native Windows: those need the vendor Vulkan driver, nothing else.

## Several runs in one table

```
python -m perception_bench.run --methods aruco --size 640x480 --out aruco.json
python -m perception_bench.merge cls.json aruco.json yolo.json
```

The runs must share the seed and the size. A scenario whose sampled frame count differs between runs (the markers change which occlusion frames pass the filter) is shown as a range.

## Real frames

Save PNG frames of a scene with one cube and no box (the USB camera at 640x480), read the cube centre in pixels, then:

```
python -m perception_bench.real frames_dir --cube 251,257
```

It lists, per method, how many frames find the cube within 40 px, the median error, and how often a box is reported although none is in view. `aruco` and `background_diff` are skipped: they need a marker and an empty reference frame.

## How it is scored

- Ground truth comes from MuJoCo segmentation rendering: the visible mask and its centroid, and the full size with the arm and decoys hidden.
- A frame counts for recall when at least 10 pixels of the target are visible. A detection is a hit within half the target width of the visible centroid.
- A frame counts for false positives when none of the target is visible. Frames in between are ignored.
- Latency is the mean time per frame at 320x240 and at 640x480 (upscaled frames): one CPU thread, except `yolo_vulkan`, which runs on the GPU.

## Scenarios

`nominal`, five lighting changes, two camera gaps (color, sensor), red and brown decoys, two occlusion levels, and two absent cases. The motion sequences slide the cube out of the view and back at two speeds, with motion blur, and score a hold policy on top of each detector: the last position is published with its age until a timeout, then the cube is declared lost.

## Not covered

Specular glare, decoy shapes other than spheres, a moving box, and real lighting, decoy and occlusion cases (the real camera was checked on one still scene). The camera gap scenarios are synthetic stand-ins.
