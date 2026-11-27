# Detecting the cube and the drop box

`so101_perception/detection.py` takes an `rgb8` image and returns the pixel `(u, v)` of the red cube
and of the drop box, or `None` when nothing is found. It has no ROS import, so it runs the same on the
simulated frames and on the real camera. Projection to 3D is the job of the next step.

## Method in the package

HSV thresholding, then the largest blob of the mask.

1. Convert the RGB frame to HSV (OpenCV scale, hue 0 to 180).
2. Keep the pixels inside the color ranges of the target.
3. Open then close the mask (3x3) to drop speckles and fill the holes of the cube.
4. Take the external contour with the largest area and return its centroid.
5. Return `None` when that area is under 0.08% of the image (a few dozen pixels at 320x240).

| Target | Hue | Saturation | Value | Why |
|---|---|---|---|---|
| Cube | 160 to 180 and 0 to 4 | 120 to 255 | 60 to 255 | Red wraps around hue 0, so two ranges are merged |
| Drop box | 5 to 20 | 100 to 255 | 40 to 255 | Brown, far from the cube hue |

## Benchmark

`scripts/perception_bench` renders the scene with MuJoCo, takes the exact visible mask and centroid of each
target from the segmentation, and scores seven detectors on 16 scenarios (40 frames each, seeded). A hit is a
detection within half the target width of the visible centroid. How to run it is in its README.

| Method | What it is | Setup cost |
|---|---|---|
| `hsv_fixed` | The package code above | None |
| `hsv_normalized` | Gray-world white balance and CLAHE on V, then the same ranges | None |
| `lab_geometry` | Lab chroma distance, growing radius, compact and square blob only | None |
| `background_diff` | Change against a stored empty scene, red pixels only | A clean reference frame, fixed camera |
| `aruco` | `cv2.aruco` on a printed marker on the cube top and in the box | A physical marker on both objects |
| `yolo` | YOLOv8n fine-tuned on 3000 rendered frames, ONNX through `cv2.dnn` on the CPU | Training (6 epochs, CPU), weights file |
| `yolo_vulkan` | The same network through ncnn on a Vulkan GPU | The same, plus the ncnn wheel and a Vulkan driver |

### Cube recall at 640x480

| Scenario | hsv_fixed | hsv_norm | lab_geom | bg_diff | aruco | yolo |
|---|---|---|---|---|---|---|
| nominal | 100% | 100% | 100% | 100% | 100% | 100% |
| light dim | 65% | 100% | 38% | 100% | 98% | 100% |
| light bright | 100% | 100% | 40% | 45% | 57% | 100% |
| light warm | 100% | 100% | 95% | 48% | 88% | 100% |
| light cool | 100% | 100% | 98% | 45% | 85% | 100% |
| camera color gap | 88% | 100% | 88% | 48% | 85% | 100% |
| camera sensor gap | 100% | 100% | 100% | 100% | 62% | 100% |
| red decoys | 80% | 80% | 80% | 65% | 82% | 100% |
| arm partly in front | 100% | 100% | 78% | 90% | 5% | 100% |
| arm mostly in front | 25% | 25% | 0% | 12% | 0% | 42% |
| cube sliding, fast | 94% | 94% | 89% | 94% | 0% | 94% |
| worst scenario | 25% | 25% | 0% | 12% | 0% | 42% |
| false positives, decoys only | 92% | 92% | 50% | 82% | 0% | 0% |

At 320x240 the picture is the same, except that `aruco` falls to 5% on the nominal scene: the marker is too
small to decode at that size, so it needs the 640x480 camera. `yolo` and `yolo_vulkan` return the same
detections (centres within 0.12 px on 20 frames, same recall in every scenario).

### Drop box recall at 640x480

| Scenario | hsv_fixed | hsv_norm | lab_geom | bg_diff | aruco | yolo |
|---|---|---|---|---|---|---|
| light dim | 40% | 88% | 28% | 40% | 100% | 100% |
| light warm | 70% | 100% | 0% | 70% | 100% | 100% |
| camera color gap | 78% | 95% | 92% | 78% | 100% | 100% |
| worst scenario | 40% | 88% | 0% | 40% | 95% | 100% |

### Position error and latency

| Method | Cube p95 error, worst scenario (px) | ms per frame at 320x240 | ms per frame at 640x480 |
|---|---|---|---|
| hsv_fixed | 263 | 1.2 | 5.5 |
| hsv_normalized | 168 | 5.0 | 40 |
| lab_geometry | 121 | 9.5 | 53 |
| background_diff | 292 | 10 | 72 |
| aruco | 13 | 3.0 | 9.7 |
| yolo, CPU | 9 | 183 | 186 |
| yolo_vulkan, AMD Radeon 780M | 9 | 34 | 35 |

The large errors are the wrong blob (a red decoy, or a warm light on the whole floor), not a noisy centre.
`aruco` is steady at 12 px because the marker centre is the top face of the cube, not the visible centroid.
YOLO costs the same at both sizes because the frame is resized to 320x320 first. The GPU is about 5 times
faster than the CPU here, and a 5 to 10 Hz loop only needs 100 to 200 ms.

### What each method does when the cube leaves the view

The benchmark puts a hold policy on every detector: the last position is published with its age for 5 frames,
then the cube is declared lost. With that policy the HSV methods declare it lost after 3 to 4 frames of
missing detections and find it again 1 to 2 frames after it returns. `aruco` needs 14 frames to find it again
after a slow exit, and never after a fast one (motion blur breaks the marker). `yolo` finds it again after 1
to 5 frames. This policy is bench code. The production version, and what the brain does with a lost target,
belong to issue #18.

## Reading the results

- **Red decoys are the main risk of every color method.** A red object other than the cube is chosen as often
  as 92% of the frames when the cube is absent. Only `yolo` (shape and context) and `aruco` (an id) reject it.
- **The arm in front of the cube is not solved by anyone.** At 42% of recall in the worst case, even `yolo` fails
  when most of the cube is hidden. The perception has to publish "lost" and the brain has to wait or look again.
- **`hsv_normalized` is the best method that needs nothing new.** It keeps 100% recall on every lighting
  scenario, where `hsv_fixed` loses the dim light (65%) and the box (40%). It costs 5 ms at 320x240 and 40 ms at
  640x480, still inside the 100 ms budget.
- **`lab_geometry` and `background_diff` do not beat HSV.** The first trades recall for fewer false positives, the second
  breaks as soon as the light changes color (48% under warm light) and needs a clean reference frame.
- **`aruco` is a ceiling, not an option as is.** It is exact on the box and under changing light, but the real
  cube has no marker, it needs 640x480, and it fails under occlusion and blur.
- **`yolo` is the most robust method on simulated frames**, and it runs at 35 ms on an integrated GPU. The next
  section shows it does not carry over to the real camera as trained.

## Real camera check

20 frames from the USB camera (640x480, `/dev/video0`, stable exposure) of one scene: the printed red cube on a
white desk with black cables and a dark power adapter, no drop box. The cube centre was read by eye at (251, 257).
`python -m perception_bench.real <frames> --cube 251,257` reproduces the table.

| Method | Cube found | Median error (px) | Box reported (none in view) |
|---|---|---|---|
| hsv_fixed | 20/20 | 1.9 | 20/20 |
| hsv_normalized | 20/20 | 3.2 | 20/20 |
| lab_geometry | 20/20 | 3.2 | 20/20 |
| yolo and yolo_vulkan | 0/20 | n/a | 20/20 |

- The HSV ranges tuned on the simulation find the real cube. The printed cube is hollow (cut-out windows) and so is
  the sim cube (`hollow_cube_rhombus_50mm.stl`).
- `yolo`, trained on rendered frames only, finds the object at the right position (within 2 px) and calls it a
  box. It does not transfer. It needs real labelled frames, or a mix with the rendered ones, before it can be used.
- Every method reports a box that is not there. The brown range (hue 5 to 20, saturation 100 and up) matches the
  dark brown power adapter and its label. Keep brown objects out of the view or tighten the box range.
- One scene, one lighting, one cube position, no arm. It checks the transfer, it does not replace the lighting and
  occlusion scenarios above.

## Decision

Keep `hsv_fixed` in the package for this PR: it is measured in simulation and on the real camera, and it costs
nothing. The benchmark says its next step is `hsv_normalized`, which fixes the light and the dim box at the same
setup cost. `yolo_vulkan` is the candidate if red decoys or a changing workspace are expected, after it is
trained on real frames. The box range needs a real fixture with a brown object in view.

### Applied in the package

The benchmark of the detection methods keeps HSV, with the white balance and contrast normalization of `hsv_normalized`
(100% recall under every lighting scenario). YOLO as trained found the cube in 0 of 20 real frames. The node and the
calibration commands call `so101_perception/detector.py`:

1. Shrink the frame to 640 px wide (`normalization.shrink_to_width`), so a 1280x720 frame costs about 10 ms, under the 100 ms of a 10 Hz loop.
2. Run `find_cube` and `find_drop_box` from `detection.py` on the plain frame.
3. For a target not found, balance the white (median of each channel), equalize the V channel (CLAHE, clip 2.0, 4x4 tiles) and look again.
4. Scale the pixels back to the full frame.

The plain frame comes first because the equalization moves the centroid. On a well lit real frame of the CM678 (cube and
wooden bowl, `test/data/real_scene.png`) the plain cube blob is 17316 px2 at (404, 477), the equalized one 6196 px2 at
(414, 445), 33 px higher: the contrast step splits the cube. The sim scene dimmed to 20% is only found after the
normalization. `detection.py` is unchanged, the benchmark imports it.

The box ranges are those of `detection.py`. A dark power adapter reads the hue and saturation of the wooden bowl, so it
is taken for the drop box (`test_detector.py`, expected failure on `real_cube.png`): keep it out of the view. The bowl is
a round object seen from the side, its blob centroid is not the base of the box on the table.

## What is not measured

- **The real camera beyond one scene.** Lighting changes, decoys and the arm in front of the cube were only run on
  rendered frames. The camera gap scenarios are synthetic stand-ins for them.
- Specular glare, decoys that are not spheres, and a moving drop box.
- Nvidia and native Windows for `yolo_vulkan`. The code path is the same (the ncnn wheel and the Vulkan driver
  of the vendor), but it was run on Linux with the AMD GPU only.
- Grasp failure and an unreachable target. They are decisions of the brain and of the IK step, not of the detector.

## Limits of the package code

- Anything else red or brown in view (a hand, a wooden table) is picked up. Keep the workspace clear, or retune the ranges.
- The returned pixel is the centroid of the visible blob, not the cube center on the table; the 3D step has to account for that.
- The ranges are checked on two real frames (`test/data/real_cube.png`, `real_scene.png`) and the simulated scenes. The red cap on the robot base passes the cube mask: it is small, and loses to the cube unless the cube is hidden.

## Tests

`test/test_detection.py` runs on saved simulator renders in `test/data`, plus their ground truth in `expected.json`.
The expected pixels come from projecting `get_object_position()` and `get_drop_box_position()` through
`get_camera_intrinsics()` and `get_camera_extrinsics()`, which is allowed in tests and forbidden in the pipeline.

Scenes: cube and box (two poses), cube only, box only, empty, dimmed light.

`test/test_detector.py` runs the same scenes through `detect`, the real frames at 640x360 and at full size, the plain frame first and the fallback on a dark scene, the scale back, the time budget and a blue cast. `test/test_normalization.py` checks the shrink, the white balance and the equalization.

Regenerate the images with `SO101Sim(seed=...)`: `connect()`, apply the scene edit (move the cube with its free joint,
move the box with `model.body_pos`, scale `model.light_diffuse`), then `render_camera()` and save as PNG.
