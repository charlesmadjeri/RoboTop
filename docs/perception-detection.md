# Detecting the cube and the drop box

`so101_perception/detection.py` takes an `rgb8` image and returns the pixel `(u, v)` of the red cube
and of the drop box, or `None` when nothing is found. It has no ROS import, so it runs the same on the
simulated frames and on the real camera. Projection to 3D is the job of the next step.

## Method

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

Measured on the simulated scene: the cube sits at hue 167 to 174, the box at hue 8 to 11, and the gray
floor and the arm stay under saturation 100. The scene comment notes the printed cube reads hue 168 and
saturation 185 on the real camera, inside the cube range.

## Why HSV

- **Cost**: no model, no training data, both targets cost 0.8 ms per frame at 320x240 and 4 ms at 640x480 on a CPU, far above the 5 to 10 Hz needed.
- **Same code on both backends**: the cube and the box have a known color, the sim and the printed PLA match.
- **Light robustness**: hue stays stable when the light changes; the saturation and value floors reject shadows and glare. A test runs the dimmed scene.
- **Resolution independent**: the area threshold is a ratio of the image, so 320x240 (sim) and 640x480 (USB camera) both work.

Rejected: YOLO (needs a dataset and a GPU to find one object of known color) and ArUco (neither object carries a marker).

### Decision from the benchmark

The benchmark of the detection methods keeps HSV, with the white balance and contrast normalization of `hsv_normalized`
(100% recall under every lighting scenario). YOLO as trained found the cube in 0 of 20 real frames. The node and the
calibration commands call `so101_perception/detector.py`:

1. Shrink the frame to 640 px wide (`normalization.shrink_to_width`), so a 1280x720 frame costs about 10 to 40 ms, under the 100 ms of a 10 Hz loop.
2. Balance the white (median of each channel) and equalize the V channel (CLAHE, clip 2.0, 4x4 tiles).
3. Run `find_cube` and `find_drop_box` from `detection.py`, then scale the pixels back to the full frame.

`detection.py` is unchanged, the benchmark imports it. The box range of the detector asks for saturation 150 or more
instead of 100: on the real frames the dark power adapter reads saturation 101 to 143 after normalization and was taken
for the box, while the simulated box reads 174 to 182. No real drop box has been photographed yet, so its saturation on
the real camera is not checked.

## Limits

- Anything else red or brown in view (a hand, a wooden table) is picked up. Keep the workspace clear, or retune the ranges.
- The returned pixel is the centroid of the visible blob, not the cube center on the table; the 3D step has to account for that.
- The cube ranges are checked on one real frame (`test/data/real_cube.png`) and the simulated scenes. The drop box range is checked on the simulation only.

## Tests

`test/test_detection.py` runs on saved simulator renders in `test/data`, plus their ground truth in `expected.json`.
The expected pixels come from projecting `get_object_position()` and `get_drop_box_position()` through
`get_camera_intrinsics()` and `get_camera_extrinsics()`, which is allowed in tests and forbidden in the pipeline.

Scenes: cube and box (two poses), cube only, box only, empty, dimmed light.

`test/test_detector.py` runs the same scenes through `detect`, the real frame at 640x360 and at full size, the scale back, the time budget and a blue cast. `test/test_normalization.py` checks the shrink, the white balance and the equalization.

Regenerate the images with `SO101Sim(seed=...)`: `connect()`, apply the scene edit (move the cube with its free joint,
move the box with `model.body_pos`, scale `model.light_diffuse`), then `render_camera()` and save as PNG.
