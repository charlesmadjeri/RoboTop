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

## Limits

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
