import os
from pathlib import Path

import numpy as np

from perception_bench.methods import yolo

WEIGHTS = Path(os.environ.get("ROBOTOP_YOLO_NCNN", Path.home() / ".cache" / "robotop" / "yolo_cube_box_ncnn"))
DEVICE = int(os.environ.get("ROBOTOP_VULKAN_DEVICE", "0"))


def make(empty: np.ndarray):
    param, binary = WEIGHTS / "model.ncnn.param", WEIGHTS / "model.ncnn.bin"
    if not (param.exists() and binary.exists()):
        raise FileNotFoundError(f"{WEIGHTS} is missing, train it with perception_bench/train_yolo.py")
    try:
        import ncnn
    except ImportError as error:
        raise RuntimeError("yolo_vulkan needs the ncnn wheel: pip install ncnn") from error
    if ncnn.get_gpu_count() <= DEVICE:
        raise RuntimeError("no Vulkan device found, install the GPU driver with Vulkan support")
    net = ncnn.Net()
    net.opt.use_vulkan_compute = True
    net.set_vulkan_device(DEVICE)
    net.load_param(str(param))
    net.load_model(str(binary))

    def forward(canvas: np.ndarray) -> np.ndarray:
        image = ncnn.Mat.from_pixels(canvas, ncnn.Mat.PixelType.PIXEL_RGB, yolo.SIZE, yolo.SIZE)
        image.substract_mean_normalize([], [1 / 255] * 3)
        extractor = net.create_extractor()
        extractor.input("in0", image)
        return np.array(extractor.extract("out0")[1])[None]

    forward(np.zeros((yolo.SIZE, yolo.SIZE, 3), np.uint8))

    def detect(rgb: np.ndarray) -> dict:
        canvas, scale, pad = yolo.letterbox(rgb)
        return yolo.decode(forward(canvas), scale, pad)

    return detect
