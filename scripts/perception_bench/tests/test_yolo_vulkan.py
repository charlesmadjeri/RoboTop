import sys

import numpy as np
import pytest

from perception_bench.methods import yolo_vulkan

FRAME = np.zeros((240, 320, 3), np.uint8)


@pytest.fixture
def weights(tmp_path, monkeypatch):
    (tmp_path / "model.ncnn.param").write_text("")
    (tmp_path / "model.ncnn.bin").write_text("")
    monkeypatch.setattr(yolo_vulkan, "WEIGHTS", tmp_path)


def test_missing_weights_name_the_training_script(monkeypatch, tmp_path):
    monkeypatch.setattr(yolo_vulkan, "WEIGHTS", tmp_path)
    with pytest.raises(FileNotFoundError, match="train_yolo"):
        yolo_vulkan.make(FRAME)


def test_missing_ncnn_names_the_wheel(weights, monkeypatch):
    monkeypatch.setitem(sys.modules, "ncnn", None)
    with pytest.raises(RuntimeError, match="pip install ncnn"):
        yolo_vulkan.make(FRAME)


def test_no_vulkan_device_is_reported(weights, monkeypatch):
    fake = type("Ncnn", (), {"get_gpu_count": staticmethod(lambda: 0)})
    monkeypatch.setitem(sys.modules, "ncnn", fake)
    with pytest.raises(RuntimeError, match="Vulkan"):
        yolo_vulkan.make(FRAME)
