import numpy as np
from sensor_msgs.msg import Image


def rgb_from_image(message: Image) -> np.ndarray:
    if message.encoding != "rgb8":
        raise ValueError(f"expected an rgb8 image, got {message.encoding}")
    rows = np.frombuffer(bytes(message.data), np.uint8).reshape(message.height, message.step)
    return rows[:, : 3 * message.width].reshape(message.height, message.width, 3)
