import cv2
import numpy as np


def preprocess(frame):
    input_size = 192

    height, width = frame.shape[:2]

    scale = min(
        input_size / width,
        input_size / height
    )

    new_width = int(width * scale)
    new_height = int(height * scale)

    resized = cv2.resize(
        frame,
        (new_width, new_height)
    )

    pad_width = input_size - new_width
    pad_height = input_size - new_height

    left = pad_width // 2
    right = pad_width - left
    top = pad_height // 2
    bottom = pad_height - top

    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )

    rgb = cv2.cvtColor(
        padded,
        cv2.COLOR_BGR2RGB
    )

    tensor = rgb.astype(np.float32) / 255.0

    tensor = np.expand_dims(
        tensor,
        axis=0
    )

    return tensor