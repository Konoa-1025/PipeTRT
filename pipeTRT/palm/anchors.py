import numpy as np


def generate_anchors():
    anchors = []

    # stride 8
    for y in range(24):
        for x in range(24):
            center_x = (x + 0.5) / 24
            center_y = (y + 0.5) / 24

            anchors.append([center_x, center_y])
            anchors.append([center_x, center_y])

    # stride 16
    for y in range(12):
        for x in range(12):
            center_x = (x + 0.5) / 12
            center_y = (y + 0.5) / 12

            for count in range(6):
                anchors.append([center_x, center_y])

    return np.array(
        anchors,
        dtype=np.float32
    )