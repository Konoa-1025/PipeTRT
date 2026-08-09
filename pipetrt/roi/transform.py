import cv2
import numpy as np
import math


def extract_roi(
    frame,
    roi,
    output_size=224
):
    height, width = frame.shape[:2]

    center_x = roi["center_x"] * width
    center_y = roi["center_y"] * height

    roi_width = roi["width"] * width
    roi_height = roi["height"] * height

    rotation = roi["rotation"]

    cos_angle = math.cos(rotation)
    sin_angle = math.sin(rotation)

    half_width = roi_width / 2.0
    half_height = roi_height / 2.0

    local_points = np.array(
        [
            [-half_width, -half_height],
            [ half_width, -half_height],
            [ half_width,  half_height],
            [-half_width,  half_height]
        ],
        dtype=np.float32
    )

    source_points = []

    for x, y in local_points:
        rotated_x = (
            x * cos_angle
            - y * sin_angle
            + center_x
        )

        rotated_y = (
            x * sin_angle
            + y * cos_angle
            + center_y
        )

        source_points.append(
            [
                rotated_x,
                rotated_y
            ]
        )

    source_points = np.array(
        source_points,
        dtype=np.float32
    )

    destination_points = np.array(
        [
            [0, 0],
            [output_size - 1, 0],
            [output_size - 1, output_size - 1],
            [0, output_size - 1]
        ],
        dtype=np.float32
    )

    transform = cv2.getPerspectiveTransform(
        source_points,
        destination_points
    )

    roi_image = cv2.warpPerspective(
        frame,
        transform,
        (output_size, output_size),
        borderMode=cv2.BORDER_CONSTANT
    )

    return roi_image, transform


def restore_landmarks_to_image(
    landmarks,
    transform
):
    inverse_transform = np.linalg.inv(
        transform
    )

    roi_points = landmarks[:, :2].astype(
        np.float32
    )

    roi_points = roi_points.reshape(
        -1,
        1,
        2
    )

    restored_points = cv2.perspectiveTransform(
        roi_points,
        inverse_transform
    )

    restored_points = restored_points.reshape(
        -1,
        2
    )

    image_landmarks = []

    for index, point in enumerate(restored_points):
        x = point[0]
        y = point[1]

        z = landmarks[index][2]

        image_landmarks.append(
            [
                x,
                y,
                z
            ]
        )

    return np.array(
        image_landmarks,
        dtype=np.float32
    )