# pipetrt/tracking/roi.py

import math
import numpy as np


def normalize_radians(angle):
    return (
        angle
        - 2.0
        * math.pi
        * math.floor(
            (angle + math.pi)
            / (2.0 * math.pi)
        )
    )


def create_tracking_roi(
    landmarks,
    image_width,
    image_height
):
    if landmarks is None:
        return None

    if len(landmarks) != 21:
        return None

    if image_width <= 0 or image_height <= 0:
        return None

    landmarks = np.asarray(
        landmarks,
        dtype=np.float32
    )

    points = landmarks[:, :2]

    # MediaPipeがTracking ROI生成に使用する
    # ランドマークのサブセット
    partial_indices = [
        0,
        1,
        2,
        3,
        5,
        6,
        9,
        10,
        13,
        14,
        17,
        18
    ]

    partial_points = points[
        partial_indices
    ]

    # =====================================
    # Rotation
    # =====================================

    # partial_points上では
    #
    # 0 = Wrist
    # 4 = Index MCP  (元Landmark 5)
    # 6 = Middle MCP (元Landmark 9)
    # 8 = Ring MCP   (元Landmark 13)

    wrist = partial_points[0]
    index_mcp = partial_points[4]
    middle_mcp = partial_points[6]
    ring_mcp = partial_points[8]

    # 人差し指と薬指の中間
    finger_center = (
        index_mcp
        + ring_mcp
    ) / 2.0

    # さらに中指を加味
    finger_center = (
        finger_center
        + middle_mcp
    ) / 2.0

    dx = (
        finger_center[0]
        - wrist[0]
    )

    dy = (
        finger_center[1]
        - wrist[1]
    )

    target_angle = math.pi / 2.0

    rotation = normalize_radians(
        target_angle
        - math.atan2(
            -dy,
            dx
        )
    )

    # =====================================
    # Axis-aligned center
    # =====================================

    min_x = np.min(
        partial_points[:, 0]
    )

    max_x = np.max(
        partial_points[:, 0]
    )

    min_y = np.min(
        partial_points[:, 1]
    )

    max_y = np.max(
        partial_points[:, 1]
    )

    axis_center_x = (
        min_x
        + max_x
    ) / 2.0

    axis_center_y = (
        min_y
        + max_y
    ) / 2.0

    # =====================================
    # 回転後のLandmark範囲を計算
    # =====================================

    reverse_angle = normalize_radians(
        -rotation
    )

    cos_reverse = math.cos(
        reverse_angle
    )

    sin_reverse = math.sin(
        reverse_angle
    )

    projected_points = []

    for point in partial_points:
        original_x = (
            point[0]
            - axis_center_x
        )

        original_y = (
            point[1]
            - axis_center_y
        )

        projected_x = (
            original_x
            * cos_reverse
            - original_y
            * sin_reverse
        )

        projected_y = (
            original_x
            * sin_reverse
            + original_y
            * cos_reverse
        )

        projected_points.append(
            [
                projected_x,
                projected_y
            ]
        )

    projected_points = np.asarray(
        projected_points,
        dtype=np.float32
    )

    min_projected_x = np.min(
        projected_points[:, 0]
    )

    max_projected_x = np.max(
        projected_points[:, 0]
    )

    min_projected_y = np.min(
        projected_points[:, 1]
    )

    max_projected_y = np.max(
        projected_points[:, 1]
    )

    projected_center_x = (
        min_projected_x
        + max_projected_x
    ) / 2.0

    projected_center_y = (
        min_projected_y
        + max_projected_y
    ) / 2.0

    # =====================================
    # Centerを元画像座標へ戻す
    # =====================================

    cos_rotation = math.cos(
        rotation
    )

    sin_rotation = math.sin(
        rotation
    )

    center_x = (
        projected_center_x
        * cos_rotation
        - projected_center_y
        * sin_rotation
        + axis_center_x
    )

    center_y = (
        projected_center_x
        * sin_rotation
        + projected_center_y
        * cos_rotation
        + axis_center_y
    )

    rect_width = (
        max_projected_x
        - min_projected_x
    )

    rect_height = (
        max_projected_y
        - min_projected_y
    )

    # =====================================
    # MediaPipe RectTransformation
    #
    # scale_x = 2.0
    # scale_y = 2.0
    # shift_y = -0.1
    # square_long = true
    # =====================================

    shift_x = 0.0
    shift_y = -0.1

    x_shift = (
        rect_width
        * shift_x
        * cos_rotation
        - rect_height
        * shift_y
        * sin_rotation
    )

    y_shift = (
        rect_width
        * shift_x
        * sin_rotation
        + rect_height
        * shift_y
        * cos_rotation
    )

    center_x += x_shift
    center_y += y_shift

    # square_long
    long_side = max(
        rect_width,
        rect_height
    )

    roi_size = (
        long_side
        * 2.0
    )

    if roi_size <= 0:
        return None

    return {
        "center_x":
            float(
                center_x
                / image_width
            ),

        "center_y":
            float(
                center_y
                / image_height
            ),

        "width":
            float(
                roi_size
                / image_width
            ),

        "height":
            float(
                roi_size
                / image_height
            ),

        "rotation":
            float(rotation)
    }