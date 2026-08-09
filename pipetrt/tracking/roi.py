# pipetrt/tracking/roi.py

import math
import numpy as np


def create_tracking_roi(
    landmarks,
    image_width,
    image_height,
    scale=1.5
):
    """
    21点のHand Landmarkから
    次フレーム用の回転ROIを生成する。

    landmarks:
        元画像座標のLandmark
        shape = (21, 3)

    return:
        {
            "center_x": 正規化座標,
            "center_y": 正規化座標,
            "width": 正規化サイズ,
            "height": 正規化サイズ,
            "rotation": rad
        }
    """

    if landmarks is None:
        return None

    if len(landmarks) != 21:
        return None

    landmarks = np.asarray(
        landmarks,
        dtype=np.float32
    )

    # --------------------------------
    # 21点のXY座標
    # --------------------------------

    points = landmarks[:, :2]

    x_values = points[:, 0]
    y_values = points[:, 1]

    # --------------------------------
    # Landmark全体の範囲
    # --------------------------------

    x_min = np.min(x_values)
    y_min = np.min(y_values)

    x_max = np.max(x_values)
    y_max = np.max(y_values)

    center_x = (
        x_min + x_max
    ) / 2.0

    center_y = (
        y_min + y_max
    ) / 2.0

    hand_width = (
        x_max - x_min
    )

    hand_height = (
        y_max - y_min
    )

    # --------------------------------
    # 正方形ROI
    # --------------------------------

    roi_size = max(
        hand_width,
        hand_height
    )

    roi_size *= scale

    # --------------------------------
    # 手の回転角
    #
    # Landmark 0 = Wrist
    # Landmark 9 = Middle MCP
    # --------------------------------

    wrist = points[0]
    middle_mcp = points[9]

    dx = (
        middle_mcp[0]
        - wrist[0]
    )

    dy = (
        middle_mcp[1]
        - wrist[1]
    )

    angle = math.atan2(
        dy,
        dx
    )

    # extract_roi()で扱う向きへ変換
    rotation = (
        angle
        - math.pi / 2.0
    )

    # --------------------------------
    # 正規化
    # --------------------------------

    center_x /= image_width
    center_y /= image_height

    roi_width = (
        roi_size / image_width
    )

    roi_height = (
        roi_size / image_height
    )

    return {
        "center_x": float(center_x),
        "center_y": float(center_y),
        "width": float(roi_width),
        "height": float(roi_height),
        "rotation": float(rotation)
    }