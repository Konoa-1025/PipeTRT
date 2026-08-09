
#! 起動高速のためにdshowを使用する

import cv2
import math
import time
import numpy as np
import pipetrt
from pipetrt.tracking.roi import create_tracking_roi


# =====================================
# Startup Benchmark
# =====================================

startup_start = time.perf_counter()


# -----------------------------
# PipeTRT Initialize
# -----------------------------

print("PipeTRT initialization start")

pipetrt_start = time.perf_counter()

landmarker = pipetrt.HandLandmarker()

pipetrt_end = time.perf_counter()

print(
    f"PipeTRT initialization: "
    f"{pipetrt_end - pipetrt_start:.2f} sec"
)


# -----------------------------
# Camera Initialize
# -----------------------------

print("Camera initialization start")

# カメラ初期化全体の計測開始
camera_init_start = time.perf_counter()


# VideoCapture
start = time.perf_counter()

cap = cv2.VideoCapture(
    0,
    cv2.CAP_DSHOW
)

print(
    f"VideoCapture : "
    f"{time.perf_counter() - start:.2f} sec"
)


# Width
start = time.perf_counter()

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    1280
)

print(
    f"Set Width    : "
    f"{time.perf_counter() - start:.2f} sec"
)


# Height
start = time.perf_counter()

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    720
)

print(
    f"Set Height   : "
    f"{time.perf_counter() - start:.2f} sec"
)


# FPS
start = time.perf_counter()

cap.set(
    cv2.CAP_PROP_FPS,
    60
)

print(
    f"Set FPS      : "
    f"{time.perf_counter() - start:.2f} sec"
)


# First Read
start = time.perf_counter()

ret, test_frame = cap.read()

print(
    f"First Read   : "
    f"{time.perf_counter() - start:.2f} sec"
)


# カメラ初期化全体の計測終了
camera_init_end = time.perf_counter()

print(
    f"Camera Total : "
    f"{camera_init_end - camera_init_start:.2f} sec"
)


# -----------------------------
# Window Initialize
# -----------------------------

print("Window initialization start")

window_start = time.perf_counter()

cv2.namedWindow(
    "PipeTRT TensorRT Benchmark",
    cv2.WINDOW_NORMAL
)

cv2.namedWindow(
    "Landmark Input ROI",
    cv2.WINDOW_NORMAL
)

window_end = time.perf_counter()

print(
    f"Window initialization: "
    f"{window_end - window_start:.2f} sec"
)


startup_end = time.perf_counter()

print()
print("==============================")
print("Startup Benchmark")
print("==============================")

print(
    f"PipeTRT : "
    f"{pipetrt_end - pipetrt_start:.2f} sec"
)

print(
    f"Camera  : "
    f"{camera_init_end - camera_init_start:.2f} sec"
)

print(
    f"Window  : "
    f"{window_end - window_start:.2f} sec"
)

print(
    f"TOTAL   : "
    f"{startup_end - startup_start:.2f} sec"
)

print("==============================")
print()


smoothed_fps = 0.0


while True:
    frame_start = time.perf_counter()

    # -----------------------------
    # Camera
    # -----------------------------

    camera_start = time.perf_counter()

    ret, frame = cap.read()

    camera_end = time.perf_counter()

    if not ret:
        print("Camera read failed")
        break

    # -----------------------------
    # PipeTRT
    # -----------------------------

    detect_start = time.perf_counter()

    result = landmarker.detect(
        frame
    )

    detect_end = time.perf_counter()

    height, width = frame.shape[:2]

    # -----------------------------
    # Draw
    # -----------------------------

    draw_start = time.perf_counter()

    # Palm bbox
    if result.palm_result:
        palm = result.palm_result[0]

        x_min, y_min, x_max, y_max = (
            palm["bbox"]
        )

        palm_x1 = int(
            x_min * width
        )

        palm_y1 = int(
            y_min * height
        )

        palm_x2 = int(
            x_max * width
        )

        palm_y2 = int(
            y_max * height
        )

        cv2.rectangle(
            frame,
            (palm_x1, palm_y1),
            (palm_x2, palm_y2),
            (255, 0, 0),
            2
        )

    # Rotated ROI
    if result.roi is not None:
        roi = result.roi

        center_x = (
            roi["center_x"]
            * width
        )

        center_y = (
            roi["center_y"]
            * height
        )

        roi_width = (
            roi["width"]
            * width
        )

        roi_height = (
            roi["height"]
            * height
        )

        rotation_degree = (
            math.degrees(
                roi["rotation"]
            )
        )

        rotated_rect = (
            (center_x, center_y),
            (roi_width, roi_height),
            rotation_degree
        )

        box = cv2.boxPoints(
            rotated_rect
        )

        box = box.astype(
            np.int32
        )

        cv2.polylines(
            frame,
            [box],
            True,
            (0, 255, 0),
            2
        )

    # 21 Landmarks
    for landmark in (
        result.hand_landmarks
    ):
        x = int(
            landmark[0]
        )

        y = int(
            landmark[1]
        )

        cv2.circle(
            frame,
            (x, y),
            4,
            (0, 255, 255),
            -1
        )

    draw_end = time.perf_counter()
    tracking_roi = None

    if len(result.hand_landmarks) == 21:
        tracking_roi = create_tracking_roi(
            result.hand_landmarks,
            width,
            height
        )

    if tracking_roi is not None:
        tracking_center_x = (
            tracking_roi["center_x"]
            * width
        )

        tracking_center_y = (
            tracking_roi["center_y"]
            * height
        )

        tracking_width = (
            tracking_roi["width"]
            * width
        )

        tracking_height = (
            tracking_roi["height"]
            * height
        )

        tracking_rotation = (
            math.degrees(
                tracking_roi["rotation"]
            )
        )

        tracking_rect = (
            (
                tracking_center_x,
                tracking_center_y
            ),
            (
                tracking_width,
                tracking_height
            ),
            tracking_rotation
        )

        tracking_box = cv2.boxPoints(
            tracking_rect
        )

        tracking_box = tracking_box.astype(
            np.int32
        )

        cv2.polylines(
            frame,
            [tracking_box],
            True,
            (255, 0, 255),
            2
        )

    # -----------------------------
    # Performance
    # -----------------------------

    frame_end = time.perf_counter()

    camera_ms = (
        camera_end
        - camera_start
    ) * 1000.0

    detect_ms = (
        detect_end
        - detect_start
    ) * 1000.0

    draw_ms = (
        draw_end
        - draw_start
    ) * 1000.0

    total_ms = (
        frame_end
        - frame_start
    ) * 1000.0

    current_fps = (
        1000.0 / total_ms
        if total_ms > 0
        else 0.0
    )

    if smoothed_fps == 0.0:
        smoothed_fps = (
            current_fps
        )

    else:
        smoothed_fps = (
            smoothed_fps * 0.9
            + current_fps * 0.1
        )

    # -----------------------------
    # Performance Draw
    # -----------------------------

    timings = result.timings

    performance_lines = [
    f"FPS          : {smoothed_fps:.1f}",
    f"Camera       : {camera_ms:.2f} ms",
    f"Palm TRT     : {timings.get('palm_trt_ms', 0.0):.2f} ms",
    f"Palm Decode  : {timings.get('palm_decode_ms', 0.0):.2f} ms",
    f"ROI          : {timings.get('roi_ms', 0.0):.2f} ms",
    f"ROI Transform: {timings.get('roi_transform_ms', 0.0):.2f} ms",
    f"Landmark TRT : {timings.get('landmark_trt_ms', 0.0):.2f} ms",
    f"Restore      : {timings.get('restore_ms', 0.0):.2f} ms",
    f"PipeTRT Total: {timings.get('total_ms', 0.0):.2f} ms",
    f"Draw         : {draw_ms:.2f} ms",
    f"Frame Total  : {total_ms:.2f} ms",
]

    text_y = 30

    for line in performance_lines:
        cv2.putText(
            frame,
            line,
            (20, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        text_y += 30


    # -----------------------------
    # Windows
    # -----------------------------

    if result.roi_image is not None:
        cv2.imshow(
            "Landmark Input ROI",
            result.roi_image
        )

    cv2.imshow(
        "PipeTRT TensorRT Benchmark",
        frame
    )

    key = (
        cv2.waitKey(1)
        & 0xFF
    )

    if key in (
        27,
        ord("q")
    ):
        break


landmarker.close()
cap.release()
cv2.destroyAllWindows()