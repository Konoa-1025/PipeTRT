import cv2
import math
import time
import numpy as np
import pipetrt


landmarker = pipetrt.HandLandmarker()

cap = cv2.VideoCapture(
    0,
    cv2.CAP_MSMF
)

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    1280
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    720
)

cap.set(
    cv2.CAP_PROP_FPS,
    60
)


cv2.namedWindow(
    "PipeTRT TensorRT Benchmark",
    cv2.WINDOW_NORMAL
)

cv2.namedWindow(
    "Landmark Input ROI",
    cv2.WINDOW_NORMAL
)


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

    performance_lines = [
        f"FPS        : {smoothed_fps:.1f}",
        f"Camera     : {camera_ms:.2f} ms",
        f"PipeTRT    : {detect_ms:.2f} ms",
        f"Draw       : {draw_ms:.2f} ms",
        f"Total      : {total_ms:.2f} ms",
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