# example/dev/realtime_trt_benchmark_v4l2.py

# ! v4l2 ubunts

import cv2
import math
import time
import numpy as np

import pipetrt

from pipetrt.tracking.roi import create_tracking_roi


# =====================================
# Settings
# =====================================

CAMERA_ID = 0

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 60

# 最初の数十フレームは
# TensorRT / Camera / Windows側の安定待ちとして除外
WARMUP_FRAMES = 60

# Tracking中Palm TRTがこれ以上動いていたら異常とする
PALM_SKIP_EPSILON_MS = 0.001


# =====================================
# Benchmark Storage
# =====================================

detection_success_times = []
detection_search_times = []

tracking_times = []
tracking_lost_times = []

frame_times = []

palm_detection_times = []
landmark_detection_times = []
landmark_tracking_times = []

tracking_palm_violation_count = 0

detection_success_count = 0
detection_search_count = 0

tracking_count = 0
tracking_lost_count = 0

total_measured_frames = 0


# =====================================
# Statistics
# =====================================

def print_stats(
    name,
    values
):
    if not values:
        print(
            f"{name:<24}: no data"
        )
        return

    values_array = np.asarray(
        values,
        dtype=np.float32
    )

    print(
        f"{name:<24}: "
        f"AVG {np.mean(values_array):7.3f} ms | "
        f"MED {np.median(values_array):7.3f} ms | "
        f"P95 {np.percentile(values_array, 95):7.3f} ms | "
        f"MAX {np.max(values_array):7.3f} ms"
    )


def print_benchmark_result():
    print()
    print("==============================================")
    print("PipeTRT Tracking Benchmark Result")
    print("==============================================")
    print()

    print(
        f"Measured frames          : "
        f"{total_measured_frames}"
    )

    print()

    print(
        f"Detection success frames : "
        f"{detection_success_count}"
    )

    print(
        f"Detection search frames  : "
        f"{detection_search_count}"
    )

    print(
        f"Tracking frames          : "
        f"{tracking_count}"
    )

    print(
        f"Tracking lost frames     : "
        f"{tracking_lost_count}"
    )

    print()

    print("----------------------------------------------")
    print("PipeTRT Processing Time")
    print("----------------------------------------------")

    print_stats(
        "Detection Success",
        detection_success_times
    )

    print_stats(
        "Detection Search",
        detection_search_times
    )

    print_stats(
        "Tracking",
        tracking_times
    )

    print_stats(
        "Tracking Lost",
        tracking_lost_times
    )

    print()

    print("----------------------------------------------")
    print("Model Processing Time")
    print("----------------------------------------------")

    print_stats(
        "Palm TRT",
        palm_detection_times
    )

    print_stats(
        "Landmark Detection",
        landmark_detection_times
    )

    print_stats(
        "Landmark Tracking",
        landmark_tracking_times
    )

    print()

    # =====================================
    # Detection vs Tracking
    # =====================================

    if (
        detection_success_times
        and tracking_times
    ):
        detection_avg = float(
            np.mean(
                detection_success_times
            )
        )

        tracking_avg = float(
            np.mean(
                tracking_times
            )
        )

        saved_ms = (
            detection_avg
            - tracking_avg
        )

        reduction = (
            saved_ms
            / detection_avg
            * 100.0
        )

        detection_theoretical_fps = (
            1000.0
            / detection_avg
            if detection_avg > 0
            else 0.0
        )

        tracking_theoretical_fps = (
            1000.0
            / tracking_avg
            if tracking_avg > 0
            else 0.0
        )

        print("----------------------------------------------")
        print("Tracking Effect")
        print("----------------------------------------------")

        print(
            f"Detection AVG            : "
            f"{detection_avg:.3f} ms"
        )

        print(
            f"Tracking AVG             : "
            f"{tracking_avg:.3f} ms"
        )

        print(
            f"Saved per frame          : "
            f"{saved_ms:.3f} ms"
        )

        print(
            f"Processing reduction     : "
            f"{reduction:.1f} %"
        )

        print()

        print(
            f"Detection theoretical FPS: "
            f"{detection_theoretical_fps:.1f}"
        )

        print(
            f"Tracking theoretical FPS : "
            f"{tracking_theoretical_fps:.1f}"
        )

        print()

    # =====================================
    # Whole Frame
    # =====================================

    if frame_times:
        frame_avg = float(
            np.mean(
                frame_times
            )
        )

        frame_fps = (
            1000.0 / frame_avg
            if frame_avg > 0
            else 0.0
        )

        print("----------------------------------------------")
        print("Application")
        print("----------------------------------------------")

        print(
            f"Frame AVG                : "
            f"{frame_avg:.3f} ms"
        )

        print(
            f"Measured FPS             : "
            f"{frame_fps:.1f}"
        )

        print()

    # =====================================
    # Palm Skip
    # =====================================

    print("----------------------------------------------")
    print("Palm Detection Skip")
    print("----------------------------------------------")

    print(
        f"Tracking Palm violations : "
        f"{tracking_palm_violation_count}"
    )

    if tracking_palm_violation_count == 0:
        print(
            "Palm skip                : OK"
        )

    else:
        print(
            "Palm skip                : NG"
        )

    print()
    print("==============================================")


# =====================================
# Startup Benchmark
# =====================================

startup_start = time.perf_counter()


# -----------------------------
# PipeTRT Initialize
# -----------------------------

print(
    "PipeTRT initialization start"
)

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

print(
    "Camera initialization start"
)

camera_init_start = time.perf_counter()


# VideoCapture

start = time.perf_counter()

cap = cv2.VideoCapture(
    CAMERA_ID,
    cv2.CAP_V4L2
)

print(
    f"VideoCapture : "
    f"{time.perf_counter() - start:.2f} sec"
)


# Width

start = time.perf_counter()

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    CAMERA_WIDTH
)

print(
    f"Set Width    : "
    f"{time.perf_counter() - start:.2f} sec"
)

## MJPG
cap.set(
    cv2.CAP_PROP_FOURCC,
    cv2.VideoWriter_fourcc(*"MJPG")
)


# Height

start = time.perf_counter()

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    CAMERA_HEIGHT
)

print(
    f"Set Height   : "
    f"{time.perf_counter() - start:.2f} sec"
)


# FPS

start = time.perf_counter()

cap.set(
    cv2.CAP_PROP_FPS,
    CAMERA_FPS
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

camera_init_end = time.perf_counter()

print(
    f"Camera Total : "
    f"{camera_init_end - camera_init_start:.2f} sec"
)


# -----------------------------
# Window Initialize
# -----------------------------

print(
    "Window initialization start"
)

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


# =====================================
# Startup Result
# =====================================

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

print(
    f"Warmup: {WARMUP_FRAMES} frames"
)

print()


# =====================================
# Runtime State
# =====================================

smoothed_fps = 0.0

last_mode = None

frame_count = 0


# =====================================
# Main Loop
# =====================================

while True:
    frame_start = time.perf_counter()


    # =====================================
    # Camera
    # =====================================

    camera_start = time.perf_counter()

    ret, frame = cap.read()

    camera_end = time.perf_counter()

    if not ret:
        print(
            "Camera read failed"
        )

        break


    # =====================================
    # PipeTRT
    # =====================================

    detect_start = time.perf_counter()

    result = landmarker.detect(
        frame
    )

    detect_end = time.perf_counter()

    height, width = frame.shape[:2]

    timings = result.timings


    # =====================================
    # Mode
    # =====================================

    mode = timings.get(
        "mode",
        "UNKNOWN"
    )

    if mode != last_mode:
        print(
            f"[MODE] "
            f"{last_mode} -> {mode}"
        )

        last_mode = mode


    # =====================================
    # Draw Start
    # =====================================

    draw_start = time.perf_counter()


    # -------------------------------------
    # Palm bbox
    # Blue
    # -------------------------------------

    if result.palm_result:
        palm = result.palm_result[0]

        (
            x_min,
            y_min,
            x_max,
            y_max
        ) = palm["bbox"]

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
            (
                palm_x1,
                palm_y1
            ),
            (
                palm_x2,
                palm_y2
            ),
            (
                255,
                0,
                0
            ),
            2
        )


    # -------------------------------------
    # Current ROI
    # Green
    # -------------------------------------

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

        rotation_degree = math.degrees(
            roi["rotation"]
        )

        rotated_rect = (
            (
                center_x,
                center_y
            ),
            (
                roi_width,
                roi_height
            ),
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
            (
                0,
                255,
                0
            ),
            2
        )


    # -------------------------------------
    # 21 Landmarks
    # Yellow
    # -------------------------------------

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
            (
                x,
                y
            ),
            4,
            (
                0,
                255,
                255
            ),
            -1
        )


    # -------------------------------------
    # Next Tracking ROI
    # Purple
    #
    # Debug visualization only
    # -------------------------------------

    tracking_roi = None

    if len(
        result.hand_landmarks
    ) == 21:

        tracking_roi = (
            create_tracking_roi(
                result.hand_landmarks,
                width,
                height
            )
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

        tracking_box = (
            tracking_box.astype(
                np.int32
            )
        )

        cv2.polylines(
            frame,
            [tracking_box],
            True,
            (
                255,
                0,
                255
            ),
            2
        )


    draw_end = time.perf_counter()


    # =====================================
    # Performance
    # =====================================

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
        1000.0
        / total_ms
        if total_ms > 0
        else 0.0
    )

    if smoothed_fps == 0.0:
        smoothed_fps = (
            current_fps
        )

    else:
        smoothed_fps = (
            smoothed_fps
            * 0.9
            + current_fps
            * 0.1
        )


    # =====================================
    # Benchmark Collection
    # =====================================

    frame_count += 1

    measuring = (
        frame_count
        > WARMUP_FRAMES
    )

    if measuring:
        total_measured_frames += 1

        frame_times.append(
            total_ms
        )

        pipetrt_total_ms = (
            timings.get(
                "total_ms",
                detect_ms
            )
        )

        palm_trt_ms = (
            timings.get(
                "palm_trt_ms",
                0.0
            )
        )

        landmark_trt_ms = (
            timings.get(
                "landmark_trt_ms",
                0.0
            )
        )
        
        if landmark_trt_ms > 100.0: print(f"\n!!!!!!!! LANDMARK OUTLIER !!!!!!!!\nTime         : {time.strftime('%H:%M:%S')}\nFrame        : {frame_count}\nMode         : {mode}\nLandmark TRT : {landmark_trt_ms:.3f} ms\nPalm TRT     : {palm_trt_ms:.3f} ms\nPipeTRT Total: {pipetrt_total_ms:.3f} ms\nCamera       : {camera_ms:.3f} ms\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")

        # ---------------------------------
        # DETECTION
        # ---------------------------------

        if mode == "DETECTION":

            # Palmを実行した時間
            palm_detection_times.append(
                palm_trt_ms
            )

            # 手を発見して
            # Landmarkまで実行したDetection
            if (
                len(
                    result.hand_landmarks
                )
                == 21
            ):
                detection_success_times.append(
                    pipetrt_total_ms
                )

                landmark_detection_times.append(
                    landmark_trt_ms
                )

                detection_success_count += 1

            # Palmで探したけど
            # 手がいなかったDetection
            else:
                detection_search_times.append(
                    pipetrt_total_ms
                )

                detection_search_count += 1


        # ---------------------------------
        # TRACKING
        # ---------------------------------

        elif mode == "TRACKING":
            tracking_times.append(
                pipetrt_total_ms
            )

            landmark_tracking_times.append(
                landmark_trt_ms
            )

            tracking_count += 1

            # Tracking中にPalm TRTが
            # 動いていたら異常
            if (
                palm_trt_ms
                > PALM_SKIP_EPSILON_MS
            ):
                tracking_palm_violation_count += 1


        # ---------------------------------
        # TRACKING LOST
        # ---------------------------------

        elif mode == "TRACKING_LOST":
            tracking_lost_times.append(
                pipetrt_total_ms
            )

            tracking_lost_count += 1


    # =====================================
    # Performance Draw
    # =====================================

    warmup_text = (
        "MEASURING"
        if measuring
        else (
            f"WARMUP "
            f"{frame_count}/{WARMUP_FRAMES}"
        )
    )

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
        f"MODE         : {mode}",
        f"BENCHMARK    : {warmup_text}",
    ]

    text_y = 30

    for line in performance_lines:
        cv2.putText(
            frame,
            line,
            (
                20,
                text_y
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (
                255,
                255,
                255
            ),
            2,
            cv2.LINE_AA
        )

        text_y += 30


    # =====================================
    # Mode Big Label
    # =====================================

    cv2.putText(
        frame,
        f"MODE: {mode}",
        (
            width - 350,
            50
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (
            255,
            255,
            255
        ),
        2,
        cv2.LINE_AA
    )


    # =====================================
    # Windows
    # =====================================

    if result.roi_image is not None:
        cv2.imshow(
            "Landmark Input ROI",
            result.roi_image
        )

    cv2.imshow(
        "PipeTRT TensorRT Benchmark",
        frame
    )


    # =====================================
    # Keyboard
    # =====================================

    key = (
        cv2.waitKey(1)
        & 0xFF
    )

    if key in (
        27,
        ord("q")
    ):
        break


# =====================================
# Result
# =====================================

print_benchmark_result()


# =====================================
# Cleanup
# =====================================

landmarker.close()

cap.release()

cv2.destroyAllWindows()
