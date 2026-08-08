import cv2
import time
import numpy as np
import onnxruntime as ort

from pipetrt.palm.preprocess import preprocess
from pipetrt.palm.decoder import decode
from pipetrt.palm.anchors import generate_anchors
from pipetrt.engine.runtime import TensorRTRuntime


MODEL_PATH = "models/palm_detection.onnx"
ENGINE_PATH = "models/palm_detection.engine"

anchors = generate_anchors()


# ==========================================
# ONNX Runtime
# ==========================================

onnx_session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

onnx_input_name = onnx_session.get_inputs()[0].name


def onnx_infer(palm_input):
    start = time.perf_counter()

    outputs = onnx_session.run(
        None,
        {
            onnx_input_name: palm_input
        }
    )

    elapsed_ms = (
        time.perf_counter() - start
    ) * 1000.0

    boxes = outputs[0]
    scores = outputs[1]

    return boxes, scores, elapsed_ms


# ==========================================
# TensorRT
# ==========================================

trt_runtime = TensorRTRuntime(
    ENGINE_PATH
)


def trt_infer(palm_input):
    start = time.perf_counter()

    outputs = trt_runtime.infer(
        palm_input
    )

    elapsed_ms = (
        time.perf_counter() - start
    ) * 1000.0

    boxes = outputs[0]
    scores = outputs[1]

    return boxes, scores, elapsed_ms


# ==========================================
# 描画
# ==========================================

def draw_results(
    image,
    results,
    title,
    inference_ms,
    fps
):
    height, width = image.shape[:2]

    for result in results:
        score = result["score"]
        bbox = result["bbox"]
        keypoints = result["keypoints"]

        x1 = int(bbox[0] * width)
        y1 = int(bbox[1] * height)
        x2 = int(bbox[2] * width)
        y2 = int(bbox[3] * height)

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        for keypoint in keypoints:
            x = int(
                keypoint[0] * width
            )

            y = int(
                keypoint[1] * height
            )

            cv2.circle(
                image,
                (x, y),
                3,
                (0, 0, 255),
                -1
            )

        cv2.putText(
            image,
            f"Score: {score:.3f}",
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1
        )

    cv2.putText(
        image,
        title,
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        image,
        f"Inference: {inference_ms:.2f} ms",
        (10, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1
    )

    cv2.putText(
        image,
        f"FPS: {fps:.1f}",
        (10, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1
    )

    return image


# ==========================================
# Camera
# ==========================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError(
        "カメラを開けませんでした"
    )


last_time = time.perf_counter()

onnx_fps = 0.0
trt_fps = 0.0


while True:
    success, frame = camera.read()

    if not success:
        break

    # --------------------------------------
    # Preprocess
    # --------------------------------------

    palm_input = preprocess(
        frame
    )

    display_base = palm_input[0]

    display_base = (
        display_base * 255.0
    ).clip(
        0,
        255
    ).astype(
        np.uint8
    )

    display_base = cv2.cvtColor(
        display_base,
        cv2.COLOR_RGB2BGR
    )

    # --------------------------------------
    # ONNX Runtime
    # --------------------------------------

    onnx_start = time.perf_counter()

    onnx_boxes, onnx_scores, onnx_ms = onnx_infer(
        palm_input
    )

    onnx_results = decode(
        onnx_boxes,
        onnx_scores,
        anchors
    )

    onnx_total = (
        time.perf_counter() - onnx_start
    )

    if onnx_total > 0:
        onnx_fps = 1.0 / onnx_total

    # --------------------------------------
    # TensorRT
    # --------------------------------------

    trt_start = time.perf_counter()

    trt_boxes, trt_scores, trt_ms = trt_infer(
        palm_input
    )

    trt_results = decode(
        trt_boxes,
        trt_scores,
        anchors
    )

    trt_total = (
        time.perf_counter() - trt_start
    )

    if trt_total > 0:
        trt_fps = 1.0 / trt_total

    # --------------------------------------
    # Raw Output Difference
    # --------------------------------------

    box_diff = np.max(
        np.abs(
            onnx_boxes - trt_boxes
        )
    )

    score_diff = np.max(
        np.abs(
            onnx_scores - trt_scores
        )
    )

    # --------------------------------------
    # Draw
    # --------------------------------------

    onnx_image = display_base.copy()

    onnx_image = draw_results(
        onnx_image,
        onnx_results,
        "ONNX Runtime CPU",
        onnx_ms,
        onnx_fps
    )

    trt_image = display_base.copy()

    trt_image = draw_results(
        trt_image,
        trt_results,
        "TensorRT GPU",
        trt_ms,
        trt_fps
    )

    combined = np.hstack(
        [
            onnx_image,
            trt_image
        ]
    )

    combined = cv2.resize(
        combined,
        (1152, 576),
        interpolation=cv2.INTER_NEAREST
    )

    cv2.putText(
        combined,
        f"Max Box Diff: {box_diff:.6f}",
        (10, 550),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1
    )

    cv2.putText(
        combined,
        f"Max Score Diff: {score_diff:.6f}",
        (350, 550),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1
    )

    cv2.imshow(
        "PipeTRT ONNX vs TensorRT",
        combined
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == 27:
        break


camera.release()
cv2.destroyAllWindows()