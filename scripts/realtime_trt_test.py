from pathlib import Path
import time

import cv2
import numpy as np
import tensorrt as trt

try:
    from cuda.bindings import runtime as cudart
except ImportError:
    from cuda import cudart


ENGINE_PATH = Path(
    "engines/hand_landmark_fp32.engine"
)

CAMERA_ID = 0

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 60

MODEL_WIDTH = 224
MODEL_HEIGHT = 224


HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
)


def check_cuda(result):
    error_code = result[0]

    if error_code != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(
            f"CUDA APIエラー: {error_code}"
        )

    if len(result) == 1:
        return None

    if len(result) == 2:
        return result[1]

    return result[1:]


class HandLandmarkTensorRT:
    def __init__(self, engine_path):
        if not engine_path.exists():
            raise FileNotFoundError(
                f"Engineが見つかりません: {engine_path}"
            )

        self.logger = trt.Logger(
            trt.Logger.WARNING
        )

        self.runtime = trt.Runtime(
            self.logger
        )

        engine_data = engine_path.read_bytes()

        self.engine = (
            self.runtime.deserialize_cuda_engine(
                engine_data
            )
        )

        if self.engine is None:
            raise RuntimeError(
                "TensorRT Engineの読み込みに失敗しました"
            )

        self.context = (
            self.engine.create_execution_context()
        )

        if self.context is None:
            raise RuntimeError(
                "ExecutionContextの作成に失敗しました"
            )

        self.stream = check_cuda(
            cudart.cudaStreamCreate()
        )

        self.input_name = None
        self.output_names = []

        self.host_buffers = {}
        self.device_buffers = {}

        self.create_buffers()

    def create_buffers(self):
        for tensor_index in range(
            self.engine.num_io_tensors
        ):
            tensor_name = (
                self.engine.get_tensor_name(
                    tensor_index
                )
            )

            tensor_mode = (
                self.engine.get_tensor_mode(
                    tensor_name
                )
            )

            tensor_shape = tuple(
                self.context.get_tensor_shape(
                    tensor_name
                )
            )

            tensor_dtype = trt.nptype(
                self.engine.get_tensor_dtype(
                    tensor_name
                )
            )

            if any(
                dimension < 0
                for dimension in tensor_shape
            ):
                raise RuntimeError(
                    f"動的Shape未設定: "
                    f"{tensor_name} {tensor_shape}"
                )

            host_buffer = np.empty(
                tensor_shape,
                dtype=tensor_dtype,
            )

            device_buffer = check_cuda(
                cudart.cudaMalloc(
                    host_buffer.nbytes
                )
            )

            self.host_buffers[
                tensor_name
            ] = host_buffer

            self.device_buffers[
                tensor_name
            ] = device_buffer

            success = (
                self.context.set_tensor_address(
                    tensor_name,
                    int(device_buffer),
                )
            )

            if not success:
                raise RuntimeError(
                    f"Tensorアドレス設定失敗: "
                    f"{tensor_name}"
                )

            if (
                tensor_mode
                == trt.TensorIOMode.INPUT
            ):
                self.input_name = tensor_name

            else:
                self.output_names.append(
                    tensor_name
                )

    def infer(self, input_data):
        host_input = self.host_buffers[
            self.input_name
        ]

        np.copyto(
            host_input,
            input_data,
        )

        input_device = self.device_buffers[
            self.input_name
        ]

        check_cuda(
            cudart.cudaMemcpyAsync(
                input_device,
                host_input.ctypes.data,
                host_input.nbytes,
                cudart.cudaMemcpyKind
                .cudaMemcpyHostToDevice,
                self.stream,
            )
        )

        success = (
            self.context.execute_async_v3(
                stream_handle=int(
                    self.stream
                )
            )
        )

        if not success:
            raise RuntimeError(
                "TensorRT推論に失敗しました"
            )

        for output_name in self.output_names:
            host_output = self.host_buffers[
                output_name
            ]

            device_output = self.device_buffers[
                output_name
            ]

            check_cuda(
                cudart.cudaMemcpyAsync(
                    host_output.ctypes.data,
                    device_output,
                    host_output.nbytes,
                    cudart.cudaMemcpyKind
                    .cudaMemcpyDeviceToHost,
                    self.stream,
                )
            )

        check_cuda(
            cudart.cudaStreamSynchronize(
                self.stream
            )
        )

        return {
            output_name:
            self.host_buffers[
                output_name
            ].copy()
            for output_name
            in self.output_names
        }

    def close(self):
        for device_buffer in (
            self.device_buffers.values()
        ):
            check_cuda(
                cudart.cudaFree(
                    device_buffer
                )
            )

        self.device_buffers.clear()

        if self.stream is not None:
            check_cuda(
                cudart.cudaStreamDestroy(
                    self.stream
                )
            )

            self.stream = None


def create_camera():
    camera = cv2.VideoCapture(
        CAMERA_ID,
        cv2.CAP_MSMF,
    )

    if not camera.isOpened():
        raise RuntimeError(
            "カメラを開けませんでした"
        )

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        CAMERA_WIDTH,
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        CAMERA_HEIGHT,
    )

    camera.set(
        cv2.CAP_PROP_FPS,
        CAMERA_FPS,
    )

    return camera


def crop_center_square(frame):
    height, width = frame.shape[:2]

    crop_size = int(
        min(width, height) * 0.75
    )

    left = (
        width - crop_size
    ) // 2

    top = (
        height - crop_size
    ) // 2

    right = left + crop_size
    bottom = top + crop_size

    crop = frame[
        top:bottom,
        left:right,
    ]

    return (
        crop,
        (left, top, right, bottom),
    )


def preprocess_image(crop):
    resized = cv2.resize(
        crop,
        (
            MODEL_WIDTH,
            MODEL_HEIGHT,
        ),
        interpolation=cv2.INTER_LINEAR,
    )

    rgb = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2RGB,
    )

    normalized = (
        rgb.astype(np.float32)
        / 255.0
    )

    chw = np.transpose(
        normalized,
        (2, 0, 1),
    )

    input_data = np.expand_dims(
        chw,
        axis=0,
    )

    return np.ascontiguousarray(
        input_data,
        dtype=np.float32,
    )


def get_landmarks(outputs):
    if "Identity" not in outputs:
        raise KeyError(
            "Identity出力が見つかりません"
        )

    return outputs[
        "Identity"
    ].reshape(
        21,
        3,
    )


def draw_landmarks(
    frame,
    landmarks,
    crop_rectangle,
):
    left, top, right, bottom = (
        crop_rectangle
    )

    crop_width = right - left
    crop_height = bottom - top

    points = []

    for landmark in landmarks:
        x = int(
            left
            + (
                float(landmark[0])
                / MODEL_WIDTH
            )
            * crop_width
        )

        y = int(
            top
            + (
                float(landmark[1])
                / MODEL_HEIGHT
            )
            * crop_height
        )

        points.append(
            (x, y)
        )

    for start, end in HAND_CONNECTIONS:
        cv2.line(
            frame,
            points[start],
            points[end],
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    for index, point in enumerate(
        points
    ):
        cv2.circle(
            frame,
            point,
            4,
            (0, 0, 255),
            -1,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            str(index),
            (
                point[0] + 5,
                point[1] - 5,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def draw_information(
    frame,
    pipeline_fps,
    camera_ms,
    preprocess_ms,
    inference_ms,
    postprocess_ms,
    draw_ms,
    total_ms,
    presence_value,
    handedness_value,
):
    information = (
        f"Pipeline FPS : {pipeline_fps:.1f}",
        f"Camera       : {camera_ms:.2f} ms",
        f"Preprocess   : {preprocess_ms:.2f} ms",
        f"TensorRT     : {inference_ms:.2f} ms",
        f"Postprocess  : {postprocess_ms:.2f} ms",
        f"Draw         : {draw_ms:.2f} ms",
        f"Total        : {total_ms:.2f} ms",
        "",
        f"Camera       : MSMF",
        f"Resolution   : {CAMERA_WIDTH}x{CAMERA_HEIGHT}",
        f"Requested FPS: {CAMERA_FPS}",
        f"Model Input  : {MODEL_WIDTH}x{MODEL_HEIGHT}",
        f"Engine       : FP32",
        f"Presence     : {presence_value:.4f}",
        f"Identity_2   : {handedness_value:.4f}",
        f"Landmarks    : 21",
    )

    y = 30

    for text in information:
        if text == "":
            y += 10
            continue

        cv2.putText(
            frame,
            text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        y += 24


def main():
    print("PipeTRT realtime test")
    print()

    print(
        f"Engine: {ENGINE_PATH}"
    )

    hand_model = (
        HandLandmarkTensorRT(
            ENGINE_PATH
        )
    )

    camera = create_camera()

    print(
        "Camera backend: MSMF"
    )

    print(
        "Resolution:",
        camera.get(
            cv2.CAP_PROP_FRAME_WIDTH
        ),
        "x",
        camera.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        ),
    )

    print(
        "FPS:",
        camera.get(
            cv2.CAP_PROP_FPS
        ),
    )

    smoothed_fps = 0.0

    try:
        while True:
            frame_start = (
                time.perf_counter()
            )

            success, frame = (
                camera.read()
            )

            camera_end = (
                time.perf_counter()
            )

            if not success:
                print(
                    "カメラフレーム取得失敗"
                )
                break

            frame = cv2.flip(
                frame,
                1,
            )

            crop, crop_rectangle = (
                crop_center_square(
                    frame
                )
            )

            preprocess_start = (
                time.perf_counter()
            )

            input_data = (
                preprocess_image(
                    crop
                )
            )

            preprocess_end = (
                time.perf_counter()
            )

            inference_start = (
                time.perf_counter()
            )

            outputs = (
                hand_model.infer(
                    input_data
                )
            )

            inference_end = (
                time.perf_counter()
            )

            postprocess_start = (
                time.perf_counter()
            )

            landmarks = get_landmarks(
                outputs
            )

            presence_value = float(
                outputs.get(
                    "Identity_1",
                    np.array([[0.0]]),
                ).flatten()[0]
            )

            handedness_value = float(
                outputs.get(
                    "Identity_2",
                    np.array([[0.0]]),
                ).flatten()[0]
            )

            postprocess_end = (
                time.perf_counter()
            )

            draw_start = (
                time.perf_counter()
            )

            left, top, right, bottom = (
                crop_rectangle
            )

            cv2.rectangle(
                frame,
                (left, top),
                (right, bottom),
                (255, 255, 0),
                2,
            )

            draw_landmarks(
                frame,
                landmarks,
                crop_rectangle,
            )

            draw_end = (
                time.perf_counter()
            )

            camera_ms = (
                camera_end
                - frame_start
            ) * 1000.0

            preprocess_ms = (
                preprocess_end
                - preprocess_start
            ) * 1000.0

            inference_ms = (
                inference_end
                - inference_start
            ) * 1000.0

            postprocess_ms = (
                postprocess_end
                - postprocess_start
            ) * 1000.0

            draw_ms = (
                draw_end
                - draw_start
            ) * 1000.0

            total_ms = (
                draw_end
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

            draw_information(
                frame,
                smoothed_fps,
                camera_ms,
                preprocess_ms,
                inference_ms,
                postprocess_ms,
                draw_ms,
                total_ms,
                presence_value,
                handedness_value,
            )

            cv2.putText(
                frame,
                "Put one hand inside this square",
                (
                    left,
                    max(
                        25,
                        top - 10,
                    ),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(
                "PipeTRT Realtime TensorRT Test",
                frame,
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key in (
                ord("q"),
                27,
            ):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()
        hand_model.close()


if __name__ == "__main__":
    main()