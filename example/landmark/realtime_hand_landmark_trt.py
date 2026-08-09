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

CAMERA_TESTS = (
    {
        "name": "DSHOW 1280x720 60FPS",
        "backend": cv2.CAP_DSHOW,
        "width": 1280,
        "height": 720,
        "fps": 60,
    },
    {
        "name": "DSHOW 640x480 60FPS",
        "backend": cv2.CAP_DSHOW,
        "width": 640,
        "height": 480,
        "fps": 60,
    },
    {
        "name": "MSMF 1280x720 60FPS",
        "backend": cv2.CAP_MSMF,
        "width": 1280,
        "height": 720,
        "fps": 60,
    },
)


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


def get_fourcc_text(camera):
    fourcc_value = int(
        camera.get(cv2.CAP_PROP_FOURCC)
    )

    return "".join(
        chr(
            (fourcc_value >> (8 * index))
            & 0xFF
        )
        for index in range(4)
    )


def create_camera(config):
    camera = cv2.VideoCapture(
        CAMERA_ID,
        config["backend"],
    )

    if not camera.isOpened():
        return None

    # 可能ならMJPEGを要求
    camera.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*"MJPG"),
    )

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        config["width"],
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        config["height"],
    )

    camera.set(
        cv2.CAP_PROP_FPS,
        config["fps"],
    )

    return camera


def benchmark_camera(
    config,
    test_frames=60,
):
    print()
    print(
        f"===== {config['name']} ====="
    )

    camera = create_camera(config)

    if camera is None:
        print("カメラを開けませんでした")
        return None

    try:
        # 起動直後の不安定なフレームを捨てる
        for _ in range(10):
            camera.read()

        read_times = []

        for _ in range(test_frames):
            start_time = time.perf_counter()

            success, frame = camera.read()

            end_time = time.perf_counter()

            if not success:
                print(
                    "フレーム取得に失敗しました"
                )
                return None

            read_times.append(
                (end_time - start_time)
                * 1000.0
            )

        average_ms = float(
            np.mean(read_times)
        )

        minimum_ms = float(
            np.min(read_times)
        )

        maximum_ms = float(
            np.max(read_times)
        )

        measured_fps = (
            1000.0 / average_ms
            if average_ms > 0
            else 0.0
        )

        width = camera.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )

        height = camera.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )

        requested_fps = camera.get(
            cv2.CAP_PROP_FPS
        )

        fourcc = get_fourcc_text(camera)

        print(
            f"Resolution : "
            f"{width:.0f}x{height:.0f}"
        )

        print(
            f"FPS setting: "
            f"{requested_fps:.2f}"
        )

        print(
            f"FOURCC     : {fourcc}"
        )

        print(
            f"Read avg   : "
            f"{average_ms:.2f} ms"
        )

        print(
            f"Read min   : "
            f"{minimum_ms:.2f} ms"
        )

        print(
            f"Read max   : "
            f"{maximum_ms:.2f} ms"
        )

        print(
            f"Measured FPS: "
            f"{measured_fps:.2f}"
        )

        return {
            "config": config,
            "average_ms": average_ms,
            "measured_fps": measured_fps,
            "fourcc": fourcc,
        }

    finally:
        camera.release()


def select_best_camera():
    print(
        "Camera benchmark starting..."
    )

    results = []

    for config in CAMERA_TESTS:
        result = benchmark_camera(
            config
        )

        if result is not None:
            results.append(result)

    if not results:
        raise RuntimeError(
            "使用可能なカメラ設定がありません"
        )

    best_result = min(
        results,
        key=lambda result:
        result["average_ms"],
    )

    print()
    print("==========================")
    print("Best camera configuration")
    print("==========================")

    print(
        best_result["config"]["name"]
    )

    print(
        f"Camera read: "
        f"{best_result['average_ms']:.2f} ms"
    )

    print(
        f"Measured FPS: "
        f"{best_result['measured_fps']:.2f}"
    )

    print(
        f"FOURCC: "
        f"{best_result['fourcc']}"
    )

    print("==========================")
    print()

    return best_result["config"]


class HandLandmarkTensorRT:
    def __init__(
        self,
        engine_path: Path,
    ):
        if not engine_path.exists():
            raise FileNotFoundError(
                f"Engineが見つかりません: "
                f"{engine_path}"
            )

        self.logger = trt.Logger(
            trt.Logger.WARNING
        )

        self.runtime = trt.Runtime(
            self.logger
        )

        engine_data = (
            engine_path.read_bytes()
        )

        self.engine = (
            self.runtime
            .deserialize_cuda_engine(
                engine_data
            )
        )

        if self.engine is None:
            raise RuntimeError(
                "Engineの読み込みに失敗しました"
            )

        self.context = (
            self.engine
            .create_execution_context()
        )

        if self.context is None:
            raise RuntimeError(
                "ExecutionContextの"
                "作成に失敗しました"
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
                self.engine
                .get_tensor_name(
                    tensor_index
                )
            )

            tensor_mode = (
                self.engine
                .get_tensor_mode(
                    tensor_name
                )
            )

            tensor_shape = tuple(
                self.context
                .get_tensor_shape(
                    tensor_name
                )
            )

            tensor_dtype = trt.nptype(
                self.engine
                .get_tensor_dtype(
                    tensor_name
                )
            )

            if any(
                size < 0
                for size in tensor_shape
            ):
                raise RuntimeError(
                    f"動的Shape未設定: "
                    f"{tensor_name} "
                    f"{tensor_shape}"
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
                self.context
                .set_tensor_address(
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
                self.input_name = (
                    tensor_name
                )

            else:
                self.output_names.append(
                    tensor_name
                )

    def infer(
        self,
        input_data: np.ndarray,
    ):
        host_input = (
            self.host_buffers[
                self.input_name
            ]
        )

        np.copyto(
            host_input,
            input_data,
        )

        input_device = (
            self.device_buffers[
                self.input_name
            ]
        )

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
            self.context
            .execute_async_v3(
                stream_handle=int(
                    self.stream
                )
            )
        )

        if not success:
            raise RuntimeError(
                "TensorRT推論に失敗しました"
            )

        for output_name in (
            self.output_names
        ):
            host_output = (
                self.host_buffers[
                    output_name
                ]
            )

            device_output = (
                self.device_buffers[
                    output_name
                ]
            )

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
        (224, 224),
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
            + landmark[0]
            / 224.0
            * crop_width
        )

        y = int(
            top
            + landmark[1]
            / 224.0
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
                point[0] + 4,
                point[1] - 4,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def draw_performance(
    frame,
    fps,
    camera_ms,
    preprocess_ms,
    inference_ms,
    draw_ms,
    total_ms,
):
    lines = (
        (
            f"FPS      : {fps:.1f}",
            (0, 255, 0),
        ),
        (
            f"Camera   : {camera_ms:.2f} ms",
            (255, 255, 255),
        ),
        (
            f"Pre      : {preprocess_ms:.2f} ms",
            (255, 255, 255),
        ),
        (
            f"TensorRT : {inference_ms:.2f} ms",
            (255, 255, 255),
        ),
        (
            f"Draw     : {draw_ms:.2f} ms",
            (255, 255, 255),
        ),
        (
            f"Total    : {total_ms:.2f} ms",
            (0, 255, 255),
        ),
    )

    y = 35

    for text, color in lines:
        cv2.putText(
            frame,
            text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )

        y += 30


def run_realtime(
    camera_config,
):
    hand_model = (
        HandLandmarkTensorRT(
            ENGINE_PATH
        )
    )

    camera = create_camera(
        camera_config
    )

    if camera is None:
        hand_model.close()

        raise RuntimeError(
            "カメラを開けませんでした"
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

            draw_start = (
                time.perf_counter()
            )

            landmarks = (
                get_landmarks(
                    outputs
                )
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
            ) * 1000

            preprocess_ms = (
                preprocess_end
                - preprocess_start
            ) * 1000

            inference_ms = (
                inference_end
                - inference_start
            ) * 1000

            draw_ms = (
                draw_end
                - draw_start
            ) * 1000

            total_ms = (
                draw_end
                - frame_start
            ) * 1000

            current_fps = (
                1000.0 / total_ms
                if total_ms > 0
                else 0.0
            )

            if smoothed_fps == 0:
                smoothed_fps = (
                    current_fps
                )
            else:
                smoothed_fps = (
                    smoothed_fps * 0.9
                    + current_fps * 0.1
                )

            draw_performance(
                frame,
                smoothed_fps,
                camera_ms,
                preprocess_ms,
                inference_ms,
                draw_ms,
                total_ms,
            )

            cv2.imshow(
                "PipeTRT Hand Landmark",
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


def main():
    best_camera_config = (
        select_best_camera()
    )

    run_realtime(
        best_camera_config
    )


if __name__ == "__main__":
    main()