from pathlib import Path
import time

import cv2
import numpy as np
import tensorrt as trt

try:
    from cuda.bindings import runtime as cudart
except ImportError:
    from cuda import cudart


HAND_CONNECTIONS = (
    # 手首から各指
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),

    # 手のひら
    (5, 9), (9, 13), (13, 17), (0, 17),
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
    def __init__(self, engine_path: Path):
        if not engine_path.exists():
            raise FileNotFoundError(
                f"Engineが見つかりません: {engine_path}"
            )

        self.logger = trt.Logger(
            trt.Logger.WARNING
        )

        # runtimeはengineより長く保持する
        self.runtime = trt.Runtime(self.logger)

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
        self.print_engine_information()

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

            if any(size < 0 for size in tensor_shape):
                raise RuntimeError(
                    f"動的Shapeが未設定です: "
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

            self.host_buffers[tensor_name] = (
                host_buffer
            )

            self.device_buffers[tensor_name] = (
                device_buffer
            )

            address_set = (
                self.context.set_tensor_address(
                    tensor_name,
                    int(device_buffer),
                )
            )

            if not address_set:
                raise RuntimeError(
                    f"Tensorアドレスの設定に失敗: "
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

        if self.input_name is None:
            raise RuntimeError(
                "入力Tensorが見つかりません"
            )

    def print_engine_information(self):
        print("--- TensorRT Engine ---")

        for tensor_index in range(
            self.engine.num_io_tensors
        ):
            tensor_name = (
                self.engine.get_tensor_name(
                    tensor_index
                )
            )

            print(
                f"name={tensor_name}, "
                f"mode="
                f"{self.engine.get_tensor_mode(tensor_name)}, "
                f"shape="
                f"{self.context.get_tensor_shape(tensor_name)}, "
                f"dtype="
                f"{self.engine.get_tensor_dtype(tensor_name)}"
            )

        print("-----------------------")

    def infer(
        self,
        input_data: np.ndarray,
    ) -> dict[str, np.ndarray]:
        expected_input = self.host_buffers[
            self.input_name
        ]

        if input_data.shape != expected_input.shape:
            raise ValueError(
                f"入力Shapeが違います: "
                f"expected={expected_input.shape}, "
                f"actual={input_data.shape}"
            )

        np.copyto(
            expected_input,
            input_data.astype(
                expected_input.dtype,
                copy=False,
            ),
        )

        input_device = self.device_buffers[
            self.input_name
        ]

        check_cuda(
            cudart.cudaMemcpyAsync(
                input_device,
                expected_input.ctypes.data,
                expected_input.nbytes,
                cudart.cudaMemcpyKind
                .cudaMemcpyHostToDevice,
                self.stream,
            )
        )

        inference_success = (
            self.context.execute_async_v3(
                stream_handle=int(self.stream)
            )
        )

        if not inference_success:
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
            self.host_buffers[output_name].copy()
            for output_name in self.output_names
        }

    def close(self):
        for device_buffer in (
            self.device_buffers.values()
        ):
            check_cuda(
                cudart.cudaFree(device_buffer)
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
    frame_height, frame_width = frame.shape[:2]

    crop_size = int(
        min(frame_width, frame_height) * 0.75
    )

    left = (frame_width - crop_size) // 2
    top = (frame_height - crop_size) // 2

    right = left + crop_size
    bottom = top + crop_size

    crop = frame[
        top:bottom,
        left:right,
    ]

    return crop, (left, top, right, bottom)


def preprocess_image(crop):
    resized = cv2.resize(
        crop,
        (224, 224),
        interpolation=cv2.INTER_LINEAR,
    )

    # OpenCVはBGRなのでRGBへ変換
    rgb_image = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2RGB,
    )

    # 0～255から0～1へ変換
    normalized = (
        rgb_image.astype(np.float32)
        / 255.0
    )

    # HWCからCHWへ変換
    chw_image = np.transpose(
        normalized,
        (2, 0, 1),
    )

    # バッチ次元を追加
    input_data = np.expand_dims(
        chw_image,
        axis=0,
    )

    return np.ascontiguousarray(
        input_data,
        dtype=np.float32,
    )


def get_landmarks(outputs):
    if "Identity" not in outputs:
        raise KeyError(
            "出力Identityが見つかりません"
        )

    landmarks = outputs[
        "Identity"
    ].reshape(21, 3)

    return landmarks


def draw_landmarks(
    frame,
    landmarks,
    crop_rectangle,
):
    left, top, right, bottom = crop_rectangle

    crop_width = right - left
    crop_height = bottom - top

    points = []

    for landmark in landmarks:
        model_x = float(landmark[0])
        model_y = float(landmark[1])

        frame_x = int(
            left
            + model_x / 224.0 * crop_width
        )

        frame_y = int(
            top
            + model_y / 224.0 * crop_height
        )

        points.append(
            (frame_x, frame_y)
        )

    for start_index, end_index in HAND_CONNECTIONS:
        cv2.line(
            frame,
            points[start_index],
            points[end_index],
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    for point_index, point in enumerate(points):
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
            str(point_index),
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


def main():
    engine_path = Path(
        "engines/hand_landmark_fp32.engine"
    )

    hand_model = HandLandmarkTensorRT(
        engine_path
    )

    camera = cv2.VideoCapture(
        0,
        cv2.CAP_DSHOW,
    )

    if not camera.isOpened():
        hand_model.close()

        raise RuntimeError(
            "カメラを開けませんでした: camera_id=0"
        )

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1280,
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        720,
    )

    previous_time = time.perf_counter()
    smoothed_fps = 0.0

    try:
        while True:
            read_success, frame = camera.read()

            if not read_success:
                print(
                    "カメラフレームの取得に失敗しました"
                )
                break

            # 鏡のように表示
            frame = cv2.flip(frame, 1)

            crop, crop_rectangle = (
                crop_center_square(frame)
            )

            input_data = preprocess_image(crop)

            inference_start = time.perf_counter()

            outputs = hand_model.infer(
                input_data
            )

            inference_end = time.perf_counter()

            landmarks = get_landmarks(outputs)

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

            current_time = time.perf_counter()
            frame_time = (
                current_time - previous_time
            )

            previous_time = current_time

            current_fps = (
                1.0 / frame_time
                if frame_time > 0
                else 0.0
            )

            if smoothed_fps == 0.0:
                smoothed_fps = current_fps
            else:
                smoothed_fps = (
                    smoothed_fps * 0.9
                    + current_fps * 0.1
                )

            inference_ms = (
                inference_end
                - inference_start
            ) * 1000.0

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

            cv2.putText(
                frame,
                f"FPS: {smoothed_fps:.1f}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                f"TensorRT: {inference_ms:.2f} ms",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                f"Identity_1: {presence_value:.4f}",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                f"Identity_2: {handedness_value:.4f}",
                (20, 135),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                "Put one hand inside the square",
                (left, max(30, top - 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(
                "PipeTRT Hand Landmark",
                frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()
        hand_model.close()


if __name__ == "__main__":
    main()