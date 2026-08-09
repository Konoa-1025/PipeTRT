from pathlib import Path

import numpy as np
import tensorrt as trt

try:
    from cuda.bindings import runtime as cudart
except ImportError:
    from cuda import cudart

from pipetrt.landmark.preprocess import preprocess_frame


ENGINE_PATH = Path(
    "engines/hand_landmark_fp32.engine"
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


class TensorRTInference:
    def __init__(
        self,
        engine_path=ENGINE_PATH
    ):
        engine_path = Path(engine_path)

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

        engine_data = engine_path.read_bytes()

        self.engine = (
            self.runtime.deserialize_cuda_engine(
                engine_data
            )
        )

        if self.engine is None:
            raise RuntimeError(
                "Engineの読み込みに失敗しました"
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
                dtype=tensor_dtype
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
                    int(device_buffer)
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

    def infer_frame(self, frame):
        input_data = preprocess_frame(
            frame
        )

        return self.run(
            input_data
        )

    def run(
        self,
        input_data
    ):
        host_input = self.host_buffers[
            self.input_name
        ]

        np.copyto(
            host_input,
            input_data
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
                self.stream
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
                    self.stream
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