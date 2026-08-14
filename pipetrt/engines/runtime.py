import numpy as np
import tensorrt as trt

try:
    from cuda.bindings import runtime as cudart
except ImportError:
    from cuda import cudart


def check_cuda(result):
    error = result[0]

    if error != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(
            f"CUDA APIエラー: {error}"
        )

    if len(result) == 1:
        return None

    if len(result) == 2:
        return result[1]

    return result[1:]


class TensorRTRuntime:

    def __init__(self, engine_path):
        self.logger = trt.Logger(
            trt.Logger.WARNING
        )

        self.runtime = trt.Runtime(
            self.logger
        )

        # Engine読み込み
        with open(engine_path, "rb") as engine_file:
            engine_data = engine_file.read()

        self.engine = (
            self.runtime.deserialize_cuda_engine(
                engine_data
            )
        )

        if self.engine is None:
            raise RuntimeError(
                "TensorRT Engineの読み込みに失敗しました"
            )

        # Execution Context
        self.context = (
            self.engine.create_execution_context()
        )

        if self.context is None:
            raise RuntimeError(
                "Execution Contextの生成に失敗しました"
            )

        # CUDA Stream
        self.stream = check_cuda(
            cudart.cudaStreamCreate()
        )

        self.input_names = []
        self.output_names = []

        self.host_buffers = {}
        self.device_buffers = {}

        self.create_buffers()

    def create_buffers(self):
        for index in range(
            self.engine.num_io_tensors
        ):
            name = self.engine.get_tensor_name(
                index
            )

            mode = self.engine.get_tensor_mode(
                name
            )

            shape = tuple(
                self.context.get_tensor_shape(
                    name
                )
            )

            dtype = trt.nptype(
                self.engine.get_tensor_dtype(
                    name
                )
            )

            if any(
                dimension < 0
                for dimension in shape
            ):
                raise RuntimeError(
                    f"動的Shapeにはまだ未対応です: "
                    f"{name} {shape}"
                )

            host_buffer = np.empty(
                shape,
                dtype=dtype
            )

            device_buffer = check_cuda(
                cudart.cudaMalloc(
                    host_buffer.nbytes
                )
            )

            self.host_buffers[
                name
            ] = host_buffer

            self.device_buffers[
                name
            ] = device_buffer

            success = (
                self.context.set_tensor_address(
                    name,
                    int(device_buffer)
                )
            )

            if not success:
                raise RuntimeError(
                    f"Tensor Addressの設定に失敗: "
                    f"{name}"
                )

            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(
                    name
                )

            elif mode == trt.TensorIOMode.OUTPUT:
                self.output_names.append(
                    name
                )

    def infer(self, input_array):
        input_array = np.ascontiguousarray(
            input_array,
            dtype=np.float32
        )

        input_name = self.input_names[0]

        host_input = self.host_buffers[
            input_name
        ]

        if host_input.shape != input_array.shape:
            raise ValueError(
                f"入力Shapeが一致しません: "
                f"expected={host_input.shape}, "
                f"actual={input_array.shape}"
            )

        # CPU Bufferへコピー
        np.copyto(
            host_input,
            input_array
        )

        # CPU → GPU
        check_cuda(
            cudart.cudaMemcpyAsync(
                self.device_buffers[
                    input_name
                ],
                host_input.ctypes.data,
                host_input.nbytes,
                cudart.cudaMemcpyKind
                .cudaMemcpyHostToDevice,
                self.stream
            )
        )

        # TensorRT推論
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

        # GPU → CPU
        for output_name in self.output_names:
            host_output = self.host_buffers[
                output_name
            ]

            check_cuda(
                cudart.cudaMemcpyAsync(
                    host_output.ctypes.data,
                    self.device_buffers[
                        output_name
                    ],
                    host_output.nbytes,
                    cudart.cudaMemcpyKind
                    .cudaMemcpyDeviceToHost,
                    self.stream
                )
            )

        # GPU処理完了待ち
        check_cuda(
            cudart.cudaStreamSynchronize(
                self.stream
            )
        )

        # 出力
        return {
            output_name:
            self.host_buffers[
                output_name
            ].copy()

            for output_name
            in self.output_names
        }

    def close(self):
        # GPU Memory解放
        for device_buffer in (
            self.device_buffers.values()
        ):
            check_cuda(
                cudart.cudaFree(
                    device_buffer
                )
            )

        self.device_buffers.clear()
        self.host_buffers.clear()

        # CUDA Stream解放
        if self.stream is not None:
            check_cuda(
                cudart.cudaStreamDestroy(
                    self.stream
                )
            )

            self.stream = None