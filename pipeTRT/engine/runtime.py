import numpy as np
import tensorrt as trt

from cuda.bindings import runtime as cudart


class TensorRTRuntime:

    def __init__(self, engine_path):
        self.logger = trt.Logger(
            trt.Logger.WARNING
        )

        self.runtime = trt.Runtime(
            self.logger
        )

        with open(engine_path, "rb") as engine_file:
            engine_data = engine_file.read()

        self.engine = self.runtime.deserialize_cuda_engine(
            engine_data
        )

        if self.engine is None:
            raise RuntimeError(
                "TensorRT Engineの読み込みに失敗しました"
            )

        self.context = self.engine.create_execution_context()

        if self.context is None:
            raise RuntimeError(
                "Execution Contextの生成に失敗しました"
            )

        self.input_names = []
        self.output_names = []

        for index in range(
            self.engine.num_io_tensors
        ):
            name = self.engine.get_tensor_name(
                index
            )

            mode = self.engine.get_tensor_mode(
                name
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

        self.context.set_input_shape(
            input_name,
            input_array.shape
        )

        # -----------------------------
        # Input GPU Memory
        # -----------------------------

        error, device_input = cudart.cudaMalloc(
            input_array.nbytes
        )

        if error != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(
                "GPU入力メモリの確保に失敗しました"
            )

        # -----------------------------
        # Output Memory
        # -----------------------------

        output_arrays = {}
        device_outputs = {}

        for output_name in self.output_names:

            output_shape = tuple(
                self.context.get_tensor_shape(
                    output_name
                )
            )

            output_dtype = trt.nptype(
                self.engine.get_tensor_dtype(
                    output_name
                )
            )

            output_array = np.empty(
                output_shape,
                dtype=output_dtype
            )

            error, device_output = cudart.cudaMalloc(
                output_array.nbytes
            )

            if error != cudart.cudaError_t.cudaSuccess:
                raise RuntimeError(
                    "GPU出力メモリの確保に失敗しました"
                )

            output_arrays[
                output_name
            ] = output_array

            device_outputs[
                output_name
            ] = device_output

        # -----------------------------
        # CUDA Stream
        # -----------------------------

        error, stream = cudart.cudaStreamCreate()

        if error != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(
                "CUDA Streamの生成に失敗しました"
            )

        # -----------------------------
        # CPU → GPU
        # -----------------------------

        cudart.cudaMemcpyAsync(
            device_input,
            input_array.ctypes.data,
            input_array.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            stream
        )

        # -----------------------------
        # Tensor Address設定
        # -----------------------------

        self.context.set_tensor_address(
            input_name,
            int(device_input)
        )

        for output_name in self.output_names:

            self.context.set_tensor_address(
                output_name,
                int(
                    device_outputs[
                        output_name
                    ]
                )
            )

        # -----------------------------
        # TensorRT GPU Inference
        # -----------------------------

        success = self.context.execute_async_v3(
            stream
        )

        if not success:
            raise RuntimeError(
                "TensorRT推論に失敗しました"
            )

        # -----------------------------
        # GPU → CPU
        # -----------------------------

        for output_name in self.output_names:

            output_array = output_arrays[
                output_name
            ]

            cudart.cudaMemcpyAsync(
                output_array.ctypes.data,
                device_outputs[
                    output_name
                ],
                output_array.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                stream
            )

        cudart.cudaStreamSynchronize(
            stream
        )

        # -----------------------------
        # GPU Memory解放
        # -----------------------------

        cudart.cudaFree(
            device_input
        )

        for device_output in device_outputs.values():
            cudart.cudaFree(
                device_output
            )

        cudart.cudaStreamDestroy(
            stream
        )

        return [
            output_arrays[
                output_name
            ]
            for output_name in self.output_names
        ]