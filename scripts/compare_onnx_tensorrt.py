from pathlib import Path

import numpy as np
import onnxruntime as ort
import tensorrt as trt

try:
    from cuda.bindings import runtime as cudart
except ImportError:
    from cuda import cudart


def check_cuda(result):
    error_code = result[0]

    if error_code != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(
            f"CUDA APIでエラーが発生しました: {error_code}"
        )

    if len(result) == 1:
        return None

    if len(result) == 2:
        return result[1]

    return result[1:]


def run_onnx(
    model_path: Path,
    input_data: np.ndarray,
) -> dict[str, np.ndarray]:
    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name
    output_names = [
        output.name
        for output in session.get_outputs()
    ]

    outputs = session.run(
        output_names,
        {
            input_name: input_data,
        },
    )

    return {
        output_name: output
        for output_name, output in zip(
            output_names,
            outputs,
        )
    }


def run_tensorrt(
    engine_path: Path,
    input_data: np.ndarray,
) -> dict[str, np.ndarray]:
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)

    engine = runtime.deserialize_cuda_engine(
        engine_path.read_bytes()
    )

    if engine is None:
        raise RuntimeError(
            "TensorRT Engineの読み込みに失敗しました"
        )

    context = engine.create_execution_context()

    if context is None:
        raise RuntimeError(
            "ExecutionContextの作成に失敗しました"
        )

    stream = None
    device_buffers = {}
    host_outputs = {}

    try:
        stream = check_cuda(
            cudart.cudaStreamCreate()
        )

        for tensor_index in range(engine.num_io_tensors):
            tensor_name = engine.get_tensor_name(
                tensor_index
            )

            tensor_mode = engine.get_tensor_mode(
                tensor_name
            )

            tensor_shape = tuple(
                context.get_tensor_shape(tensor_name)
            )

            tensor_dtype = trt.nptype(
                engine.get_tensor_dtype(tensor_name)
            )

            if tensor_mode == trt.TensorIOMode.INPUT:
                host_array = np.ascontiguousarray(
                    input_data.astype(
                        tensor_dtype,
                        copy=False,
                    )
                )
            else:
                host_array = np.empty(
                    tensor_shape,
                    dtype=tensor_dtype,
                )

                host_outputs[tensor_name] = host_array

            device_pointer = check_cuda(
                cudart.cudaMalloc(
                    host_array.nbytes
                )
            )

            device_buffers[tensor_name] = device_pointer

            address_set = context.set_tensor_address(
                tensor_name,
                int(device_pointer),
            )

            if not address_set:
                raise RuntimeError(
                    f"テンソルアドレスの設定に失敗: "
                    f"{tensor_name}"
                )

            if tensor_mode == trt.TensorIOMode.INPUT:
                check_cuda(
                    cudart.cudaMemcpyAsync(
                        device_pointer,
                        host_array.ctypes.data,
                        host_array.nbytes,
                        cudart.cudaMemcpyKind
                        .cudaMemcpyHostToDevice,
                        stream,
                    )
                )

        execution_succeeded = context.execute_async_v3(
            stream_handle=int(stream)
        )

        if not execution_succeeded:
            raise RuntimeError(
                "TensorRT推論に失敗しました"
            )

        for tensor_name, host_output in host_outputs.items():
            device_pointer = device_buffers[
                tensor_name
            ]

            check_cuda(
                cudart.cudaMemcpyAsync(
                    host_output.ctypes.data,
                    device_pointer,
                    host_output.nbytes,
                    cudart.cudaMemcpyKind
                    .cudaMemcpyDeviceToHost,
                    stream,
                )
            )

        check_cuda(
            cudart.cudaStreamSynchronize(stream)
        )

        return {
            tensor_name: output.copy()
            for tensor_name, output
            in host_outputs.items()
        }

    finally:
        for device_pointer in device_buffers.values():
            check_cuda(
                cudart.cudaFree(device_pointer)
            )

        if stream is not None:
            check_cuda(
                cudart.cudaStreamDestroy(stream)
            )


def compare_outputs(
    onnx_outputs: dict[str, np.ndarray],
    tensorrt_outputs: dict[str, np.ndarray],
):
    print("\n--- Output comparison ---")

    all_outputs_match = True

    for output_name, onnx_output in onnx_outputs.items():
        if output_name not in tensorrt_outputs:
            print(
                f"{output_name}: "
                "TensorRT側に存在しません"
            )

            all_outputs_match = False
            continue

        tensorrt_output = tensorrt_outputs[
            output_name
        ]

        absolute_difference = np.abs(
            onnx_output - tensorrt_output
        )

        maximum_error = float(
            np.max(absolute_difference)
        )

        mean_error = float(
            np.mean(absolute_difference)
        )

        outputs_match = np.allclose(
            onnx_output,
            tensorrt_output,
            rtol=1e-4,
            atol=1e-5,
        )

        print(f"\nOutput: {output_name}")
        print(f"Shape: {onnx_output.shape}")
        print(f"ONNX sample:     {onnx_output.flatten()[:8]}")
        print(
            f"TensorRT sample: "
            f"{tensorrt_output.flatten()[:8]}"
        )
        print(f"Maximum error: {maximum_error:.10f}")
        print(f"Mean error:    {mean_error:.10f}")
        print(
            f"Result: "
            f"{'OK' if outputs_match else 'NG'}"
        )

        if not outputs_match:
            all_outputs_match = False

    print()

    if all_outputs_match:
        print(
            "ONNX RuntimeとTensorRTの出力は"
            "許容誤差内で一致しました"
        )
    else:
        raise RuntimeError(
            "ONNX RuntimeとTensorRTの出力に"
            "大きな差があります"
        )


def main():
    model_path = Path(
        "models/hand_landmark.onnx"
    )

    engine_path = Path(
        "engines/hand_landmark_fp32.engine"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"ONNXモデルが見つかりません: "
            f"{model_path}"
        )

    if not engine_path.exists():
        raise FileNotFoundError(
            f"Engineが見つかりません: "
            f"{engine_path}"
        )

    # 毎回同じ疑似入力を生成する
    random_generator = np.random.default_rng(
        seed=1025
    )

    input_data = random_generator.random(
        (1, 3, 224, 224),
        dtype=np.float32,
    )

    print("--- Input ---")
    print(f"Shape: {input_data.shape}")
    print(f"Data type: {input_data.dtype}")
    print(f"Minimum: {input_data.min():.6f}")
    print(f"Maximum: {input_data.max():.6f}")

    print("\nRunning ONNX Runtime...")
    onnx_outputs = run_onnx(
        model_path,
        input_data,
    )

    print("Running TensorRT...")
    tensorrt_outputs = run_tensorrt(
        engine_path,
        input_data,
    )

    compare_outputs(
        onnx_outputs,
        tensorrt_outputs,
    )


if __name__ == "__main__":
    main()