# scripts/run_test_engine.py

from pathlib import Path

import numpy as np
import tensorrt as trt

try:
    # 新しいcuda-python
    from cuda.bindings import runtime as cudart
except ImportError:
    # 旧形式との互換用
    from cuda import cudart


def check_cuda(result):
    """
    cuda-pythonの戻り値を確認する。

    戻り値の先頭がCUDAエラーコード、
    2個目以降が実際の値。
    """
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


def print_engine_information(engine):
    print("--- Engine information ---")

    for tensor_index in range(engine.num_io_tensors):
        tensor_name = engine.get_tensor_name(tensor_index)
        tensor_mode = engine.get_tensor_mode(tensor_name)
        tensor_shape = engine.get_tensor_shape(tensor_name)
        tensor_dtype = engine.get_tensor_dtype(tensor_name)

        print(
            f"name={tensor_name}, "
            f"mode={tensor_mode}, "
            f"shape={tensor_shape}, "
            f"dtype={tensor_dtype}"
        )


def main():
    engine_path = Path("engines/test_model.engine")

    if not engine_path.exists():
        raise FileNotFoundError(
            f"Engineが見つかりません: {engine_path}"
        )

    logger = trt.Logger(trt.Logger.WARNING)

    # EngineファイルをTensorRT上へ復元
    runtime = trt.Runtime(logger)
    engine_data = engine_path.read_bytes()
    engine = runtime.deserialize_cuda_engine(engine_data)

    if engine is None:
        raise RuntimeError("Engineの読み込みに失敗しました")

    context = engine.create_execution_context()

    if context is None:
        raise RuntimeError(
            "ExecutionContextの作成に失敗しました"
        )

    print_engine_information(engine)

    input_name = "input"
    output_name = "output"

    # 今回のテストモデルへ渡す入力
    input_data = np.array(
        [[1.0, 2.0, 3.0, 4.0]],
        dtype=np.float32,
    )

    # GPUから受け取る出力領域
    output_data = np.empty(
        (1, 4),
        dtype=np.float32,
    )

    input_device = None
    output_device = None
    stream = None

    try:
        # GPUメモリ確保
        input_device = check_cuda(
            cudart.cudaMalloc(input_data.nbytes)
        )

        output_device = check_cuda(
            cudart.cudaMalloc(output_data.nbytes)
        )

        # CUDA Streamを作成
        stream = check_cuda(
            cudart.cudaStreamCreate()
        )

        # CPU側の入力をGPUへ転送
        check_cuda(
            cudart.cudaMemcpyAsync(
                input_device,
                input_data.ctypes.data,
                input_data.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                stream,
            )
        )

        # TensorRTへ入出力のGPUアドレスを登録
        input_address_set = context.set_tensor_address(
            input_name,
            int(input_device),
        )

        output_address_set = context.set_tensor_address(
            output_name,
            int(output_device),
        )

        if not input_address_set:
            raise RuntimeError(
                "入力テンソルのアドレス設定に失敗しました"
            )

        if not output_address_set:
            raise RuntimeError(
                "出力テンソルのアドレス設定に失敗しました"
            )

        # TensorRT推論を実行
        execution_succeeded = context.execute_async_v3(
            stream_handle=int(stream)
        )

        if not execution_succeeded:
            raise RuntimeError(
                "TensorRT推論の実行に失敗しました"
            )

        # GPU上の出力をCPUへ戻す
        check_cuda(
            cudart.cudaMemcpyAsync(
                output_data.ctypes.data,
                output_device,
                output_data.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                stream,
            )
        )

        # GPU処理が全部終わるまで待つ
        check_cuda(
            cudart.cudaStreamSynchronize(stream)
        )

        expected_output = input_data * 2.0 + 1.0

        print()
        print("--- Inference result ---")
        print(f"Input:    {input_data}")
        print(f"Output:   {output_data}")
        print(f"Expected: {expected_output}")

        if np.allclose(output_data, expected_output):
            print("TensorRT GPU inference: OK")
        else:
            raise RuntimeError(
                "推論結果が期待値と一致しません"
            )

    finally:
        # 確保したGPU資源を必ず解放
        if stream is not None:
            check_cuda(
                cudart.cudaStreamDestroy(stream)
            )

        if input_device is not None:
            check_cuda(
                cudart.cudaFree(input_device)
            )

        if output_device is not None:
            check_cuda(
                cudart.cudaFree(output_device)
            )


if __name__ == "__main__":
    main()