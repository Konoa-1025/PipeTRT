# scripts/build_test_engine.py

from pathlib import Path

import tensorrt as trt


def print_parser_errors(parser):
    for error_index in range(parser.num_errors):
        print(parser.get_error(error_index))


def main():
    onnx_path = Path("models/test_model.onnx")
    engine_path = Path("engines/test_model.engine")

    if not onnx_path.exists():
        raise FileNotFoundError(
            f"ONNXモデルが見つかりません: {onnx_path}"
        )

    engine_path.parent.mkdir(exist_ok=True)

    logger = trt.Logger(trt.Logger.INFO)

    strongly_typed_flag = (
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )

    builder = trt.Builder(logger)
    network = builder.create_network(strongly_typed_flag)
    parser = trt.OnnxParser(network, logger)
    config = builder.create_builder_config()

    # TensorRTが最適化時に利用できるGPUメモリ上限
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        1 << 30,
    )

    print(f"Loading ONNX model: {onnx_path}")

    onnx_data = onnx_path.read_bytes()

    if not parser.parse(onnx_data):
        print("ONNX parser error:")
        print_parser_errors(parser)
        raise RuntimeError("ONNXモデルの解析に失敗しました")

    print(f"Network inputs: {network.num_inputs}")
    print(f"Network outputs: {network.num_outputs}")
    print("Building TensorRT engine...")

    serialized_engine = builder.build_serialized_network(
        network,
        config,
    )

    if serialized_engine is None:
        raise RuntimeError(
            "TensorRTエンジンの構築に失敗しました"
        )

    engine_path.write_bytes(serialized_engine)

    print(f"TensorRT engine created: {engine_path}")
    print(f"Engine size: {engine_path.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()