from pathlib import Path

import tensorrt as trt


def print_parser_errors(parser):
    for error_index in range(parser.num_errors):
        print(parser.get_error(error_index))


def print_network_information(network):
    print("\n--- Network inputs ---")

    for input_index in range(network.num_inputs):
        tensor = network.get_input(input_index)

        print(
            f"name={tensor.name}, "
            f"shape={tensor.shape}, "
            f"dtype={tensor.dtype}"
        )

    print("\n--- Network outputs ---")

    for output_index in range(network.num_outputs):
        tensor = network.get_output(output_index)

        print(
            f"name={tensor.name}, "
            f"shape={tensor.shape}, "
            f"dtype={tensor.dtype}"
        )


def main():
    onnx_path = Path("models/hand_landmark.onnx")
    engine_path = Path("engines/hand_landmark_fp32.engine")

    if not onnx_path.exists():
        raise FileNotFoundError(
            f"ONNXモデルが見つかりません: {onnx_path}"
        )

    engine_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger = trt.Logger(trt.Logger.INFO)

    network_flag = (
        1 << int(
            trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED
        )
    )

    builder = trt.Builder(logger)
    network = builder.create_network(network_flag)
    parser = trt.OnnxParser(network, logger)
    config = builder.create_builder_config()

    # Engine構築時にTensorRTが利用できる作業用GPUメモリ
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        2 << 30,
    )

    print(f"Loading ONNX model: {onnx_path}")

    model_data = onnx_path.read_bytes()

    if not parser.parse(model_data):
        print("\n--- ONNX parser errors ---")
        print_parser_errors(parser)

        raise RuntimeError(
            "Hand Landmark ONNXの解析に失敗しました"
        )

    print_network_information(network)

    print("\nBuilding FP32 TensorRT engine...")

    serialized_engine = builder.build_serialized_network(
        network,
        config,
    )

    if serialized_engine is None:
        raise RuntimeError(
            "TensorRT Engineの生成に失敗しました"
        )

    engine_path.write_bytes(serialized_engine)

    print()
    print(f"Engine created: {engine_path}")
    print(
        f"Engine size: "
        f"{engine_path.stat().st_size / 1024 / 1024:.2f} MB"
    )
    print("Hand Landmark FP32 Engine build: OK")


if __name__ == "__main__":
    main()