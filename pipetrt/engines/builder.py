# pipetrt/engines/builder.py

import tensorrt as trt


def build_engine(onnx_path, engine_path):
    logger = trt.Logger(trt.Logger.WARNING)

    builder = trt.Builder(logger)

    network = builder.create_network()

    parser = trt.OnnxParser(
        network,
        logger
    )

    with open(onnx_path, "rb") as model_file:
        model_data = model_file.read()

    if not parser.parse(model_data):
        print("ONNXモデルの読み込みに失敗しました")

        for index in range(parser.num_errors):
            print(parser.get_error(index))

        return False

    config = builder.create_builder_config()

    serialized_engine = builder.build_serialized_network(
        network,
        config
    )

    if serialized_engine is None:
        print("TensorRT Engineの生成に失敗しました")
        return False

    with open(engine_path, "wb") as engine_file:
        engine_file.write(serialized_engine)

    print(
        f"TensorRT Engineを保存しました: {engine_path}"
    )

    return True