import tempfile
from pathlib import Path

import onnx
import tensorrt as trt


def build_engine(
    onnx_path,
    engine_path,
    precision="fp32"
):
    onnx_path = Path(onnx_path)
    engine_path = Path(engine_path)

    if precision not in (
        "fp16",
        "fp32"
    ):
        raise ValueError(
            f"Unsupported precision: {precision}"
        )

    build_onnx_path = onnx_path
    temporary_onnx_path = None

    # =====================================
    # FP16
    # =====================================

    if precision == "fp16":
        try:
            from modelopt.onnx.autocast import (
                convert_to_mixed_precision
            )
        except ImportError as error:
            raise RuntimeError(
                "FP16 Engine generation requires "
                "NVIDIA Model Optimizer. "
                'Install it with: '
                'pip install "nvidia-modelopt[onnx]"'
            ) from error

        converted_model = (
            convert_to_mixed_precision(
                onnx_path=str(
                    onnx_path
                ),
                low_precision_type="fp16",
                keep_io_types=True
            )
        )

        temporary_file = tempfile.NamedTemporaryFile(
            suffix="_fp16.onnx",
            delete=False
        )

        temporary_file.close()

        temporary_onnx_path = Path(
            temporary_file.name
        )

        onnx.save(
            converted_model,
            temporary_onnx_path
        )

        build_onnx_path = (
            temporary_onnx_path
        )

    # =====================================
    # TensorRT Build
    # =====================================

    logger = trt.Logger(
        trt.Logger.WARNING
    )

    builder = trt.Builder(
        logger
    )

    network = builder.create_network()

    parser = trt.OnnxParser(
        network,
        logger
    )

    with open(
        build_onnx_path,
        "rb"
    ) as file:
        model_data = file.read()

    if not parser.parse(
        model_data
    ):
        for index in range(
            parser.num_errors
        ):
            print(
                parser.get_error(
                    index
                )
            )

        if (
            temporary_onnx_path
            is not None
        ):
            temporary_onnx_path.unlink(
                missing_ok=True
            )

        return False

    config = (
        builder.create_builder_config()
    )

    serialized_engine = (
        builder.build_serialized_network(
            network,
            config
        )
    )

    if serialized_engine is None:
        if (
            temporary_onnx_path
            is not None
        ):
            temporary_onnx_path.unlink(
                missing_ok=True
            )

        return False

    engine_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        engine_path,
        "wb"
    ) as file:
        file.write(
            serialized_engine
        )

    # 一時FP16 ONNX削除
    if (
        temporary_onnx_path
        is not None
    ):
        temporary_onnx_path.unlink(
            missing_ok=True
        )

    return True