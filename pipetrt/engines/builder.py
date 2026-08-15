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

    if precision not in ("fp16", "fp32"):
        raise ValueError(
            f"Unsupported precision: {precision}"
        )

    trt_major_version = int(
        trt.__version__.split(".")[0]
    )

    build_onnx_path = onnx_path
    temporary_onnx_path = None

    # =====================================
    # FP16
    # =====================================

    # TensorRT 11以降では
    # BuilderFlag.FP16が使用できないため、
    # ModelOptでONNXをMixed Precision化する
    if (
        precision == "fp16"
        and trt_major_version >= 11
    ):
        try:
            from modelopt.onnx.autocast import (
                convert_to_mixed_precision
            )

        except ImportError as error:
            raise RuntimeError(
                "FP16 Engine generation with "
                "TensorRT 11 or later requires "
                "NVIDIA Model Optimizer. "
                'Install it with: '
                'pip install "nvidia-modelopt[onnx]"'
            ) from error

        converted_model = (
            convert_to_mixed_precision(
                onnx_path=str(onnx_path),
                low_precision_type="fp16",
                keep_io_types=True
            )
        )

        temporary_file = (
            tempfile.NamedTemporaryFile(
                suffix="_fp16.onnx",
                delete=False
            )
        )

        temporary_file.close()

        temporary_onnx_path = Path(temporary_file.name)

        onnx.save(converted_model,temporary_onnx_path)

        build_onnx_path = (temporary_onnx_path)

    # =====================================
    # TensorRT Build
    # =====================================

    logger = trt.Logger(
        trt.Logger.WARNING
    )

    builder = trt.Builder(
        logger
    )

    # =====================================
    # Network
    # =====================================

    # TensorRT 8系ではONNX Parser利用時に
    # EXPLICIT_BATCHを明示する必要がある
    if trt_major_version < 10:
        network_flags = (
            1
            << int(
                trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH
            )
        )

        network = builder.create_network(
            network_flags
        )

    else:
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
                parser.get_error(index)
            )

        if temporary_onnx_path is not None:
            temporary_onnx_path.unlink(
                missing_ok=True
            )

        return False

    config = (
        builder.create_builder_config()
    )

    # =====================================
    # TensorRT 10以前 FP16
    # =====================================

    # TensorRT 8.xなどでは
    # BuilderFlag.FP16を利用する
    if (
        precision == "fp16"
        and trt_major_version < 11
    ):
        config.set_flag(
            trt.BuilderFlag.FP16
        )

    # =====================================
    # Engine Build
    # =====================================

    serialized_engine = (
        builder.build_serialized_network(
            network,
            config
        )
    )

    if serialized_engine is None:
        if temporary_onnx_path is not None:
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

    # =====================================
    # Temporary ONNX Cleanup
    # =====================================

    if temporary_onnx_path is not None:
        temporary_onnx_path.unlink(
            missing_ok=True
        )

    return True