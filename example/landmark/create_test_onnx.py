# scripts/create_test_onnx.py

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def main():
    output_directory = Path("models")
    output_directory.mkdir(exist_ok=True)

    output_path = output_directory / "test_model.onnx"

    input_tensor = helper.make_tensor_value_info(
        "input",
        TensorProto.FLOAT,
        [1, 4],
    )

    output_tensor = helper.make_tensor_value_info(
        "output",
        TensorProto.FLOAT,
        [1, 4],
    )

    multiplier = numpy_helper.from_array(
        np.array([2.0], dtype=np.float32),
        name="multiplier",
    )

    bias = numpy_helper.from_array(
        np.array([1.0], dtype=np.float32),
        name="bias",
    )

    multiply_node = helper.make_node(
        "Mul",
        inputs=["input", "multiplier"],
        outputs=["multiplied"],
    )

    add_node = helper.make_node(
        "Add",
        inputs=["multiplied", "bias"],
        outputs=["output"],
    )

    graph = helper.make_graph(
        nodes=[multiply_node, add_node],
        name="PipeTRTTestModel",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[multiplier, bias],
    )

    model = helper.make_model(
        graph,
        producer_name="PipeTRT",
        opset_imports=[helper.make_opsetid("", 17)],
    )

    # TensorRTとの互換性を考慮して固定
    model.ir_version = 10

    onnx.checker.check_model(model)
    onnx.save(model, output_path)

    print(f"ONNX model created: {output_path}")
    print("Calculation: output = input * 2 + 1")


if __name__ == "__main__":
    main()