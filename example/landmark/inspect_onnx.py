# scripts/inspect_onnx.py

from pathlib import Path

import onnx


def format_shape(value_info):
    dimensions = []

    for dimension in value_info.type.tensor_type.shape.dim:
        if dimension.dim_value:
            dimensions.append(dimension.dim_value)
        elif dimension.dim_param:
            dimensions.append(dimension.dim_param)
        else:
            dimensions.append("?")

    return dimensions


def main():
    model_path = Path("models/hand_landmark.onnx")

    if not model_path.exists():
        raise FileNotFoundError(
            f"ONNXモデルが見つかりません: {model_path}"
        )

    model = onnx.load(model_path)
    onnx.checker.check_model(model)

    initializer_names = {
        initializer.name
        for initializer in model.graph.initializer
    }

    print(f"Model: {model_path}")
    print(f"IR version: {model.ir_version}")
    print(
        "Opset:",
        [
            f"{opset.domain or 'ai.onnx'}:{opset.version}"
            for opset in model.opset_import
        ],
    )

    print("\n--- Inputs ---")

    for model_input in model.graph.input:
        if model_input.name in initializer_names:
            continue

        print(
            f"name={model_input.name}, "
            f"shape={format_shape(model_input)}, "
            f"element_type="
            f"{model_input.type.tensor_type.elem_type}"
        )

    print("\n--- Outputs ---")

    for model_output in model.graph.output:
        print(
            f"name={model_output.name}, "
            f"shape={format_shape(model_output)}, "
            f"element_type="
            f"{model_output.type.tensor_type.elem_type}"
        )

    print(f"\nNodes: {len(model.graph.node)}")
    print("ONNX inspection: OK")


if __name__ == "__main__":
    main()