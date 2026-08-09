# scripts/inspect_onnx.py
# ONNXモデルの入力・出力情報を確認する

from pathlib import Path

import onnx
from onnx import TensorProto


MODEL_PATH = Path("models/hand_landmark.onnx")


def get_dimension_value(dimension) -> int | str:
    """ONNXの次元情報を、数値または文字列として取得する。"""

    if dimension.HasField("dim_value"):
        return dimension.dim_value

    if dimension.HasField("dim_param"):
        return dimension.dim_param

    return "unknown"


def get_tensor_shape(value_info) -> list[int | str]:
    """入力・出力テンソルのshapeを取得する。"""

    tensor_type = value_info.type.tensor_type

    return [
        get_dimension_value(dimension)
        for dimension in tensor_type.shape.dim
    ]


def get_tensor_type(value_info) -> str:
    """ONNXのデータ型を文字列へ変換する。"""

    element_type = value_info.type.tensor_type.elem_type

    try:
        return TensorProto.DataType.Name(element_type)
    except ValueError:
        return f"UNKNOWN_TYPE({element_type})"


def print_value_info(title: str, values) -> None:
    """入力または出力の情報を表示する。"""

    print(f"\n=== {title} ===")

    if not values:
        print("なし")
        return

    for index, value in enumerate(values):
        print(f"[{index}]")
        print(f"  name  : {value.name}")
        print(f"  shape : {get_tensor_shape(value)}")
        print(f"  type  : {get_tensor_type(value)}")


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"ONNXモデルが見つかりません: {MODEL_PATH.resolve()}"
        )

    print(f"モデルを読み込み中: {MODEL_PATH}")

    model = onnx.load(MODEL_PATH)

    # ONNXモデルとして構造が正しいか確認
    onnx.checker.check_model(model)

    print("ONNXモデルの検証に成功しました")
    print(f"IR version: {model.ir_version}")

    for opset in model.opset_import:
        domain = opset.domain or "ai.onnx"
        print(f"Opset: {domain} version {opset.version}")

    print_value_info("INPUTS", model.graph.input)
    print_value_info("OUTPUTS", model.graph.output)


if __name__ == "__main__":
    main()