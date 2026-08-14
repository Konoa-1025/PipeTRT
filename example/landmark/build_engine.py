from pathlib import Path

from pipetrt.engines.builder import build_engine


def main():
    onnx_path = Path(
        "models/hand_landmark.onnx"
    )

    engine_path = Path(
        "engines/hand_landmark_fp32.engine"
    )

    # enginesフォルダが無ければ作る
    engine_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        f"ONNX:   {onnx_path}"
    )

    print(
        f"Engine: {engine_path}"
    )

    success = build_engine(
        onnx_path,
        engine_path
    )

    if success:
        print(
            "Hand Landmark Engine生成完了"
        )
    else:
        print(
            "Hand Landmark Engine生成失敗"
        )


if __name__ == "__main__":
    main()