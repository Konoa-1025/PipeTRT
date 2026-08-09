from pipetrt.engine.builder import build_engine


def main():
    onnx_path = "models/hand_landmark.onnx"
    engine_path = "engines/hand_landmark_fp32.engine"

    success = build_engine(
        onnx_path,
        engine_path
    )

    if success:
        print("Hand Landmark Engine生成完了")
    else:
        print("Hand Landmark Engine生成失敗")


if __name__ == "__main__":
    main()