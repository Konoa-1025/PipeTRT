from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipetrt.onnx_inference import ONNXInference

IMAGE_PATH = PROJECT_ROOT / "samples" / "hand.jpg"


def main():

    model = ONNXInference()

    outputs = model.infer(IMAGE_PATH)

    # ===== 追加ここから =====
    landmarks = outputs[0].reshape(21, 3)

    print("=== LANDMARKS ===")

    for index, landmark in enumerate(landmarks):
        x, y, z = landmark

        print(
            f"{index:02d}: "
            f"x={x:8.3f}, "
            f"y={y:8.3f}, "
            f"z={z:8.3f}"
        )

    print()
    # ===== 追加ここまで =====

    for index, output in enumerate(outputs):

        print(f"Output {index}")

        print(f"shape : {output.shape}")

        print(f"dtype : {output.dtype}")

        print(output)

        print()


if __name__ == "__main__":
    main()