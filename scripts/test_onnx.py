from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipetrt.onnx_inference import ONNXInference


IMAGE_PATH = PROJECT_ROOT / "samples" / "hand.jpg"


def main():

    model = ONNXInference()

    outputs = model.infer(IMAGE_PATH)

    print()

    for index, output in enumerate(outputs):

        print(f"Output {index}")

        print(f"shape : {output.shape}")

        print(f"dtype : {output.dtype}")

        print(output)

        print()


if __name__ == "__main__":
    main()