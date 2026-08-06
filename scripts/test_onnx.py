from pathlib import Path
import sys

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipetrt.onnx_inference import ONNXInference


IMAGE_PATH = PROJECT_ROOT / "samples" / "hand.jpg"


def main():

    model = ONNXInference()

    outputs = model.infer(IMAGE_PATH)

    image = cv2.imread(str(IMAGE_PATH))
    image = cv2.resize(image, (224, 224))

    landmarks = outputs[0].reshape(21, 3)

    for landmark in landmarks:
        x = int(landmark[0])
        y = int(landmark[1])

        cv2.circle(
            image,
            (x, y),
            5,
            (0, 255, 0),
            -1
        )

    cv2.imshow("PipeTRT Landmark Preview", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print()

    for index, output in enumerate(outputs):

        print(f"Output {index}")
        print(f"shape : {output.shape}")
        print(f"dtype : {output.dtype}")
        print(output)
        print()


if __name__ == "__main__":
    main()