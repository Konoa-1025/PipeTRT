import cv2

from pipetrt.palm.preprocess import preprocess
from pipetrt.engine.runtime import TensorRTRuntime


frame = cv2.imread(
    "example/data/hand.jpg"
)

if frame is None:
    raise FileNotFoundError(
        "画像を読み込めませんでした"
    )

palm_input = preprocess(
    frame
)

runtime = TensorRTRuntime(
    "models/palm_detection.engine"
)

outputs = runtime.infer(
    palm_input
)

print(
    "Output Count:",
    len(outputs)
)

for index, output in enumerate(outputs):

    print()
    print(
        f"OUTPUT {index}"
    )

    print(
        "Shape:",
        output.shape
    )

    print(
        "Dtype:",
        output.dtype
    )

    print(
        "Min:",
        output.min()
    )

    print(
        "Max:",
        output.max()
    )

    print(
        "Sample:",
        output.flatten()[:10]
    )