import cv2

from pipetrt.palm.tensorrt_inference import PalmTensorRTInference


frame = cv2.imread(
    "example/data/hand.jpg"
)

if frame is None:
    raise FileNotFoundError(
        "画像を読み込めませんでした"
    )


palm_model = PalmTensorRTInference()

outputs = palm_model.infer_frame(
    frame
)

for name, output in outputs.items():
    print(
        name,
        output.shape
    )

palm_model.close()