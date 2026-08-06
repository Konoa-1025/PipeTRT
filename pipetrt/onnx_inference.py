# pipetrt/onnx_inference.py

from pathlib import Path

import onnxruntime as ort

from pipetrt.preprocess import preprocess_frame
from pipetrt.preprocess import preprocess_image


MODEL_PATH = Path("models/hand_landmark.onnx")


class ONNXInference:

    def __init__(self):

        self.session = ort.InferenceSession(
            str(MODEL_PATH),
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name

        self.output_names = [
            output.name
            for output in self.session.get_outputs()
        ]

    def infer(self, image_path):
        input_tensor = preprocess_image(image_path)
        return self.run(input_tensor)

    def infer_frame(self, frame):
        input_tensor = preprocess_frame(frame)
        return self.run(input_tensor)

    def run(self, input_tensor):
        outputs = self.session.run(
            self.output_names,
            {
                self.input_name: input_tensor,
            },
        )

        return outputs