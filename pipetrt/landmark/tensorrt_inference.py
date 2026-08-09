from pathlib import Path

from pipetrt.engines.runtime import TensorRTRuntime
from pipetrt.landmark.preprocess import preprocess_frame


ENGINE_PATH = Path(
    "engines/hand_landmark_fp32.engine"
)


class TensorRTInference:
    def __init__(
        self,
        engine_path=ENGINE_PATH
    ):
        self.runtime = TensorRTRuntime(
            engine_path
        )

    def infer_frame(
        self,
        frame
    ):
        input_tensor = preprocess_frame(
            frame
        )

        return self.runtime.infer(
            input_tensor
        )

    def close(self):
        self.runtime.close()