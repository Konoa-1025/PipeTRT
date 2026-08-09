from pathlib import Path

from pipetrt.engines.runtime import TensorRTRuntime
from pipetrt.palm.preprocess import preprocess


ENGINE_PATH = Path(
    "engines/palm_detection.engine"
)


class PalmTensorRTInference:
    def __init__(
        self,
        engine_path=ENGINE_PATH
    ):
        self.runtime = TensorRTRuntime(
            engine_path
        )

    def infer_frame(self, frame):
        input_tensor = preprocess(
            frame
        )

        return self.runtime.infer(
            input_tensor
        )

    def close(self):
        self.runtime.close()