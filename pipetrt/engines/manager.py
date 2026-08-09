from pathlib import Path

from pipetrt.engines.builder import build_engine


class EngineManager:
    def __init__(
        self,
        model="full",
        precision="fp16"
    ):
        if model not in (
            "lite",
            "full"
        ):
            raise ValueError(
                f"Unsupported model: {model}"
            )

        if precision not in (
            "fp16",
            "fp32"
        ):
            raise ValueError(
                f"Unsupported precision: "
                f"{precision}"
            )

        self.model = model
        self.precision = precision

        self.engine_dir = Path(
            "engines"
        )

        self.engine_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    def get_palm_engine_path(self):
        return (
            self.engine_dir
            / (
                f"palm_"
                f"{self.model}_"
                f"{self.precision}.engine"
            )
        )

    def get_landmark_engine_path(self):
        return (
            self.engine_dir
            / (
                f"landmark_"
                f"{self.model}_"
                f"{self.precision}.engine"
            )
        )

    def get_palm_onnx_path(self):
        return Path(
            "models/palm_detection.onnx"
        )

    def get_landmark_onnx_path(self):
        return Path(
            "models/hand_landmark.onnx"
        )

    def ensure_engines(self):
        palm_engine = (
            self.get_palm_engine_path()
        )

        landmark_engine = (
            self.get_landmark_engine_path()
        )

        if not palm_engine.exists():
            self.build_palm_engine(
                palm_engine
            )

        if not landmark_engine.exists():
            self.build_landmark_engine(
                landmark_engine
            )

        return {
            "palm": palm_engine,
            "landmark": landmark_engine
        }

    def build_palm_engine(
        self,
        engine_path
    ):
        onnx_path = (
            self.get_palm_onnx_path()
        )

        if not onnx_path.exists():
            raise FileNotFoundError(
                f"Palm ONNX model not found: "
                f"{onnx_path}"
            )

        success = build_engine(
            onnx_path,
            engine_path,
            precision=self.precision
        )

        if not success:
            raise RuntimeError(
                f"Failed to build Palm Engine: "
                f"{engine_path}"
            )

    def build_landmark_engine(
        self,
        engine_path
    ):
        onnx_path = (
            self.get_landmark_onnx_path()
        )

        if not onnx_path.exists():
            raise FileNotFoundError(
                f"Landmark ONNX model not found: "
                f"{onnx_path}"
            )

        success = build_engine(
            onnx_path,
            engine_path,
            precision=self.precision
        )

        if not success:
            raise RuntimeError(
                f"Failed to build Landmark Engine: "
                f"{engine_path}"
            )