from pipetrt.engine.builder import build_engine


build_engine(
    "models/palm_detection.onnx",
    "models/palm_detection.engine"
)