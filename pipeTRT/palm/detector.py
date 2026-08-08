import onnxruntime as ort


session = ort.InferenceSession(
    "models/palm_detection.onnx",
    providers=["CPUExecutionProvider"]
)


def detect(palm_input):
    input_name = session.get_inputs()[0].name

    outputs = session.run(
        None,
        {
            input_name: palm_input
        }
    )

    boxes = outputs[0]
    scores = outputs[1]

    return boxes, scores