import onnxruntime as ort

session = ort.InferenceSession(
    "models/palm_detection.onnx",
    providers=["CPUExecutionProvider"]
)

print("INPUT")
for i in session.get_inputs():
    print(i.name)
    print(i.shape)
    print(i.type)

print("OUTPUT")
for o in session.get_outputs():
    print(o.name)
    print(o.shape)
    print(o.type)