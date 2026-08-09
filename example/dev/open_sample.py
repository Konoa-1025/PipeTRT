import pipetrt


landmarker = pipetrt.HandLandmarker()

print(
    "Model:",
    landmarker.model
)

print(
    "Precision:",
    landmarker.precision
)

landmarker.close()