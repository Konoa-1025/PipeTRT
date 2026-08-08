import cv2

from pipetrt.palm.preprocess import preprocess
from pipetrt.palm.detector import detect


frame = cv2.imread("example/images/hand.jpg")

if frame is None:
    raise FileNotFoundError("画像を読み込めませんでした")

palm_input = preprocess(frame)

outputs = detect(palm_input)

print("=" * 50)
print("Palm Detection Result")
print("=" * 50)

print(f"Output Count : {len(outputs)}")

for index, output in enumerate(outputs):
    print()
    print(f"OUTPUT {index}")
    print(f"Shape : {output.shape}")
    print(f"Dtype : {output.dtype}")
    print(f"Min   : {output.min()}")
    print(f"Max   : {output.max()}")

    print("First 10 Values")
    print(output.flatten()[:10])