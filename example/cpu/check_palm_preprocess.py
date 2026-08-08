
import cv2

from pipetrt.palm.preprocess import preprocess

frame = cv2.imread("example/data/hand.jpg")  # またはカメラ

palm_input = preprocess(frame)

print("shape :", palm_input.shape)
print("dtype :", palm_input.dtype)
print("min   :", palm_input.min())
print("max   :", palm_input.max())