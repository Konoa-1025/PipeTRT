import cv2
import numpy as np

from pipetrt.palm.preprocess import preprocess
from pipetrt.palm.detector import detect


frame = cv2.imread("example/data/hand.jpg")

if frame is None:
    raise FileNotFoundError("画像を読み込めませんでした")

palm_input = preprocess(frame)

boxes, scores = detect(palm_input)

best_index = scores.argmax()

raw_score = scores.reshape(-1)[best_index]

probability = 1.0 / (1.0 + np.exp(-raw_score))

print("Best Anchor :", best_index)
print("Raw Score   :", raw_score)
print("Probability :", probability)

print()
print("Box Data")
print(boxes.reshape(-1, 18)[best_index])