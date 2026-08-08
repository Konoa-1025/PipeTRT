import cv2
import numpy as np

from pipetrt.palm.preprocess import preprocess
from pipetrt.palm.detector import detect


def load_anchors():
    anchors = []

    # 192x192 Palm Detection用
    # stride 8 : 24x24 × 2 anchors
    for y in range(24):
        for x in range(24):
            center_x = (x + 0.5) / 24
            center_y = (y + 0.5) / 24

            anchors.append([center_x, center_y])
            anchors.append([center_x, center_y])

    # stride 16 : 12x12 × 6 anchors
    for y in range(12):
        for x in range(12):
            center_x = (x + 0.5) / 12
            center_y = (y + 0.5) / 12

            for _ in range(6):
                anchors.append([center_x, center_y])

    return np.array(anchors, dtype=np.float32)


frame = cv2.imread("example/data/hand2.jpg")

if frame is None:
    raise FileNotFoundError("画像を読み込めませんでした")

palm_input = preprocess(frame)

boxes, scores = detect(palm_input)

best_index = scores.argmax()

raw_score = scores.reshape(-1)[best_index]
probability = 1.0 / (1.0 + np.exp(-raw_score))

anchors = load_anchors()

print("Anchor Count :", len(anchors))
print("Best Anchor  :", best_index)
print("Probability  :", probability)

anchor_x = anchors[best_index][0]
anchor_y = anchors[best_index][1]

height, width = frame.shape[:2]

pixel_x = int(anchor_x * width)
pixel_y = int(anchor_y * height)

cv2.circle(
    frame,
    (pixel_x, pixel_y),
    10,
    (0, 0, 255),
    -1
)

cv2.putText(
    frame,
    f"Anchor {best_index}",
    (pixel_x + 10, pixel_y - 10),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (0, 0, 255),
    2
)

cv2.imshow("Best Palm Anchor", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()