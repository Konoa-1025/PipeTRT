import cv2

from pipetrt.palm.preprocess import preprocess
from pipetrt.palm.detector import detect
from pipetrt.palm.decoder import decode
from pipetrt.palm.anchors import generate_anchors


frame = cv2.imread(
    "example/data/hand.jpg"
)

if frame is None:
    raise FileNotFoundError(
        "画像を読み込めませんでした"
    )

palm_input = preprocess(frame)

boxes, scores = detect(palm_input)

anchors = generate_anchors()

results = decode(
    boxes,
    scores,
    anchors
)

print(
    "Detection Count:",
    len(results)
)

for index, result in enumerate(results):

    print()
    print(f"=== Palm {index} ===")

    print(
        "Score:",
        result["score"]
    )

    print(
        "BBox:",
        result["bbox"]
    )

    print("Keypoints:")
    print(
        result["keypoints"]
    )