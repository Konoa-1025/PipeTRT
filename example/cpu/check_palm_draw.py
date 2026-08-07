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

boxes, scores = detect(
    palm_input
)

anchors = generate_anchors()

results = decode(
    boxes,
    scores,
    anchors
)


# Palmモデルが実際に見た192x192画像を表示用に戻す
display_image = palm_input[0]

display_image = (
    display_image * 255.0
).clip(
    0,
    255
).astype(
    "uint8"
)

display_image = cv2.cvtColor(
    display_image,
    cv2.COLOR_RGB2BGR
)


for result in results:

    score = result["score"]
    bbox = result["bbox"]
    keypoints = result["keypoints"]

    height, width = display_image.shape[:2]

    # ------------------------------
    # Bounding Box
    # ------------------------------

    x1 = int(bbox[0] * width)
    y1 = int(bbox[1] * height)

    x2 = int(bbox[2] * width)
    y2 = int(bbox[3] * height)

    cv2.rectangle(
        display_image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    # ------------------------------
    # Score
    # ------------------------------

    cv2.putText(
        display_image,
        f"Palm {score:.3f}",
        (x1, max(y1 - 5, 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (0, 255, 0),
        1
    )

    # ------------------------------
    # 7 Keypoints
    # ------------------------------

    for index, keypoint in enumerate(keypoints):

        x = int(
            keypoint[0] * width
        )

        y = int(
            keypoint[1] * height
        )

        cv2.circle(
            display_image,
            (x, y),
            3,
            (0, 0, 255),
            -1
        )

        cv2.putText(
            display_image,
            str(index),
            (x + 4, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (0, 0, 255),
            1
        )


# 見やすく拡大
display_image = cv2.resize(
    display_image,
    (768, 768),
    interpolation=cv2.INTER_NEAREST
)

cv2.imshow(
    "PipeTRT Palm Detection",
    display_image
)

cv2.waitKey(0)

cv2.destroyAllWindows()