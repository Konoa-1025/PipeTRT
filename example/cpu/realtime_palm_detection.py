import cv2

from pipetrt.palm.preprocess import preprocess
from pipetrt.palm.detector import detect
from pipetrt.palm.decoder import decode
from pipetrt.palm.anchors import generate_anchors


anchors = generate_anchors()

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("カメラを開けませんでした")


window_name = "PipeTRT Realtime Palm Detection"

cv2.namedWindow(
    window_name,
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    window_name,
    960,
    720
)


while True:
    success, frame = camera.read()

    if not success:
        print("フレーム取得に失敗しました")
        break

    # --------------------------------
    # Palm前処理
    # --------------------------------

    palm_input = preprocess(frame)

    # --------------------------------
    # Palm推論
    # --------------------------------

    boxes, scores = detect(
        palm_input
    )

    # --------------------------------
    # Decode
    # --------------------------------

    results = decode(
        boxes,
        scores,
        anchors
    )

    # --------------------------------
    # Palmモデルが見た画像を表示用に戻す
    # --------------------------------

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

    height, width = display_image.shape[:2]

    # --------------------------------
    # 描画
    # --------------------------------

    for result in results:
        score = result["score"]
        bbox = result["bbox"]
        keypoints = result["keypoints"]

        # Bounding Box
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

        cv2.putText(
            display_image,
            f"Palm {score:.3f}",
            (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1
        )

        # 7 Keypoints
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

    # --------------------------------
    # 拡大表示
    # --------------------------------

    display_image = cv2.resize(
        display_image,
        (768, 768),
        interpolation=cv2.INTER_NEAREST
    )

    cv2.imshow(
        window_name,
        display_image
    )

    # --------------------------------
    # キー入力
    # --------------------------------

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == 27:
        break


camera.release()
cv2.destroyAllWindows()