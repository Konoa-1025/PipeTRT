import cv2
import math
import numpy as np
import pipetrt


landmarker = pipetrt.HandLandmarker()

cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()

    if not ret:
        break

    result = landmarker.detect(frame)

    height, width = frame.shape[:2]

    # Palm bbox
    if result.palm_result:
        palm = result.palm_result[0]

        x_min, y_min, x_max, y_max = palm["bbox"]

        palm_x1 = int(x_min * width)
        palm_y1 = int(y_min * height)
        palm_x2 = int(x_max * width)
        palm_y2 = int(y_max * height)

        cv2.rectangle(
            frame,
            (palm_x1, palm_y1),
            (palm_x2, palm_y2),
            (255, 0, 0),
            2
        )

    # Rotated ROI
    if result.roi is not None:
        roi = result.roi

        center_x = roi["center_x"] * width
        center_y = roi["center_y"] * height

        roi_width = roi["width"] * width
        roi_height = roi["height"] * height

        rotation_degree = math.degrees(
            roi["rotation"]
        )

        rotated_rect = (
            (center_x, center_y),
            (roi_width, roi_height),
            rotation_degree
        )

        box = cv2.boxPoints(
            rotated_rect
        )

        box = box.astype(
            np.int32
        )

        cv2.polylines(
            frame,
            [box],
            True,
            (0, 255, 0),
            2
        )

        cv2.circle(
            frame,
            (
                int(center_x),
                int(center_y)
            ),
            5,
            (0, 0, 255),
            -1
        )

    # 224x224 ROI画像 + 21点Landmark
    if result.roi_image is not None:
        roi_view = result.roi_image.copy()

        for landmark in result.hand_landmarks:
            x = int(landmark[0])
            y = int(landmark[1])

            cv2.circle(
                roi_view,
                (x, y),
                3,
                (0, 255, 0),
                -1
            )

        cv2.imshow(
            "Landmark Input ROI",
            roi_view
        )

    cv2.imshow(
        "PipeTRT API",
        frame
    )

    if cv2.waitKey(1) & 0xFF == 27:
        break


landmarker.close()
cap.release()
cv2.destroyAllWindows()