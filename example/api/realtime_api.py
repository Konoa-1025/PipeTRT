import cv2
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

    # ROI
    if result.roi is not None:
        roi = result.roi

        center_x = int(roi["center_x"] * width)
        center_y = int(roi["center_y"] * height)

        roi_width = int(roi["width"] * width)
        roi_height = int(roi["height"] * height)

        roi_x1 = int(center_x - roi_width / 2)
        roi_y1 = int(center_y - roi_height / 2)

        roi_x2 = int(center_x + roi_width / 2)
        roi_y2 = int(center_y + roi_height / 2)

        cv2.rectangle(
            frame,
            (roi_x1, roi_y1),
            (roi_x2, roi_y2),
            (0, 255, 0),
            2
        )

        cv2.circle(
            frame,
            (center_x, center_y),
            5,
            (0, 0, 255),
            -1
        )

    cv2.imshow("PipeTRT API", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break


landmarker.close()
cap.release()
cv2.destroyAllWindows()