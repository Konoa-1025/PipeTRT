import cv2
import pipetrt


def main():
    # =====================================
    # PipeTRT
    # =====================================

    landmarker = pipetrt.HandLandmarker()


    # =====================================
    # Camera
    # =====================================

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Camera could not be opened.")
        landmarker.close()
        return


    # =====================================
    # Main Loop
    # =====================================

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to read camera frame.")
            break


        # =====================================
        # PipeTRT Detection
        # =====================================

        result = landmarker.detect(
            frame
        )


        # =====================================
        # Draw Landmarks
        # =====================================

        for landmark in result.hand_landmarks:
            x = int(
                landmark[0]
            )

            y = int(
                landmark[1]
            )

            cv2.circle(
                frame,
                (x, y),
                5,
                (0, 255, 255),
                -1
            )


        # =====================================
        # Preview
        # =====================================

        cv2.imshow(
            "PipeTRT Hand Landmarks",
            frame
        )


        # Q / ESC で終了
        key = (
            cv2.waitKey(1)
            & 0xFF
        )

        if key in (
            ord("q"),
            27
        ):
            break


    # =====================================
    # Cleanup
    # =====================================

    landmarker.close()
    cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()