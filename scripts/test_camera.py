from pathlib import Path
import sys
import time

import cv2


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipetrt.onnx_inference import ONNXInference


CAMERA_ID = 0
MODEL_WIDTH = 224
MODEL_HEIGHT = 224


def main():

    model = ONNXInference()

    camera = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)

    if not camera.isOpened():
        raise RuntimeError(
            f"カメラを開けませんでした: camera_id={CAMERA_ID}"
        )

    previous_time = time.perf_counter()

    try:
        while True:

            success, frame = camera.read()

            if not success:
                print("カメラ映像を取得できませんでした")
                break

            # 鏡のように左右反転
            frame = cv2.flip(frame, 1)

            # 現段階では画面全体を224×224にして推論
            outputs = model.infer_frame(frame)

            landmarks = outputs[0].reshape(21, 3)

            preview = cv2.resize(
                frame,
                (MODEL_WIDTH, MODEL_HEIGHT),
            )

            for index, landmark in enumerate(landmarks):

                x = int(landmark[0])
                y = int(landmark[1])

                cv2.circle(
                    preview,
                    (x, y),
                    4,
                    (0, 255, 0),
                    -1,
                )

                cv2.putText(
                    preview,
                    str(index),
                    (x + 4, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

            current_time = time.perf_counter()
            elapsed_time = current_time - previous_time

            fps = 0.0

            if elapsed_time > 0:
                fps = 1.0 / elapsed_time

            previous_time = current_time

            hand_score = float(outputs[1][0][0])
            handedness_score = float(outputs[2][0][0])

            cv2.putText(
                preview,
                f"FPS: {fps:.1f}",
                (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.putText(
                preview,
                f"Hand: {hand_score:.3f}",
                (5, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.putText(
                preview,
                f"Side: {handedness_score:.3f}",
                (5, 49),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(
                "PipeTRT Realtime Preview",
                preview,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()