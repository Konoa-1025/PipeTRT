import cv2


def main():
    for camera_id in range(10):
        camera = cv2.VideoCapture(
            camera_id,
            cv2.CAP_DSHOW,
        )

        opened = camera.isOpened()

        print(
            f"camera_id={camera_id}: "
            f"{'OPEN' if opened else 'NG'}"
        )

        if opened:
            success, frame = camera.read()

            if success:
                print(f"  frame shape: {frame.shape}")
            else:
                print("  カメラは開いたがフレーム取得失敗")

        camera.release()


if __name__ == "__main__":
    main()