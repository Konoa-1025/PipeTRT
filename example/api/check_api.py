import cv2
import pipetrt


frame = cv2.imread("example/data/hand.jpg")

if frame is None:
    raise FileNotFoundError("画像を読み込めませんでした")


landmarker = pipetrt.HandLandmarker()

result = landmarker.detect(frame)


print("Palm Result:")
print(result.palm_result)
print(result.roi)

print(result.palm_result)

print()

print("ROI Result:")
print(result.roi)


landmarker.close()