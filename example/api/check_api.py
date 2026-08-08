import cv2
import pipetrt


frame = cv2.imread("example/data/hand.jpg")

landmarker = pipetrt.HandLandmarker()

result = landmarker.detect(frame)

print(result.palm_result)

landmarker.close()