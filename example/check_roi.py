import cv2

from pipetrt.roi.roi import create_axis_aligned_roi


palm_result = {
    "score": 0.9730098843574524,
    "bbox": [
        0.386535,
        0.47173136,
        0.68325907,
        0.7684122
    ],
    "keypoints": [
        [0.55862015, 0.758764],
        [0.5870974, 0.48125684],
        [0.5182824, 0.4943241],
        [0.45346725, 0.516191],
        [0.39253128, 0.547279],
        [0.6273008, 0.6887498],
        [0.6880876, 0.58719814]
    ]
}


image = cv2.imread("example/data/hand.jpg")

roi = create_axis_aligned_roi(palm_result)

image_height, image_width = image.shape[:2]

center_x = int(roi["center_x"] * image_width)
center_y = int(roi["center_y"] * image_height)

roi_width = int(roi["width"] * image_width)
roi_height = int(roi["height"] * image_height)

x_min = int(center_x - roi_width / 2)
y_min = int(center_y - roi_height / 2)

x_max = int(center_x + roi_width / 2)
y_max = int(center_y + roi_height / 2)

cv2.rectangle(
    image,
    (x_min, y_min),
    (x_max, y_max),
    (0, 255, 0),
    2
)

cv2.circle(
    image,
    (center_x, center_y),
    5,
    (0, 0, 255),
    -1
)

cv2.imshow("ROI Check", image)
cv2.waitKey(0)
cv2.destroyAllWindows()