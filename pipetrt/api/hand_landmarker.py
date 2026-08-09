from pipetrt.palm.decoder import decode
from pipetrt.palm.anchors import generate_anchors
from pipetrt.palm.tensorrt_inference import PalmTensorRTInference

from pipetrt.roi.roi import create_roi
from pipetrt.roi.transform import (
    extract_roi,
    restore_landmarks_to_image
)

from pipetrt.landmark.tensorrt_inference import TensorRTInference

from pipetrt.api.hand_landmarker_result import HandLandmarkerResult


class HandLandmarker:
    def __init__(self):
        self.anchors = generate_anchors()

        self.palm_model = PalmTensorRTInference()
        self.landmark_model = TensorRTInference()

    def detect(self, frame):
        # Palm Detection TensorRT
        palm_outputs = self.palm_model.infer_frame(
            frame
        )

        boxes = palm_outputs["Identity"]
        scores = palm_outputs["Identity_1"]

        palm_results = decode(
            boxes,
            scores,
            self.anchors
        )

        if not palm_results:
            return HandLandmarkerResult(
                palm_result=[],
                roi=None,
                roi_image=None,
                hand_landmarks=[]
            )

        # ROI
        image_height, image_width = frame.shape[:2]

        roi = create_roi(
            palm_results[0],
            image_width,
            image_height
        )

        roi_image, transform = extract_roi(
            frame,
            roi,
            output_size=224
        )

        # Landmark TensorRT
        landmark_outputs = (
            self.landmark_model.infer_frame(
                roi_image
            )
        )

        roi_landmarks = (
            landmark_outputs["Identity"]
            .reshape(
                21,
                3
            )
        )

        # ROI座標 → 元画像座標
        image_landmarks = (
            restore_landmarks_to_image(
                roi_landmarks,
                transform
            )
        )

        return HandLandmarkerResult(
            palm_result=palm_results,
            roi=roi,
            roi_image=roi_image,
            hand_landmarks=image_landmarks
        )

    def close(self):
        self.palm_model.close()
        self.landmark_model.close()