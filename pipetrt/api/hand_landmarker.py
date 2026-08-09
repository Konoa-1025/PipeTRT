from pipetrt.palm.preprocess import preprocess
from pipetrt.palm.detector import detect
from pipetrt.palm.decoder import decode
from pipetrt.palm.anchors import generate_anchors

from pipetrt.roi.roi import create_roi
from pipetrt.roi.transform import extract_roi

from pipetrt.landmark.onnx_inference import ONNXInference

from pipetrt.api.hand_landmarker_result import HandLandmarkerResult


class HandLandmarker:
    def __init__(self):
        self.anchors = generate_anchors()

        self.landmark_model = ONNXInference()

    def detect(self, frame):
        # Palm Detection
        palm_input = preprocess(frame)

        boxes, scores = detect(palm_input)

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

        # ROI生成
        image_height, image_width = frame.shape[:2]

        roi = create_roi(
            palm_results[0],
            image_width,
            image_height
        )

        # 回転ROIを224x224へ変換
        roi_image, transform = extract_roi(
            frame,
            roi,
            output_size=224
        )

        # Hand Landmark
        landmark_outputs = self.landmark_model.infer_frame(
            roi_image
        )

        landmarks = landmark_outputs[0].reshape(
            21,
            3
        )

        return HandLandmarkerResult(
            palm_result=palm_results,
            roi=roi,
            roi_image=roi_image,
            hand_landmarks=landmarks
        )

    def close(self):
        pass