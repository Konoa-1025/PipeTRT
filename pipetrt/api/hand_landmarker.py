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

        # Landmark ONNX Runtime
        self.landmark_model = ONNXInference()

    def detect(self, frame):
        # Palm
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

        # Landmark
        landmark_outputs = self.landmark_model.infer_frame(
            roi_image
        )

        print("Landmark Output:")
        for index, output in enumerate(landmark_outputs):
            print(
                index,
                output.shape,
                output
            )

        return HandLandmarkerResult(
            palm_result=palm_results,
            roi=roi,
            roi_image=roi_image,
            hand_landmarks=[]
        )

    def close(self):
        pass