# pipetrt/api/hand_landmarker.py
from pipetrt.palm.preprocess import preprocess
from pipetrt.palm.detector import detect
from pipetrt.palm.decoder import decode
from pipetrt.palm.anchors import generate_anchors

from .hand_landmarker_result import HandLandmarkerResult


class HandLandmarker:
    def __init__(self):
        self.anchors = generate_anchors()

    def detect(self, frame):
        palm_input = preprocess(frame)

        boxes, scores = detect(palm_input)

        palm_results = decode(
            boxes,
            scores,
            self.anchors
        )

        return HandLandmarkerResult(
            palm_result=palm_results
        )

    def close(self):
        pass