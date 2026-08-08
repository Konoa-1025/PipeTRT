# pipetrt/api/hand_landmarker.py

from .hand_landmarker_result import HandLandmarkerResult


class HandLandmarker:
    def __init__(self):
        pass

    def detect(self, frame):
        return HandLandmarkerResult()

    def close(self):
        pass