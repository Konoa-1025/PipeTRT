# pipetrt/api/hand_landmarker_result.py

class HandLandmarkerResult:
    def __init__(
        self,
        hand_landmarks=None,
        palm_result=None,
        roi=None
    ):
        self.hand_landmarks = (
            hand_landmarks
            if hand_landmarks is not None
            else []
        )

        self.palm_result = palm_result
        self.roi = roi