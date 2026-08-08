# pipetrt/api/hand_landmarker.py

from .hand_landmarker_result import HandLandmarkerResult


class HandLandmarker:
    def __init__(self):
        # Palm用Engineをここで初期化
        pass

    def detect(self, frame):
        # 1. Palm前処理
        # 2. TensorRT推論
        # 3. Decoder
        # 4. palm_resultを返す

        return HandLandmarkerResult(
            palm_result=palm_result
        )

    def close(self):
        # Engine / CUDAリソース解放
        pass