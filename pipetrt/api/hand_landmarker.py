import time

from pipetrt.palm.decoder import decode
from pipetrt.palm.anchors import generate_anchors
from pipetrt.palm.tensorrt_inference import PalmTensorRTInference

from pipetrt.roi.roi import create_roi
from pipetrt.roi.transform import (
    extract_roi,
    restore_landmarks_to_image
)

from pipetrt.landmark.tensorrt_inference import TensorRTInference

from pipetrt.tracking.roi import create_tracking_roi

from pipetrt.api.hand_landmarker_result import HandLandmarkerResult

from pipetrt.engines.manager import EngineManager


import time

from pipetrt.palm.decoder import decode
from pipetrt.palm.anchors import generate_anchors
from pipetrt.palm.tensorrt_inference import PalmTensorRTInference

from pipetrt.roi.roi import create_roi
from pipetrt.roi.transform import (
    extract_roi,
    restore_landmarks_to_image
)

from pipetrt.landmark.tensorrt_inference import TensorRTInference

from pipetrt.tracking.roi import create_tracking_roi

from pipetrt.engines.manager import EngineManager

from pipetrt.api.hand_landmarker_result import HandLandmarkerResult


class HandLandmarker:
    def __init__(
        self,
        model="full",
        precision="fp16"
    ):
        self.model = model
        self.precision = precision

        self.anchors = generate_anchors()

        # Engineを確認
        # 無ければONNXから自動生成
        self.engine_manager = EngineManager(
            model=self.model,
            precision=self.precision
        )

        engine_paths = (
            self.engine_manager.ensure_engines()
        )

        # 生成済みEngineを使用
        self.palm_model = PalmTensorRTInference(
            engine_path=engine_paths["palm"]
        )

        self.landmark_model = TensorRTInference(
            engine_path=engine_paths["landmark"]
        )

        self.tracking = False
        self.tracking_roi = None


    def detect(self, frame):
        total_start = time.perf_counter()

        image_height, image_width = frame.shape[:2]

        # =====================================
        # TRACKING MODE
        # =====================================

        if (
            self.tracking
            and self.tracking_roi is not None
        ):
            roi = self.tracking_roi

            # -----------------------------
            # ROI Transform
            # -----------------------------

            transform_start = time.perf_counter()

            roi_image, transform = extract_roi(
                frame,
                roi,
                output_size=224
            )

            transform_end = time.perf_counter()

            # -----------------------------
            # Landmark TensorRT
            # -----------------------------

            landmark_start = time.perf_counter()

            landmark_outputs = (
                self.landmark_model.infer_frame(
                    roi_image
                )
            )

            landmark_end = time.perf_counter()

            hand_presence = float(
                landmark_outputs["Identity_1"][0][0]
            )

            # MediaPipeと同じく0.5未満なら
            # Tracking失敗と判断
            if hand_presence < 0.5:
                self.tracking = False
                self.tracking_roi = None

                total_end = time.perf_counter()

                return HandLandmarkerResult(
                    palm_result=[],
                    roi=roi,
                    roi_image=roi_image,
                    hand_landmarks=[],
                    timings={
                        "mode": "TRACKING_LOST",

                        "palm_trt_ms": 0.0,
                        "palm_decode_ms": 0.0,
                        "roi_ms": 0.0,

                        "roi_transform_ms":
                            (
                                transform_end
                                - transform_start
                            ) * 1000.0,

                        "landmark_trt_ms":
                            (
                                landmark_end
                                - landmark_start
                            ) * 1000.0,

                        "restore_ms": 0.0,

                        "total_ms":
                            (
                                total_end
                                - total_start
                            ) * 1000.0,
                    }
                )

            # -----------------------------
            # Landmark取得
            # -----------------------------

            roi_landmarks = (
                landmark_outputs["Identity"]
                .reshape(
                    21,
                    3
                )
            )

            # -----------------------------
            # ROI座標 → 元画像座標
            # -----------------------------

            restore_start = time.perf_counter()

            image_landmarks = (
                restore_landmarks_to_image(
                    roi_landmarks,
                    transform
                )
            )

            restore_end = time.perf_counter()

            # -----------------------------
            # 次フレーム用Tracking ROI更新
            # -----------------------------

            new_tracking_roi = create_tracking_roi(
                image_landmarks,
                image_width,
                image_height
            )

            # Tracking ROI生成失敗
            if new_tracking_roi is None:
                self.tracking = False
                self.tracking_roi = None

            else:
                # -----------------------------
                # ROI急縮小チェック
                # -----------------------------

                old_area = (
                    self.tracking_roi["width"]
                    * self.tracking_roi["height"]
                )

                new_area = (
                    new_tracking_roi["width"]
                    * new_tracking_roi["height"]
                )

                if old_area > 0:
                    area_ratio = (
                        new_area / old_area
                    )
                else:
                    area_ratio = 0.0

                # 前フレームの50%未満まで
                # 一気に縮んだらTracking失敗
                if area_ratio < 0.5:
                    self.tracking = False
                    self.tracking_roi = None

                else:
                    # 正常なら次フレーム用ROIとして採用
                    self.tracking_roi = new_tracking_roi

            # ROI生成失敗
            if self.tracking_roi is None:
                self.tracking = False

            total_end = time.perf_counter()

            return HandLandmarkerResult(
                palm_result=[],
                roi=roi,
                roi_image=roi_image,
                hand_landmarks=image_landmarks,
                timings={
                    "mode": "TRACKING",

                    "palm_trt_ms": 0.0,
                    "palm_decode_ms": 0.0,
                    "roi_ms": 0.0,

                    "roi_transform_ms":
                        (
                            transform_end
                            - transform_start
                        ) * 1000.0,

                    "landmark_trt_ms":
                        (
                            landmark_end
                            - landmark_start
                        ) * 1000.0,

                    "restore_ms":
                        (
                            restore_end
                            - restore_start
                        ) * 1000.0,

                    "total_ms":
                        (
                            total_end
                            - total_start
                        ) * 1000.0,
                }
            )

        # =====================================
        # DETECTION MODE
        # =====================================

        # -----------------------------
        # Palm TensorRT
        # -----------------------------

        palm_start = time.perf_counter()

        palm_outputs = (
            self.palm_model.infer_frame(
                frame
            )
        )

        palm_end = time.perf_counter()

        boxes = palm_outputs["Identity"]
        scores = palm_outputs["Identity_1"]

        # -----------------------------
        # Palm Decode
        # -----------------------------

        decode_start = time.perf_counter()

        palm_results = decode(
            boxes,
            scores,
            self.anchors
        )

        decode_end = time.perf_counter()

        # -----------------------------
        # Palmが見つからない
        # -----------------------------

        if not palm_results:
            self.tracking = False
            self.tracking_roi = None

            total_end = time.perf_counter()

            return HandLandmarkerResult(
                palm_result=[],
                roi=None,
                roi_image=None,
                hand_landmarks=[],
                timings={
                    "mode": "DETECTION",

                    "palm_trt_ms":
                        (
                            palm_end
                            - palm_start
                        ) * 1000.0,

                    "palm_decode_ms":
                        (
                            decode_end
                            - decode_start
                        ) * 1000.0,

                    "roi_ms": 0.0,
                    "roi_transform_ms": 0.0,
                    "landmark_trt_ms": 0.0,
                    "restore_ms": 0.0,

                    "total_ms":
                        (
                            total_end
                            - total_start
                        ) * 1000.0,
                }
            )

        # -----------------------------
        # Palm → ROI
        # -----------------------------

        roi_start = time.perf_counter()

        roi = create_roi(
            palm_results[0],
            image_width,
            image_height
        )

        roi_end = time.perf_counter()

        # -----------------------------
        # ROI Transform
        # -----------------------------

        transform_start = time.perf_counter()

        roi_image, transform = extract_roi(
            frame,
            roi,
            output_size=224
        )

        transform_end = time.perf_counter()

        # -----------------------------
        # Landmark TensorRT
        # -----------------------------

        landmark_start = time.perf_counter()

        landmark_outputs = (
            self.landmark_model.infer_frame(
                roi_image
            )
        )

        landmark_end = time.perf_counter()

        roi_landmarks = (
            landmark_outputs["Identity"]
            .reshape(
                21,
                3
            )
        )

        # -----------------------------
        # ROI座標 → 元画像座標
        # -----------------------------

        restore_start = time.perf_counter()

        image_landmarks = (
            restore_landmarks_to_image(
                roi_landmarks,
                transform
            )
        )

        restore_end = time.perf_counter()

        # -----------------------------
        # Tracking開始
        # -----------------------------

        self.tracking_roi = (
            create_tracking_roi(
                image_landmarks,
                image_width,
                image_height
            )
        )

        if self.tracking_roi is not None:
            self.tracking = True

        total_end = time.perf_counter()

        return HandLandmarkerResult(
            palm_result=palm_results,
            roi=roi,
            roi_image=roi_image,
            hand_landmarks=image_landmarks,
            timings={
                "mode": "DETECTION",

                "palm_trt_ms":
                    (
                        palm_end
                        - palm_start
                    ) * 1000.0,

                "palm_decode_ms":
                    (
                        decode_end
                        - decode_start
                    ) * 1000.0,

                "roi_ms":
                    (
                        roi_end
                        - roi_start
                    ) * 1000.0,

                "roi_transform_ms":
                    (
                        transform_end
                        - transform_start
                    ) * 1000.0,

                "landmark_trt_ms":
                    (
                        landmark_end
                        - landmark_start
                    ) * 1000.0,

                "restore_ms":
                    (
                        restore_end
                        - restore_start
                    ) * 1000.0,

                "total_ms":
                    (
                        total_end
                        - total_start
                    ) * 1000.0,
            }
        )

    def close(self):
        self.palm_model.close()
        self.landmark_model.close()