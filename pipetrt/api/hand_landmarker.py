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


class HandLandmarker:
    def __init__(self):
        self.anchors = generate_anchors()

        self.palm_model = PalmTensorRTInference()
        self.landmark_model = TensorRTInference()

        # Tracking状態
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

            # 一時的なデバッグ
            print(
    "Identity_1:",
    landmark_outputs["Identity_1"]
)

            print(
    "Identity_2:",
    landmark_outputs["Identity_2"]
)

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

            self.tracking_roi = (
                create_tracking_roi(
                    image_landmarks,
                    image_width,
                    image_height
                )
            )

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