import cv2
import numpy as np


INPUT_WIDTH = 192
INPUT_HEIGHT = 192


def sigmoid(values):
    values = np.clip(values, -50.0, 50.0)

    return 1.0 / (
        1.0 + np.exp(-values)
    )


def decode(
    boxes,
    scores,
    anchors,
    score_threshold=0.5,
    nms_threshold=0.3
):
    raw_boxes = boxes[0]
    raw_scores = scores[0, :, 0]

    probabilities = sigmoid(raw_scores)

    box_delta = raw_boxes[:, 0:4]

    landmark_delta = raw_boxes[:, 4:18].reshape(
        -1,
        7,
        2
    )

    center_delta = box_delta[:, 0:2] / np.array(
        [INPUT_WIDTH, INPUT_HEIGHT],
        dtype=np.float32
    )

    size_delta = box_delta[:, 2:4] / np.array(
        [INPUT_WIDTH, INPUT_HEIGHT],
        dtype=np.float32
    )

    box_center = center_delta + anchors

    box_min = box_center - size_delta / 2.0
    box_max = box_center + size_delta / 2.0

    decoded_boxes = np.concatenate(
        [box_min, box_max],
        axis=1
    )

    decoded_keypoints = landmark_delta / np.array(
        [INPUT_WIDTH, INPUT_HEIGHT],
        dtype=np.float32
    )

    decoded_keypoints = (
        decoded_keypoints
        + anchors[:, np.newaxis, :]
    )

    valid_indices = np.where(
        probabilities >= score_threshold
    )[0]

    if len(valid_indices) == 0:
        return []

    candidate_boxes = decoded_boxes[
        valid_indices
    ]

    candidate_keypoints = decoded_keypoints[
        valid_indices
    ]

    candidate_scores = probabilities[
        valid_indices
    ]

    nms_boxes = []

    for box in candidate_boxes:
        x1, y1, x2, y2 = box

        nms_boxes.append(
            [
                float(x1),
                float(y1),
                float(x2 - x1),
                float(y2 - y1)
            ]
        )

    keep_indices = cv2.dnn.NMSBoxes(
        nms_boxes,
        candidate_scores.tolist(),
        score_threshold,
        nms_threshold
    )

    if len(keep_indices) == 0:
        return []

    results = []

    for keep_index in keep_indices:
        index = int(keep_index)

        results.append(
            {
                "score": float(
                    candidate_scores[index]
                ),

                "bbox": candidate_boxes[
                    index
                ].copy(),

                "keypoints": candidate_keypoints[
                    index
                ].copy()
            }
        )

    return results