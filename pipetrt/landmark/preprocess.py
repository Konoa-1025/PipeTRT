# pipetrt/preprocess.py

from pathlib import Path

import cv2
import numpy as np


INPUT_WIDTH = 224
INPUT_HEIGHT = 224


def preprocess_frame(image):
    """
    OpenCV画像をモデルの入力形式へ変換する。

    入力:
        OpenCV画像
        shape = (高さ, 幅, 3)
        BGR形式

    出力:
        shape = (1, 3, 224, 224)
        dtype = float32
    """

    if image is None:
        raise ValueError("入力画像がNoneです")

    image = cv2.resize(
        image,
        (INPUT_WIDTH, INPUT_HEIGHT),
        interpolation=cv2.INTER_LINEAR,
    )

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = image.astype(np.float32)
    image = image / 255.0

    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)

    image = np.ascontiguousarray(
        image,
        dtype=np.float32,
    )

    return image


def preprocess_image(image_path):
    """
    画像ファイルを読み込み、モデル入力形式へ変換する。
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"画像が見つかりません: {image_path.resolve()}"
        )

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(
            f"画像を読み込めませんでした: {image_path.resolve()}"
        )

    return preprocess_frame(image)