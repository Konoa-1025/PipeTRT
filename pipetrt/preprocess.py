# pipetrt/preprocess.py
# Hand Landmarkモデルへ入力する画像の前処理

from pathlib import Path

import cv2
import numpy as np


INPUT_WIDTH = 224
INPUT_HEIGHT = 224


def preprocess_image(image_path):
    """
    画像をHand Landmarkモデルの入力形式へ変換する。

    入力:
        画像ファイルのパス

    出力:
        shape = (1, 3, 224, 224)
        dtype = float32
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"画像が見つかりません: {image_path.resolve()}"
        )

    # OpenCVで画像を読み込む
    # この時点では BGR、shapeは (高さ, 幅, チャンネル)
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(
            f"画像を読み込めませんでした: {image_path.resolve()}"
        )

    print(f"読み込み後 shape : {image.shape}")
    print(f"読み込み後 dtype : {image.dtype}")
    print(f"読み込み後 range : {image.min()} ～ {image.max()}")

    # モデルの入力サイズへ変更
    image = cv2.resize(
        image,
        (INPUT_WIDTH, INPUT_HEIGHT),
        interpolation=cv2.INTER_LINEAR,
    )

    # OpenCVのBGRから、モデル用のRGBへ変換
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # uint8からfloat32へ変換
    image = image.astype(np.float32)

    # 画素値を 0～255 から 0.0～1.0 へ正規化
    image = image / 255.0

    # (高さ, 幅, チャンネル) → (チャンネル, 高さ, 幅)
    image = np.transpose(image, (2, 0, 1))

    # (3, 224, 224) → (1, 3, 224, 224)
    image = np.expand_dims(image, axis=0)

    # TensorRTなどが扱いやすいよう、メモリを連続配置にする
    image = np.ascontiguousarray(image, dtype=np.float32)

    return image