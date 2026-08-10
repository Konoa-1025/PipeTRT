# PipeTRT JP

MediaPipe-inspired TensorRT hand landmark library for NVIDIA GPUs

MediaPipeのハンドトラッキング構成を参考に開発した、NVIDIA GPU向けTensorRT手骨格検出ライブラリです。

このプロジェクトは、MediaPipeのようなシンプルな使い方で、TensorRTによる高速な手の検出・21点ランドマーク推論を利用できるようにすることを目的としています。

PipeTRTは、自分自身の開発だけではなく、将来同じようにJetsonやNVIDIA GPUを用いたエッジAI開発を行う学生や開発者が、手軽に利用できるライブラリを目指しています。

## デモンストレーション映像

![PipeTRT v0.1.0 Demo](./assets/pipetrt0.1.0_demo_full.gif)


## リリース 0.1.0

PipeTRTの最初のリリースです。

現在のバージョンでは、カメラ映像や画像から手を検出し、21点のHand Landmarkを取得できます。

内部では以下の処理を行います。

1. Palm Detection
2. ROI生成・変換
3. Hand Landmark推論
4. ROI Tracking
5. 21点ランドマークの出力

Palm DetectionとHand Landmarkの推論にはNVIDIA TensorRTを使用しています。

> [!NOTE]
> v0.1.0は初期リリースです。
> APIや内部仕様は今後変更される可能性があります。


## 機能

v0.1.0では以下の機能を利用できます。

- TensorRTによるPalm Detection
- 手領域（ROI）の生成
- 回転を考慮したROI変換
- TensorRTによるHand Landmark推論
- 21点のHand Landmark取得
- Landmarkを利用したROI Tracking
- Palm DetectionとTrackingの自動切り替え
- MediaPipe風の`HandLandmarker` API
- 静止画・リアルタイムカメラ映像への対応
- 各処理の推論時間・処理時間の取得


## 動作確認環境

現在、以下の環境で動作確認しています。

- Windows
- Python 3.11
- NVIDIA GPU
- NVIDIA TensorRT
- CUDA

開発時の主な動作確認GPU：

- NVIDIA GeForce RTX 3070

その他の環境については、現在検証中です。


## インストール

現在はPyPIでは公開していないため、GitHubからインストールしてください。

リポジトリをCloneします。

```bash
git clone https://github.com/Konoa-1025/PipeTRT.git
cd PipeTRT
```

PipeTRTをインストールします。

```bash
pip install .
```

開発目的の場合はEditable Installも利用できます。

```bash
pip install -e .
```


## 使い方

### 基本

```python
import pipetrt

hands = pipetrt.HandLandmarker()

result = hands.detect(frame)

print(result.hand_landmarks)

hands.close()
```

`detect()`にOpenCVの画像（NumPy配列）を渡すことで、手のランドマークを取得できます。


### OpenCVでリアルタイム検出

```python
import cv2
import pipetrt


hands = pipetrt.HandLandmarker()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    result = hands.detect(frame)

    if result.hand_landmarks is not None:
        for landmark in result.hand_landmarks:
            x = int(landmark[0])
            y = int(landmark[1])

            cv2.circle(
                frame,
                (x, y),
                4,
                (0, 255, 0),
                -1
            )

    cv2.imshow("PipeTRT", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
hands.close()
cv2.destroyAllWindows()
```


## 取得できるデータ

`HandLandmarker.detect()`は`HandLandmarkerResult`を返します。

現在、以下のデータを取得できます。

- `hand_landmarks`
- `palm_result`
- `roi`
- `roi_image`
- `timings`

### hand_landmarks

21点のHand Landmarkを取得できます。

```python
result.hand_landmarks
```

### timings

各処理にかかった時間を取得できます。

```python
print(result.timings)
```

Palm Detection、ROI変換、Landmark推論などの処理時間を確認できます。


## 処理の流れ

```text
Input Frame
     |
     v
Palm Detection
     |
     v
ROI Generation
     |
     v
Hand Landmark
     |
     v
ROI Tracking
     |
     v
21 Hand Landmarks
```


## 注意

- 本プロジェクトはMediaPipeにインスパイアされていますが、Google公式のMediaPipeプロジェクトとは独立した非公式プロジェクトです。
- TensorRTを利用していますが、NVIDIA公式のプロジェクトではありません。
- GitHub上などで公開されている情報・データを調査し、利用条件を確認した上で開発しています。使用・参考にしたものについては`reference.md`に記載しています。
- このライブラリでは画像処理をローカル環境内で実行します。PipeTRT自体が画像を第三者へ送信したり、外部サーバーへ保存したりする機能はありません。
- 使用しているデータ等について権利者から案内・通告があった場合、プログラムの修正や該当リリースの公開停止を行う場合があります。
- 現在の開発状況については、各ブランチの「現在の進捗状況」を確認してください。
- このプロジェクトでは学習を目的としてAIを開発補助に使用しています。生成された内容をそのまま公開するのではなく、内容の確認・理解・独自の実装やリファクタリングを行った上でリリースしています。そのため、開発版とリリース版には時間差が生じる場合があります。最新の開発状況については`dev`ブランチを確認してください。


## License

Apache License 2.0