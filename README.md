# pipeTRT
MediaPipe-inspired TensorRT hand landmark library for NVIDIA GPUs

## roi branch
このブランチは、pipeTRTのroi検証を行うための作業場です。
このブランチだけではpipeTRTを使うことできません。動作確認済みのmainブランチか、リリースを確認してください

## Issue
問題があればmainか、修正したほうがいい場所が分かってる場合はそのブランチ内のissueを切ってください！
区切りがついたら修正します。

## roi
手のひらからAIが解析できるように四角の手の範囲を切り出す。

## 理論

入力
bbox = [x_min, y_min, x_max, y_max]
keypoints = [...]

        ↓

1. 中心座標
center_x = (x_min + x_max) / 2
center_y = (y_min + y_max) / 2

        ↓

2. bboxサイズ
width  = x_max - x_min
height = y_max - y_min

        ↓

3. ROIサイズ決定
roi_size = max(width, height)

        ↓

4. 正方形ROI
roi_x_min
roi_y_min
roi_x_max
roi_y_max

        ↓

5. 元画像に描画

## 開発ロードマップ

- [x] フェーズ0: ROI処理の調査と開発目標の作成
  - MediaPipe HandsにおけるROI処理の役割を理解する
  - Palm DetectionからROI生成までの流れを調査する
  - ROIに必要な入力・出力を整理する

- [ ] フェーズ1: 軸に平行なROIの作成
  - Palm Detectionのbbox・keypointsを利用する
  - ROIの中心座標を計算する
  - ROIのサイズを計算する
  - 回転なしの矩形ROIを生成する
  - ROIを画像上に描画して確認する

- [ ] フェーズ2: 軸に平行なROIをHand Landmarkへ入力
  - ROIをLandmarkモデルの入力サイズへ変換する
  - Hand Landmark推論を実行する
  - ROI内で21点ランドマークを取得する
  - 元画像上へランドマークを戻して描画する

- [ ] フェーズ3: 回転ROIの作成
  - Palm Detectionのkeypointsから手の向きを計算する
  - ROIの回転角を計算する
  - 回転を考慮したROIを生成する
  - 回転ROIを画像上に描画して確認する

- [ ] フェーズ4: 回転ROIをHand Landmarkへ入力
  - 回転ROIをLandmarkモデルへ入力する
  - ROI内で21点ランドマークを取得する
  - 座標変換によって元画像上へランドマークを戻す
  - 手の角度を変えて動作確認する

- [ ] フェーズ5: ROI処理の整理・検証
  - 軸平行ROIと回転ROIの結果を比較する
  - ROI生成処理をモジュール化する
  - サンプル・テストコードを整理する
  - 処理内容をREADMEへまとめる
  - `dev`ブランチへ統合できる状態にする

