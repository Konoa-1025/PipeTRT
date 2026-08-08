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

入力｜Palm Detection のデコード結果 -> roi.py

{
    "score": 0.97,
    "center_x": ...,
    "center_y": ...,
    "width": ...,
    "height": ...,
    "keypoints": [...]
}

出力｜軸平行ROI

{
    "center_x": ...,
    "center_y": ...,
    "width": ...,
    "height": ...
}


入力｜軸平行ROI + 元画像 -> crop.py

出力｜Landmark入力用ROI画像


入力｜Palm Detection のデコード結果 -> rotation.py

出力｜回転情報付きROI

{
    "center_x": ...,
    "center_y": ...,
    "width": ...,
    "height": ...,
    "rotation": ...
}


入力｜回転ROI + 元画像 -> crop.py

出力｜回転・切り抜き済みLandmark入力画像


入力｜Landmark入力画像 -> Hand Landmark Model

出力｜21点ランドマーク

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

