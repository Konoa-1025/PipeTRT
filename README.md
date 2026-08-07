# PipeTRT
MediaPipe-inspired TensorRT hand landmark library for NVIDIA GPUs

## landmark branch
このブランチは、pipeTRTのlandmark検証を行うための作業場です。
このブランチだけではpipeTRTを使うことできません。動作確認済みのmainブランチか、リリースを確認してください

## Issue
問題があればmainか、修正したほうがいい場所が分かってる場合はそのブランチ内のissueを切ってください！
区切りがついたら修正します。

## 開発目標（ゴール）
- フェーズ0:企画・調査
- フェーズ1:ONNX Runtimeでhand認識を完成
- フェーズ2:動画追跡
- フェーズ3:TensorRTバックエンド
- フェーズ4:Pythonライブラリ化
- フェーズ5:品質検証
- フェーズ6:実機操作
- フェーズ7:公開準備
- フェーズ8:公開
- フェーズ?:Pose,Faceの導入

## 現在の進捗
### フェーズ0 8/5
- [x] リポジトリの作成
- [x] READMEの作成
- [x] 既存実装の調査

### フェーズ1 8/6
- [x] Hand LandmarkのONNXモデルを用意
- [x] 入出力shapeを確認
- [x] 1枚の画像を前処理
- [x] ONNX Runtimeで推論
- [x] 出力値を表示
- [x] カメラ映像でリアルタイム推論