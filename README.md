# PipeTRT
MediaPipe-inspired TensorRT hand landmark library for NVIDIA GPUs

## このプロジェクトとは何か
このプロジェクトは、mediapipe風の使い方でTensorRTを使用できるようにするプロジェクトです。

PipeTRTは、自分自身の開発だけではなく、将来同じようにJetsonやNVIDIA GPUを用いたエッジAI開発を行う学生や開発者が、手軽に利用できるライブラリを目指しています。

※ mediapipeにインスパイアを受けて作られていますが、mediapipeはGoogle社が開発した機械学習フレームワークのことで直接的にこのプロジェクトに関係はしていません。

## 開発目標（ゴール）
### フェーズ0:企画・調査
### フェーズ1:ONNX Runtimeでhand認識を完成
### フェーズ2:動画追跡
### フェーズ3:TensorRTバックエンド
### フェーズ4:Pythonライブラリ化
### フェーズ5:品質検証
### フェーズ6:実機操作
### フェーズ7:公開準備
### フェーズ8:公開
### フェーズ?:Pose,Faceの導入

## 現在の進捗
### フェーズ0 8/5
- [x] リポジトリの作成
- [x] READMEの作成
- [ ] 既存実装の調査
- [ ] API設計
- [ ] モデル比較
