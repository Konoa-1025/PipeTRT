# pipeTRT
MediaPipe-inspired TensorRT hand landmark library for NVIDIA GPUs

## api branch
このブランチは、pipeTRTのAPI仕様の検証を行うための作業場です。
このブランチだけではpipeTRTを使うことできません。動作確認済みのmainブランチか、リリースを確認してください

## Issue
問題があればmainか、修正したほうがいい場所が分かってる場合はそのブランチ内のissueを切ってください！
区切りがついたら修正します。

## 開発ロードマップ

- [x] フェーズ0: APIの仕様決定
  - `import PipeTRT` で利用できる構成を決める
  - `HandLandmarker()` を公開クラスにする
  - 初期リリースではモデル指定・Optionsは実装しない
  - 推論メソッドはMediaPipe Tasks風に `detect()` を採用する
  - 戻り値は `HandLandmarkerResult` とする
  - 21点は `result.hand_landmarks` から取得する
  - `close()` でTensorRT/CUDA関連リソースを解放できるようにする
  - `process(frame)` の入力形式を決める
  - `detect(frame)` の入力形式を確定する
  - Palm → ROI → Landmark の接続仕様を整理する

- [x] フェーズ1: APIの基本構造を作成
  - `PipeTRT` パッケージの入口を作成する
  - `__init__.py` から公開APIを呼び出せるようにする
  - `HandLandmarker` クラスを作成する
  - `HandLandmarkerResult` を作成する
  - `detect(frame)` の基本処理を作成する
  - `close()` の基本処理を作成する
  - 最小構成で動作確認する

- [x] フェーズ2: Palm DetectionをAPIへ接続
  - 入力画像をPalm Detectionへ渡す
  - Palm Decoderまで実行する
  - Palm Detection結果を取得する
  - APIの戻り値としてPalm結果を返す
  - 静止画・リアルタイムで動作確認する

- [x] フェーズ3: ROIをAPIへ接続
  - Palm Detection結果をROIへ渡す
  - ROIを生成する
  - ROI情報をAPIの戻り値へ追加する
  - Palm DetectionとROIの連続処理を確認する
  - リアルタイムでROIを描画して確認する

- [x] フェーズ4: Hand LandmarkをAPIへ接続
  - ROI画像をHand Landmarkへ入力する
  - 21点ランドマークを取得する
  - ROI座標から元画像座標へ変換する
  - APIの戻り値へLandmark結果を追加する
  - 元画像上へ21点を描画して確認する

- [ ] フェーズ5: TrackingをAPIへ接続
  - Landmark結果から次フレームのROIを生成する
  - Palm Detectionを毎フレーム実行しない処理を作成する
  - Tracking失敗時にPalm Detectionへ戻る処理を作成する
  - リアルタイムでTrackingを確認する

- [ ] フェーズ6: 公開APIの整理
  - 戻り値の形式を統一する
  - 設定値を整理する
  - エラー処理を追加する
  - 型ヒントを追加する
  - 不要な内部処理を外部から隠す
  - `import PipeTRT` だけで利用できることを確認する

- [ ] フェーズ7: サンプル・ドキュメント作成
  - 静止画サンプルを作成する
  - リアルタイムカメラサンプルを作成する
  - APIの使用方法をREADMEへ記載する
  - 最小サンプルコードを作成する
  - `dev`ブランチへ統合できる状態にする