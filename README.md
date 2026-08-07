# pipeTRT
MediaPipe-inspired TensorRT hand landmark library for NVIDIA GPUs

## palm branch
このブランチは、pipeTRTのpalm検証を行うための作業場です。（パルムおいしいよね。食べたくなった。）

このブランチだけではpipeTRTを使うことできません。動作確認済みのmainブランチか、リリースを確認してください


## Issue
問題があればmainか、修正したほうがいい場所が分かってる場合はそのブランチ内のissueを切ってください！
区切りがついたら修正します。

## palm
画像全体から手のひらを検出し、後続のROI生成に必要な位置情報を取得する。

Palm Detectionは、Hand Landmarkモデルへ渡す「手の領域」を見つけるための最初の処理。

## 理論
入力｜カメラ画像 / 静止画 -> detector.py

出力｜Raw Output  

入力｜Raw Output -> decoder.py

出力｜{ "score": 0.97, "center_x": ..., "center_y": ..., "width": ..., "height": ..., "keypoints": [...] }

## 開発目標（ゴール）

- [x] **フェーズ0：調査と開発目標の作成**
  - Palm Detectionの役割を調査する
  - 使用するモデルを決定する
  - 入力と出力の仕様を確認する
  - 開発ロードマップを作成する

- [ ] **フェーズ1：モデルの用意**
  - Palm DetectionのONNXモデルを用意する
  - モデルの入力shapeを確認する
  - モデルの出力shapeを確認する
  - 入出力データの型を確認する

- [ ] **フェーズ2：画像前処理**
  - 入力画像をモデル入力サイズへリサイズする
  - 色形式を変換する
  - モデルが要求する値の範囲へ変換する
  - Tensor形式へ変換する
  - 前処理後のshapeを確認する

- [ ] **フェーズ3：ONNX RuntimeでPalm推論**
  - ONNX Runtimeでモデルを読み込む
  - 前処理した画像を入力する
  - Palm Detectionを実行する
  - Raw Outputを取得する
  - Raw Outputのshapeと内容を確認する

- [ ] **フェーズ4：Palm Decoder**
  - Raw Outputの構造を理解する
  - 検出スコアをデコードする
  - Bounding Boxをデコードする
  - KeyPointをデコードする
  - 必要な後処理を実装する
  - 扱いやすいPalm Detection Resultとして返す

- [ ] **フェーズ5：検出結果の確認**
  - Bounding Boxを元画像へ描画する
  - KeyPointを元画像へ描画する
  - 検出信頼度を表示する
  - 静止画で検出を確認する
  - カメラ映像でリアルタイム検出を確認する

- [ ] **フェーズ6：TensorRT Engine**
  - Palm Detection ONNXからEngineを生成する
  - Engineをファイルへ保存する
  - Engineを読み込む
  - TensorRTでPalm推論を実行する
  - TensorRTのRaw OutputをDecoderへ渡す
  - ONNX Runtime版と検出結果を比較する
  - 推論速度を比較する

- [ ] **フェーズ7：Palm Detection完成**
  - Palm処理を整理する
  - 推論処理とDecoderを分離する
  - example用プログラムを整理する
  - ROIから利用できる形式で検出結果を返せるようにする
  - `dev`ブランチへ統合できる状態にする

---
