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

  - 開発方針を決定する
  - ディレクトリ構成を決定する
  - ブランチ構成を決定する
  - 開発ロードマップを作成する

- [x] **フェーズ1：モデルの用意**

  - Palm Detectionモデルを用意する
  - Hand Landmarkモデルを用意する
  - モデル構造を確認する
  - 入力・出力サイズを確認する

- [ ] **フェーズ2：画像前処理**

  - モデル入力サイズへリサイズする
  - RGB変換を行う
  - Tensorへ変換する
  - 前処理結果を確認する

- [ ] **フェーズ3：TensorRT Engine**

  - Palm Engineを生成する
  - Palm Engineを読み込む
  - Palm Engineで推論する
  - Landmark Engineを生成する
  - Landmark Engineを読み込む
  - Landmark Engineで推論する

- [ ] **フェーズ4：Palm Detection**

  - Palmモデルを推論する
  - Raw Outputを取得する
  - Palm Decoderを実装する
  - Bounding Boxを取得する
  - KeyPointを取得する
  - 検出結果を描画する

- [ ] **フェーズ5：ROI**

  - Palm Detection結果からROIを生成する
  - ROIをCropする
  - ROIをLandmark入力サイズへ変換する
  - Landmarkへ入力する
  - 元画像座標へ変換する

- [ ] **フェーズ6：Hand Landmark**

  - Landmarkモデルを推論する
  - Raw Outputを取得する
  - Landmark Decoderを実装する
  - 21点ランドマークを取得する
  - ランドマークを描画する

- [ ] **フェーズ7：Tracking**

  - Landmarkから次フレームROIを生成する
  - Tracking中はPalm Detectionを省略する
  - Tracking失敗時はPalm Detectionへ戻る

- [ ] **フェーズ8：API**

  - `Hands`クラスを作成する
  - Palm・ROI・Landmark・Trackingを統合する
  - Engine自動生成を実装する
  - MediaPipe互換APIを作成する

---

## 現在の進捗

- [x] フェーズ0：調査と開発目標の作成
- [ ] フェーズ1：モデルの用意
- [ ] フェーズ2：画像前処理
- [ ] フェーズ3：TensorRT Engine
- [ ] フェーズ4：Palm Detection
- [ ] フェーズ5：ROI
- [ ] フェーズ6：Hand Landmark
- [ ] フェーズ7：Tracking
- [ ] フェーズ8：API
