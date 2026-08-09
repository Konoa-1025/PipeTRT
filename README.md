# pipeTRT
MediaPipe-inspired TensorRT hand landmark library for NVIDIA GPUs

## tracking branch
このブランチは、pipeTRTのtracking検証を行うための作業場です。
このブランチだけではpipeTRTを使うことできません。動作確認済みのmainブランチか、リリースを確認してください

## Issue
問題があればmainか、修正したほうがいい場所が分かってる場合はそのブランチ内のissueを切ってください！
区切りがついたら修正します。

## 開発ロードマップ
- [x] フェーズ0: Tracking ROIの生成確認

  - [x] 21点Landmarkから手全体の範囲を取得する
  - [x] WristとMiddle MCPから手の向きを計算する
  - [x] Tracking用回転ROIを生成する
  - [x] Tracking ROIを画像上に描画する
  - [x] 手の移動・回転・グー/パーで追従することを確認する

- [ ] フェーズ1: Tracking状態を作成

  -  `HandLandmarker`にTracking中かどうかの状態を追加する
  -  前フレームのTracking ROIを保存する
  -  初回はPalm Detectionを使用する
  -  Palm Detection後にTracking状態へ移行する
  -  Tracking ROIが存在しない場合はDetectionへ戻る

- [ ] フェーズ2: Palm DetectionをスキップしてLandmark推論

  -  Tracking中はPalm TensorRTを実行しない
  -  前フレームのTracking ROIから224×224画像を生成する
  -  Tracking ROIをHand Landmark TensorRTへ入力する
  -  21点を元画像座標へ戻す
  -  新しい21点から次フレーム用Tracking ROIを更新する
  -  Trackingだけで連続して手を追従できることを確認する

- [ ] フェーズ3: Tracking継続判定

  -  Hand Landmarkの信頼度出力を取得する
  -  Tracking継続用の閾値を決める
  -  信頼度が十分ならTrackingを継続する
  -  信頼度が低下したらTrackingを終了する
  -  Tracking失敗時にPalm Detectionへ戻る

- [ ] フェーズ4: Detection / Tracking切り替え確認

  -  現在の状態を`DETECTION` / `TRACKING`として表示する
  -  手を画面から消してPalm Detectionへ戻ることを確認する
  -  手を再表示してTrackingへ復帰することを確認する
  -  画面端での追従性能を確認する
  -  高速移動・回転時の挙動を確認する

- [ ] フェーズ5: Tracking性能計測

  -  Tracking時の処理時間を計測する
  -  Palm Detection実行時との処理時間を比較する
  -  FPSを比較する
  -  CPU / GPU使用率を確認する
  -  Palm TensorRTがTracking中にスキップされていることを確認する

- [ ] フェーズ6: Tracking処理の整理

  -  Tracking関連コードを`pipetrt/tracking/`へ整理する
  -  API側から内部処理を隠蔽する
  -  サンプルコードを整理する
  -  READMEへDetection / Trackingの流れをまとめる
  -  `dev`ブランチへ統合できる状態にする