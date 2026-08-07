import cv2
import numpy as np

from pipetrt.palm.preprocess import preprocess
from pipetrt.palm.detector import detect


# ==========================================
# Anchor生成
# ==========================================

def generate_anchors():
    anchors = []

    # stride 8
    # 24 × 24 × 2 = 1152
    for y in range(24):
        for x in range(24):
            center_x = (x + 0.5) / 24
            center_y = (y + 0.5) / 24

            anchors.append([center_x, center_y])
            anchors.append([center_x, center_y])

    # stride 16
    # 12 × 12 × 6 = 864
    for y in range(12):
        for x in range(12):
            center_x = (x + 0.5) / 12
            center_y = (y + 0.5) / 12

            for count in range(6):
                anchors.append([center_x, center_y])

    return np.array(
        anchors,
        dtype=np.float32
    )


def sigmoid(values):
    # 極端な値でoverflowしないよう制限
    values = np.clip(values, -50, 50)

    return 1.0 / (
        1.0 + np.exp(-values)
    )


# ==========================================
# Model Input → 表示画像
# ==========================================

def tensor_to_image(palm_input):
    image = palm_input[0]

    image = (
        image * 255.0
    ).clip(
        0,
        255
    ).astype(
        np.uint8
    )

    # Palm入力はRGBなのでOpenCV表示用BGRへ
    image = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR
    )

    return image


# ==========================================
# Score → HeatMap色
# ==========================================

def score_to_color(score):

    # 0.0 ～ 1.0 を
    # 青 → 水色 → 緑 → 黄 → 赤
    # に変換

    value = int(
        np.clip(score, 0.0, 1.0) * 255
    )

    color_map_input = np.array(
        [[value]],
        dtype=np.uint8
    )

    color = cv2.applyColorMap(
        color_map_input,
        cv2.COLORMAP_JET
    )[0, 0]

    return (
        int(color[0]),
        int(color[1]),
        int(color[2])
    )


# ==========================================
# Best Anchor
# ==========================================

def draw_best_anchor(
    image,
    anchors,
    probabilities,
    best_index
):
    center_x = anchors[best_index][0]
    center_y = anchors[best_index][1]

    pixel_x = int(
        center_x * image.shape[1]
    )

    pixel_y = int(
        center_y * image.shape[0]
    )

    cv2.circle(
        image,
        (pixel_x, pixel_y),
        5,
        (0, 0, 255),
        -1
    )

    cv2.circle(
        image,
        (pixel_x, pixel_y),
        9,
        (0, 0, 255),
        1
    )

    return image


# ==========================================
# 全Anchor
# ==========================================

def draw_all_anchors(
    image,
    anchors,
    best_index
):
    overlay = image.copy()

    for index, anchor in enumerate(anchors):

        center_x = anchor[0]
        center_y = anchor[1]

        pixel_x = int(
            center_x * image.shape[1]
        )

        pixel_y = int(
            center_y * image.shape[0]
        )

        cv2.circle(
            overlay,
            (pixel_x, pixel_y),
            2,
            (220, 220, 220),
            -1
        )

    # 薄く合成
    image = cv2.addWeighted(
        overlay,
        0.20,
        image,
        0.80,
        0
    )

    # Bestだけ強調
    center_x = anchors[best_index][0]
    center_y = anchors[best_index][1]

    pixel_x = int(
        center_x * image.shape[1]
    )

    pixel_y = int(
        center_y * image.shape[0]
    )

    cv2.circle(
        image,
        (pixel_x, pixel_y),
        5,
        (0, 0, 255),
        -1
    )

    return image


# ==========================================
# HeatMap
# ==========================================

def draw_heatmap(
    image,
    anchors,
    probabilities,
    best_index
):
    overlay = image.copy()

    # 同じ座標に複数Anchorが存在するので
    # その位置の最大Scoreだけ使用する
    anchor_scores = {}

    for index, anchor in enumerate(anchors):

        center_x = float(anchor[0])
        center_y = float(anchor[1])

        key = (
            center_x,
            center_y
        )

        score = float(
            probabilities[index]
        )

        if key not in anchor_scores:
            anchor_scores[key] = score

        else:
            anchor_scores[key] = max(
                anchor_scores[key],
                score
            )

    for (
        center_x,
        center_y
    ), score in anchor_scores.items():

        pixel_x = int(
            center_x * image.shape[1]
        )

        pixel_y = int(
            center_y * image.shape[0]
        )

        color = score_to_color(
            score
        )

        # Scoreが高いほど少し大きくする
        radius = 2 + int(
            score * 4
        )

        cv2.circle(
            overlay,
            (pixel_x, pixel_y),
            radius,
            color,
            -1
        )

    # 半透明
    image = cv2.addWeighted(
        overlay,
        0.35,
        image,
        0.65,
        0
    )

    # Best Anchorを白枠で囲む
    center_x = anchors[best_index][0]
    center_y = anchors[best_index][1]

    pixel_x = int(
        center_x * image.shape[1]
    )

    pixel_y = int(
        center_y * image.shape[0]
    )

    cv2.circle(
        image,
        (pixel_x, pixel_y),
        8,
        (255, 255, 255),
        1
    )

    return image


# ==========================================
# 情報表示
# ==========================================

def draw_information(
    image,
    mode,
    best_index,
    best_score
):
    mode_names = {
        1: "BEST",
        2: "ALL ANCHORS",
        3: "HEATMAP"
    }

    cv2.putText(
        image,
        f"MODE: {mode_names[mode]}",
        (5, 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (0, 255, 0),
        1
    )

    cv2.putText(
        image,
        f"BEST: {best_index}",
        (5, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (0, 255, 0),
        1
    )

    cv2.putText(
        image,
        f"SCORE: {best_score:.3f}",
        (5, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (0, 255, 0),
        1
    )

    return image


# ==========================================
# 初期化
# ==========================================

anchors = generate_anchors()

if len(anchors) != 2016:
    raise RuntimeError(
        f"Anchor数がおかしいです: {len(anchors)}"
    )

print(
    f"Anchor Count: {len(anchors)}"
)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError(
        "カメラを開けませんでした"
    )


# 1 = Best
# 2 = All
# 3 = HeatMap
mode = 1

window_name = "PipeTRT Palm Debugger"

cv2.namedWindow(
    window_name,
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    window_name,
    768,
    768
)


print()
print("=== PipeTRT Palm Debugger ===")
print()
print("1 : Best Anchor")
print("2 : All Anchors")
print("3 : HeatMap")
print("H : Mode Change")
print("Q : Quit")
print()


# ==========================================
# Main Loop
# ==========================================

while True:

    success, frame = camera.read()

    if not success:
        print(
            "フレーム取得に失敗しました"
        )
        break

    # --------------------------------------
    # Preprocess
    # --------------------------------------

    palm_input = preprocess(
        frame
    )

    # --------------------------------------
    # Palm Inference
    # --------------------------------------

    boxes, scores = detect(
        palm_input
    )

    raw_scores = scores.reshape(
        -1
    )

    probabilities = sigmoid(
        raw_scores
    )

    best_index = int(
        np.argmax(probabilities)
    )

    best_score = float(
        probabilities[best_index]
    )

    # --------------------------------------
    # AIが実際に見ている画像
    # --------------------------------------

    display_image = tensor_to_image(
        palm_input
    )

    # --------------------------------------
    # Mode
    # --------------------------------------

    if mode == 1:

        display_image = draw_best_anchor(
            display_image,
            anchors,
            probabilities,
            best_index
        )

    elif mode == 2:

        display_image = draw_all_anchors(
            display_image,
            anchors,
            best_index
        )

    elif mode == 3:

        display_image = draw_heatmap(
            display_image,
            anchors,
            probabilities,
            best_index
        )

    # --------------------------------------
    # Information
    # --------------------------------------

    display_image = draw_information(
        display_image,
        mode,
        best_index,
        best_score
    )

    # 192x192だと小さいので拡大
    display_image = cv2.resize(
        display_image,
        (768, 768),
        interpolation=cv2.INTER_NEAREST
    )

    cv2.imshow(
        window_name,
        display_image
    )

    # --------------------------------------
    # Keyboard
    # --------------------------------------

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == 27:
        break

    elif key == ord("1"):
        mode = 1

    elif key == ord("2"):
        mode = 2

    elif key == ord("3"):
        mode = 3

    elif key == ord("h"):

        mode += 1

        if mode > 3:
            mode = 1


camera.release()
cv2.destroyAllWindows()