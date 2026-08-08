import cv2
import numpy as np

from pipetrt.palm.preprocess import preprocess
from pipetrt.palm.detector import detect


def generate_anchors():
    anchors = []

    # stride 8
    # 192 / 8 = 24
    # 24 × 24 × 2 = 1152 anchors
    for y in range(24):
        for x in range(24):
            center_x = (x + 0.5) / 24
            center_y = (y + 0.5) / 24

            anchors.append([center_x, center_y])
            anchors.append([center_x, center_y])

    # stride 16
    # 192 / 16 = 12
    # 12 × 12 × 6 = 864 anchors
    for y in range(12):
        for x in range(12):
            center_x = (x + 0.5) / 12
            center_y = (y + 0.5) / 12

            for count in range(6):
                anchors.append([center_x, center_y])

    return np.array(anchors, dtype=np.float32)


# ============================================
# 画像・モデル準備
# ============================================

frame = cv2.imread("example/data/hand2.jpg")

if frame is None:
    raise FileNotFoundError(
        "example/images/hand.jpg を読み込めませんでした"
    )

palm_input = preprocess(frame)

boxes, scores = detect(palm_input)

anchors = generate_anchors()

print("Anchor Count:", len(anchors))

if len(anchors) != 2016:
    raise RuntimeError(
        f"Anchor数が2016ではありません: {len(anchors)}"
    )


# ============================================
# モデル入力画像を表示用画像へ戻す
# ============================================

display_image = palm_input[0]

# 0〜1 → 0〜255
display_image = (
    display_image * 255.0
).clip(0, 255).astype(np.uint8)

# RGB → BGR
display_image = cv2.cvtColor(
    display_image,
    cv2.COLOR_RGB2BGR
)


# ============================================
# Score
# ============================================

flat_scores = scores.reshape(-1)

best_index = int(np.argmax(flat_scores))

selected_index = best_index


def sigmoid(value):
    return 1.0 / (1.0 + np.exp(-value))


# ============================================
# 描画
# ============================================

def draw_anchor():
    image = display_image.copy()

    anchor = anchors[selected_index]

    center_x = anchor[0]
    center_y = anchor[1]

    pixel_x = int(center_x * 192)
    pixel_y = int(center_y * 192)

    raw_score = flat_scores[selected_index]
    probability = sigmoid(raw_score)

    # 選択Anchor
    cv2.circle(
        image,
        (pixel_x, pixel_y),
        5,
        (0, 0, 255),
        -1
    )

    # 十字線
    cv2.line(
        image,
        (pixel_x - 8, pixel_y),
        (pixel_x + 8, pixel_y),
        (0, 0, 255),
        1
    )

    cv2.line(
        image,
        (pixel_x, pixel_y - 8),
        (pixel_x, pixel_y + 8),
        (0, 0, 255),
        1
    )

    # 情報表示
    cv2.putText(
        image,
        f"Anchor: {selected_index}",
        (5, 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (0, 255, 0),
        1
    )

    cv2.putText(
        image,
        f"Center: ({center_x:.3f}, {center_y:.3f})",
        (5, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (0, 255, 0),
        1
    )

    cv2.putText(
        image,
        f"Score: {probability:.4f}",
        (5, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (0, 255, 0),
        1
    )

    if selected_index == best_index:
        cv2.putText(
            image,
            "BEST",
            (5, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2
        )

    # 見やすいように拡大
    image = cv2.resize(
        image,
        (576, 576),
        interpolation=cv2.INTER_NEAREST
    )

    cv2.imshow(
        "PipeTRT Anchor Viewer",
        image
    )


# ============================================
# マウス操作
# ============================================

def mouse_callback(event, x, y, flags, param):
    global selected_index

    if event == cv2.EVENT_MOUSEWHEEL:

        if flags > 0:
            selected_index += 1
        else:
            selected_index -= 1

        selected_index = max(
            0,
            min(selected_index, len(anchors) - 1)
        )

        draw_anchor()


# ============================================
# Viewer
# ============================================

window_name = "PipeTRT Anchor Viewer"

cv2.namedWindow(window_name)

cv2.setMouseCallback(
    window_name,
    mouse_callback
)

print()
print("=== PipeTRT Anchor Viewer ===")
print("Mouse Wheel : Anchor切り替え")
print("A / Left    : 前のAnchor")
print("D / Right   : 次のAnchor")
print("B           : Best Anchorへ移動")
print("Q / ESC     : 終了")
print()
print("Best Anchor:", best_index)
print(
    "Best Score :",
    sigmoid(flat_scores[best_index])
)

draw_anchor()

while True:

    key = cv2.waitKey(20) & 0xFF

    if key == ord("q") or key == 27:
        break

    elif key == ord("a"):
        selected_index -= 1

    elif key == ord("d"):
        selected_index += 1

    elif key == ord("b"):
        selected_index = best_index

    selected_index = max(
        0,
        min(selected_index, len(anchors) - 1)
    )

    draw_anchor()


cv2.destroyAllWindows()