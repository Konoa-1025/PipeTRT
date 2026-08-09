import math


def normalize_radians(angle):
    while angle < -math.pi:
        angle += 2.0 * math.pi

    while angle >= math.pi:
        angle -= 2.0 * math.pi

    return angle


def create_roi(palm_result, image_width, image_height):
    bbox = palm_result["bbox"]
    keypoints = palm_result["keypoints"]

    x_min, y_min, x_max, y_max = bbox

    # Palm bbox
    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0

    width = x_max - x_min
    height = y_max - y_min

    # --------------------------------
    # rotation
    # keypoint 0 : wrist center
    # keypoint 2 : middle finger MCP
    # --------------------------------

    wrist = keypoints[0]
    middle_mcp = keypoints[2]

    x0 = wrist[0] * image_width
    y0 = wrist[1] * image_height

    x1 = middle_mcp[0] * image_width
    y1 = middle_mcp[1] * image_height

    target_angle = math.pi / 2.0

    rotation = target_angle - math.atan2(
        -(y1 - y0),
        x1 - x0
    )

    rotation = normalize_radians(rotation)

    # --------------------------------
    # MediaPipe RectTransformation
    # --------------------------------

    shift_x = 0.0
    shift_y = -0.5

    scale_x = 2.6
    scale_y = 2.6

    # shiftは回転したROI座標系で行う
    x_shift = (
        image_width
        * width
        * shift_x
        * math.cos(rotation)
        -
        image_height
        * height
        * shift_y
        * math.sin(rotation)
    ) / image_width

    y_shift = (
        image_width
        * width
        * shift_x
        * math.sin(rotation)
        +
        image_height
        * height
        * shift_y
        * math.cos(rotation)
    ) / image_height

    center_x += x_shift
    center_y += y_shift

    # --------------------------------
    # square_long
    # --------------------------------

    long_side = max(
        width * image_width,
        height * image_height
    )

    width = long_side / image_width
    height = long_side / image_height

    # MediaPipe scale
    width *= scale_x
    height *= scale_y

    return {
        "center_x": float(center_x),
        "center_y": float(center_y),
        "width": float(width),
        "height": float(height),
        "rotation": float(rotation)
    }