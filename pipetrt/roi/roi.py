def create_axis_aligned_roi(palm_result, scale=1.5):
    x_min, y_min, x_max, y_max = palm_result["bbox"]

    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2

    width = x_max - x_min
    height = y_max - y_min

    roi_size = max(width, height) * scale

    roi = {
        "center_x": center_x,
        "center_y": center_y,
        "width": roi_size,
        "height": roi_size,
        "rotation": 0.0
    }

    return roi