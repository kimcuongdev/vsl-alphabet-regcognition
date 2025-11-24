import cv2


def draw_result(frame, bbox, pred_class, pred_conf):
    """Vẽ bounding box và nhãn dự đoán lên frame

    Args:
        frame: ảnh BGR
        bbox: BoundingBox
        pred_class: tên lớp dự đoán
        pred_conf: độ tin cậy dự đoán
    """
    if bbox is not None:
        x_min, y_min, x_max, y_max = bbox.get_ltrb()
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        label = f"{pred_class}: {pred_conf:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        label_y_min = max(y_min, label_size[1] + 10)
        cv2.rectangle(
            frame,
            (x_min, label_y_min - label_size[1] - 10),
            (x_min + label_size[0], label_y_min),
            (0, 255, 0),
            cv2.FILLED,
        )
        cv2.putText(
            frame,
            label,
            (x_min, label_y_min - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
        )
    return frame
