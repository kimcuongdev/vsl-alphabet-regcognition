# app.py
import cv2
import numpy as np
import torch

from src.dataset.skeleton_dataset import CLASS_NAMES, NUM_CLASSES
from src.inference.load_model import load_trained_resnet18
from src.inference.predictor import predict_from_opencv_frame
from src.keypoints_extractor.extractor import HandSkeletonExtractor
from src.utils import draw_result


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # 1. Load model đã fine-tune
    model = load_trained_resnet18(
        model_path="saved_models/resnet18_full_finetune.pth",
        num_classes=NUM_CLASSES,
        device=device,
    )

    # extractor phải tương đồng với extractor khi tạo dataset
    extractor = HandSkeletonExtractor()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("no camera")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            print("failed to capture frame")
            break

        keypoints_img, _, _ = extractor.process_frame(
            frame,
            inference_mode=True,
            show_annotated=False,
            show_keypoints_image=True,
        )

        _, annotated_frame, bbox = extractor.process_frame(
            cv2.flip(frame, 1),
            inference_mode=True,
            show_annotated=False,
            show_keypoints_image=False,
        )
        # cv2.imshow("Annotated Frame", annotated_frame)
        # print(bbox.get_ltrb() if bbox is not None else None)

        pred_class, pred_conf = None, None
        if keypoints_img is not None:
            pred_class, pred_conf = predict_from_opencv_frame(
                keypoints_img, model, device, CLASS_NAMES
            )
            print(f"Predicted: {pred_class} ({pred_conf:.4f})")
            annotated_frame = draw_result(annotated_frame, bbox, pred_class, pred_conf)
        cv2.imshow("Result", annotated_frame)

        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break


if __name__ == "__main__":
    main()
