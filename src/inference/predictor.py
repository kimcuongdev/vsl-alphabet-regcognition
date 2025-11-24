# src/inference/predict.py
import cv2
import torch
from PIL import Image
from torchvision import transforms

from src.inference.transforms import infer_tfms


@torch.no_grad()
def predict_image(image_path: str, model, device: str, class_names: list[str]):
    # 1. Đọc ảnh
    img = Image.open(image_path).convert("RGB")

    # 2. Transform
    tensor = infer_tfms(img).unsqueeze(0).to(device)  # [1, 3, 224, 224]

    # 3. Forward
    logits = model(tensor)  # [1, num_classes]
    probs = torch.softmax(logits, dim=1)

    top_prob, top_idx = probs.max(dim=1)
    pred_class = class_names[top_idx.item()]
    pred_conf = top_prob.item()

    return pred_class, pred_conf, probs.squeeze().cpu().numpy()


@torch.no_grad()
def predict_from_opencv_frame(frame, model, device, class_names):
    # BGR -> RGB
    input_channels = frame.shape[2]
    # print(f"predict_from_opencv_frame frame_bgr shape: {frame.shape}")
    if input_channels == 3:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    img = Image.fromarray(frame_rgb)

    tensor = infer_tfms(img).unsqueeze(0).to(device)

    with torch.no_grad():
        # print("predicting")
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        top_prob, top_idx = probs.max(dim=1)

    return class_names[top_idx.item()], top_prob.item()
