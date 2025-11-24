# src/inference/load_model.py
import torch

from src.models.resnet import get_resnet18


def load_trained_resnet18(model_path: str, num_classes: int, device: str = "cuda"):
    model = get_resnet18(num_classes=num_classes, pretrained=False)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()  # rất quan trọng cho inference
    return model
