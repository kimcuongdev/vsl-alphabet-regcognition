# src/models/resnet.py
import torch.nn as nn
from torchvision import models


def get_resnet18(num_classes: int, pretrained: bool = True):
    if pretrained:
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    else:
        model = models.resnet18(weights=None)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)  # layer phân loại mới

    # Fine-tune toàn bộ -> đảm bảo tất cả params đều trainable
    for param in model.parameters():
        param.requires_grad = True

    return model
