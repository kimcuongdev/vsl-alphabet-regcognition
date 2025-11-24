import os
from pathlib import Path

import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


class SignLanguageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.samples = []
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        for cls in self.classes:
            cls_path = os.path.join(root_dir, cls)
            for img_name in os.listdir(cls_path):
                self.samples.append(
                    (os.path.join(cls_path, img_name), self.class_to_idx[cls])
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def get_transforms():
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    train_transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    test_transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, test_transform


def get_dataloaders(data_root, batch_size=32):
    """
    data_root: đường dẫn đến thư mục chứa train, val, test
    batch_size: kích thước batch
    """
    train_transform, test_transform = get_transforms()

    train_ds = SignLanguageDataset(
        os.path.join(data_root, "train"), transform=train_transform
    )
    val_ds = SignLanguageDataset(
        os.path.join(data_root, "valid"), transform=test_transform
    )
    test_ds = SignLanguageDataset(
        os.path.join(data_root, "test"), transform=test_transform
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2
    )

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return train_loader, val_loader, test_loader, len(train_ds.classes)


def get_class_names(data_root: str):
    train_ds = datasets.ImageFolder(Path(data_root) / "train")
    return train_ds.classes  # ví dụ ['A', 'B', ..., 'mũ', 'Râu', ...]


CLASS_NAMES = get_class_names(
    data_root=r"data\keypoint_img\grayscale\vsl_224_subset_split"
)
NUM_CLASSES = len(CLASS_NAMES)
