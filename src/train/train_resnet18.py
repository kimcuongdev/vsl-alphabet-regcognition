# train_resnet18.py
import torch
import torch.nn as nn
import torch.optim as optim

# from src.models.resnet import get_resnet18
# from src.datasets.dataloader import get_dataloaders
# from src.training.engine import train_one_epoch, evaluate

DATA_ROOT = "/kaggle/working/vsl_split"
EPOCHS = 15
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():

    train_loader, val_loader, test_loader, num_classes = get_dataloaders(DATA_ROOT)

    # model
    model = get_resnet18(num_classes=num_classes, pretrained=True)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    # Adam theo slide — learning_rate thấp vì finetune
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # StepLR — giảm LR theo lịch cố định (ResNet dùng)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    best_state = None
    best_val_acc = 0

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train: loss={train_loss:.4f}, acc={train_acc:.4f} | "
            f"Val: loss={val_loss:.4f}, acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()

    # Load best model
    model.load_state_dict(best_state)

    # Test
    test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)
    print("Test:", test_loss, test_acc)

    torch.save(model.state_dict(), "resnet18_full_finetune.pth")


if __name__ == "__main__":
    main()
