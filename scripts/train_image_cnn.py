"""Train a transfer-learning CNN for image authenticity detection (requires PyTorch).

Usage:
  python scripts/train_image_cnn.py --data-dir data/image_authenticity --epochs 10 --batch-size 32

This script expects `data/image_authenticity/ai_generated` and `data/image_authenticity/real` folders.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms, models
except Exception as e:  # pragma: no cover
    print("PyTorch not installed. Install torch and torchvision to run this script.")
    raise


def build_transforms(image_size: int = 224):
    return {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(int(image_size * 1.14)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/image_authenticity")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output", default="artifacts/image_authenticity_cnn.pt")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print("Data directory not found:", data_dir)
        sys.exit(2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = build_transforms(224)

    train_ds = datasets.ImageFolder(data_dir, transform=t["train"])  # expects ai_generated/ and real/ subfolders
    # split into train/val
    n = len(train_ds)
    val_size = max(1, int(0.15 * n))
    train_size = n - val_size
    train_set, val_set = torch.utils.data.random_split(train_ds, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = models.efficientnet_b0(pretrained=True)
    # replace classifier
    in_features = model.classifier[1].in_features if hasattr(model, "classifier") else model.classifier.in_features
    model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, 2))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        correct = 0
        total = 0
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            running += loss.item() * labels.size(0)

        train_acc = correct / max(total, 1)

        # val
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / max(total, 1)
        print(f"Epoch {epoch+1}/{args.epochs} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model_state_dict": model.state_dict(), "classes": train_ds.classes}, args.output)

    print("Finished. Best val acc:", best_acc)


if __name__ == "__main__":
    main()
