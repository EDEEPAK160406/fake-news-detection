"""Train a hybrid CNN+LSTM video authenticity model.

Usage:
  python scripts/train_video_cnn.py --data-dir data/video_authenticity --epochs 6 --batch-size 4

Expects folders: data/video_authenticity/ai_generated and /real with video files.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from torchvision import models, transforms
    import cv2
    import numpy as np
except Exception as e:  # pragma: no cover
    print("Missing dependencies. Ensure torch, torchvision and opencv-python are installed.")
    raise


class VideoFolderDataset(Dataset):
    def __init__(self, root: str, sample_frames: int = 16, transform=None):
        self.root = Path(root)
        self.samples = []
        self.transform = transform
        self.sample_frames = sample_frames
        for lbl, name in enumerate(["real", "ai_generated"]):
            p = self.root / name
            if not p.exists():
                continue
            for f in p.iterdir():
                if f.suffix.lower() in {'.mp4', '.mov', '.avi', '.mkv'}:
                    self.samples.append((str(f), lbl))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, lbl = self.samples[idx]
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            frames.append(rgb)
        cap.release()
        if len(frames) == 0:
            # return zeros
            seq = [np.zeros((224, 224, 3), dtype=np.uint8)] * self.sample_frames
        else:
            # sample uniformly
            idxs = np.linspace(0, len(frames) - 1, self.sample_frames).astype(int)
            seq = [frames[i] for i in idxs]

        if self.transform:
            seq = [self.transform(f) for f in seq]
        seq = torch.stack(seq)  # T,C,H,W
        return seq, int(lbl)


def build_transforms():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/video_authenticity")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output", default="artifacts/video_authenticity_cnn.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = build_transforms()

    ds = VideoFolderDataset(args.data_dir, sample_frames=16, transform=t)
    if len(ds) == 0:
        print("No video files found under", args.data_dir)
        sys.exit(2)

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2)

    backbone = models.efficientnet_b0(pretrained=True)
    # replace classifier -> we'll use backbone features
    import torch.nn as nn
    feat_dim = 1280  # efficientnet_b0 feature dim
    backbone.classifier = nn.Identity()
    backbone = backbone.to(device).eval()

    hidden_size = 256
    lstm = nn.LSTM(input_size=feat_dim, hidden_size=hidden_size, batch_first=True)
    classifier = nn.Linear(hidden_size, 2)
    lstm = lstm.to(device)
    classifier = classifier.to(device)

    params = list(lstm.parameters()) + list(classifier.parameters())
    optimizer = optim.Adam(params, lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(args.epochs):
        lstm.train(); classifier.train()
        correct = 0; total = 0
        for seqs, labels in loader:
            # seqs: B,T,C,H,W
            B, T, C, H, W = seqs.shape
            seqs = seqs.to(device)
            labels = labels.to(device)
            # compute per-frame features
            feats = []
            with torch.no_grad():
                for t_i in range(T):
                    f = backbone(seqs[:, t_i])
                    feats.append(f)
            feats = torch.stack(feats, dim=1)  # B,T,feat
            optimizer.zero_grad()
            out, _ = lstm(feats)
            last = out[:, -1, :]
            logits = classifier(last)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        acc = correct / max(total, 1)
        print(f"Epoch {epoch+1}/{args.epochs} acc={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save({
                'lstm_state': lstm.state_dict(),
                'classifier_state': classifier.state_dict(),
                'hidden_size': hidden_size,
                'model_name': 'efficientnet_b0+lstm',
            }, args.output)

    print("Finished. Best acc:", best_acc)


if __name__ == '__main__':
    main()
