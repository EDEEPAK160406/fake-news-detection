"""Train a multimodal video authenticity model (visual + optional audio).

This is a training scaffold optimized for transfer learning. It expects a processed
dataset folder structure created by `prepare_video_dataset.py`:

data/video_authenticity/processed/{real,ai_generated}/{video_id}/frame_0000.jpg ...
and optional audio_mfcc.npy per video folder.

This script is a practical starting point; adjust batch sizes, augmentations and
hyperparameters for your environment (GPU recommended).
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
    from torchvision import transforms, models
    import numpy as np
    from PIL import Image
except Exception:
    print('Missing deep-learning dependencies. Install torch and torchvision.')
    raise


class ProcessedVideoDataset(Dataset):
    def __init__(self, root: str, seq_len: int = 16, transform=None, audio: bool = False):
        self.root = Path(root)
        self.items = []  # (video_folder, label)
        self.seq_len = seq_len
        self.transform = transform
        self.audio = audio
        for label_idx, label in enumerate(['real', 'ai_generated']):
            base = Path(root) / label
            if not base.exists():
                continue
            for vid in base.iterdir():
                if vid.is_dir():
                    self.items.append((vid, label_idx))

    def __len__(self):
        return len(self.items)

    def _load_frames(self, folder: Path):
        files = sorted(folder.glob('frame_*.jpg'))
        if len(files) == 0:
            # fallback zeros
            return [Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))] * self.seq_len
        # sample uniformly
        if len(files) < self.seq_len:
            files = files + files[: (self.seq_len - len(files))]
        idxs = np.linspace(0, len(files) - 1, self.seq_len).astype(int)
        imgs = [Image.open(files[i]).convert('RGB') for i in idxs]
        return imgs

    def __getitem__(self, idx):
        folder, label = self.items[idx]
        imgs = self._load_frames(folder)
        if self.transform:
            imgs = [self.transform(img) for img in imgs]
        seq = torch.stack(imgs)  # T,C,H,W
        audio_feat = None
        if self.audio:
            audio_path = folder / 'audio_mfcc.npy'
            if audio_path.exists():
                audio_feat = np.load(audio_path)
                audio_feat = torch.from_numpy(audio_feat).float()
        return seq, audio_feat, torch.tensor(label, dtype=torch.long)


def build_model(device, audio=False):
    try:
        backbone = models.efficientnet_b0(pretrained=True)
    except Exception:
        print('Could not load pretrained weights, falling back to pretrained=False')
        backbone = models.efficientnet_b0(pretrained=False)
    import torch.nn as nn
    feat_dim = 1280
    backbone.classifier = nn.Identity()
    backbone = backbone.to(device).eval()

    class TemporalClassifier(nn.Module):
        def __init__(self, feat_dim, hidden=256, audio=False):
            super().__init__()
            self.lstm = nn.LSTM(input_size=feat_dim, hidden_size=hidden, batch_first=True)
            if audio:
                self.audio_fc = nn.Linear(13 * 10, hidden)  # approximate
                self.classifier = nn.Linear(hidden * 2, 2)
            else:
                self.classifier = nn.Linear(hidden, 2)

        def forward(self, x, audio_feat=None):
            # x: B,T,C,H,W
            B, T, C, H, W = x.shape
            # collapse time dimension by feeding per-frame features
            x = x.view(B * T, C, H, W)
            with torch.no_grad():
                feats = backbone(x)
            feats = feats.view(B, T, -1).float()
            out, _ = self.lstm(feats)
            last = out[:, -1, :]
            if audio_feat is not None and hasattr(self, 'audio_fc'):
                a = audio_feat.view(B, -1)
                a = torch.relu(self.audio_fc(a))
                comb = torch.cat([last, a], dim=1)
                return self.classifier(comb)
            return self.classifier(last)

    model = TemporalClassifier(feat_dim, hidden=256, audio=audio).to(device)
    return model


def collate_fn(batch):
    seqs = [b[0] for b in batch]
    auds = [b[1] for b in batch]
    labels = torch.stack([b[2] for b in batch])
    seqs = torch.stack(seqs)  # B,T,C,H,W
    # pad audio to same shape or None
    return seqs, auds, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data/video_authenticity/processed')
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--audio', action='store_true')
    parser.add_argument('--output', default='artifacts/video_multimodal.pt')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    ds = ProcessedVideoDataset(args.data_dir, seq_len=8, transform=t, audio=args.audio)
    if len(ds) == 0:
        print('No processed videos found under', args.data_dir)
        sys.exit(2)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2)

    model = build_model(device, audio=args.audio)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        correct = 0; total = 0
        for seqs, auds, labels in loader:
            seqs = seqs.to(device)
            labels = labels.to(device)
            aud_tensor = None
            if args.audio:
                # convert list of arrays to tensor, fill missing with zeros
                processed = []
                for a in auds:
                    if a is None:
                        processed.append(torch.zeros((13, 10)))
                    elif isinstance(a, torch.Tensor):
                        processed.append(a)
                    else:
                        processed.append(torch.from_numpy(a))
                aud_tensor = torch.stack(processed).float().to(device)
            optimizer.zero_grad()
            logits = model(seqs, aud_tensor)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        acc = correct / max(total, 1)
        print(f'Epoch {epoch+1}/{args.epochs} acc={acc:.4f}')
        if acc > best_acc:
            best_acc = acc
            torch.save({'model_state': model.state_dict()}, args.output)

    print('Finished. Best acc:', best_acc)


if __name__ == '__main__':
    main()
