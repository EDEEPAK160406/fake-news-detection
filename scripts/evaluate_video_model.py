"""Evaluate a trained video authenticity model with metrics and threshold tuning."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import sys

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_video_multimodal import ProcessedVideoDataset, build_model


def collect_probs(model, dataset, device, audio_enabled: bool) -> Tuple[List[int], List[float]]:
    labels: List[int] = []
    probs: List[float] = []
    for idx in range(len(dataset)):
        seq, audio_feat, label = dataset[idx]
        if not isinstance(seq, torch.Tensor):
            raise TypeError("Dataset must be created with transforms so frames become tensors")
        audio_tensor = None
        if audio_feat is not None:
            audio_tensor = torch.from_numpy(audio_feat).float().unsqueeze(0).to(device)
        elif audio_enabled:
            audio_tensor = torch.zeros((1, 13, 10), dtype=torch.float32, device=device)
        with torch.no_grad():
            logits = model(seq.unsqueeze(0).to(device), audio_tensor)
            prob = torch.softmax(logits, dim=1)[0, 1].item()
        labels.append(int(label.item()))
        probs.append(float(prob))
    return labels, probs


def metrics_at_threshold(labels: List[int], probs: List[float], threshold: float) -> Dict[str, float]:
    preds = [1 if p >= threshold else 0 for p in probs]
    metrics = {
        "threshold": threshold,
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }
    try:
        metrics["roc_auc"] = roc_auc_score(labels, probs)
    except Exception:
        metrics["roc_auc"] = float("nan")
    return metrics


def tune_threshold(labels: List[int], probs: List[float]) -> Dict[str, float]:
    candidates = np.linspace(0.10, 0.90, 81)
    scored = [metrics_at_threshold(labels, probs, float(t)) for t in candidates]
    # Favor recall a bit more to reduce fake-vs-real false negatives, then F1.
    scored.sort(key=lambda m: (m["recall"], m["f1"], m["precision"]), reverse=True)
    return scored[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/video_authenticity/processed_audio")
    parser.add_argument("--model", default="artifacts/video_multimodal_audio.pt")
    parser.add_argument("--audio", action="store_true")
    parser.add_argument("--output", default="artifacts/video_eval_metrics.json")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(str(model_path), map_location=device)
    state_dict = payload.get("model_state", payload) if isinstance(payload, dict) else payload
    audio_enabled = args.audio or any(k.startswith("audio_fc.") for k in state_dict.keys())

    model = build_model(device, audio=audio_enabled)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dataset = ProcessedVideoDataset(args.data_dir, seq_len=8, transform=transform, audio=audio_enabled)
    if len(dataset) == 0:
        raise SystemExit(f"No processed samples found under {args.data_dir}")

    labels, probs = collect_probs(model, dataset, device, audio_enabled)

    # split by label for a light validation/test style report
    y = np.asarray(labels)
    p = np.asarray(probs)
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, test_size=0.25, random_state=42, stratify=y)
    val_idx, test_idx = train_test_split(test_idx, test_size=0.5, random_state=42, stratify=y[test_idx])

    val_labels = y[val_idx].tolist()
    val_probs = p[val_idx].tolist()
    test_labels = y[test_idx].tolist()
    test_probs = p[test_idx].tolist()

    tuned = tune_threshold(val_labels, val_probs)
    test_metrics = metrics_at_threshold(test_labels, test_probs, tuned["threshold"])
    cm = confusion_matrix(test_labels, [1 if x >= tuned["threshold"] else 0 for x in test_probs]).tolist()

    report = {
        "dataset_size": len(dataset),
        "audio_enabled": audio_enabled,
        "validation_best_threshold": tuned["threshold"],
        "validation_metrics": tuned,
        "test_metrics": test_metrics,
        "test_confusion_matrix": cm,
    }

    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
