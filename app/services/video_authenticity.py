"""Video authenticity detection service.

Provides `VideoAuthenticityDetector` which extracts frames, computes CNN embeddings,
and runs a temporal aggregator (LSTM) if a trained model is available. Falls back to
an embedding-averaging heuristic when no model is present.

This module is intentionally conservative: it requires PyTorch for CNN paths but
will gracefully fall back without it.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from app.core.config import settings


@dataclass
class VideoAuthenticityPrediction:
    label: str
    confidence: float
    frame_scores: List[float]
    suspicious_frames: List[int]


class VideoAuthenticityDetector:
    def __init__(self):
        self.device = None
        self.cnn = None
        self.temporal_model = None
        self.model = None
        self.model_audio_enabled = False
        self.model_name = None
        self._transform = None
        self._load_optional_models()

    def _load_optional_models(self):
        # try loading PyTorch and any saved model artifacts
        try:
            import torch
            import torch.nn as nn
            from torchvision import models
            from torchvision import transforms

            from scripts.train_video_multimodal import build_model as build_video_model

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

            candidate_paths = [
                Path(settings.video_auth_model_path),
                Path("artifacts/video_multimodal.pt"),
                Path("artifacts/video_multimodal_audio.pt"),
                Path(settings.video_auth_cnn_path),
            ]

            for candidate in candidate_paths:
                if not candidate.exists():
                    continue
                try:
                    payload = torch.load(str(candidate), map_location=self.device)
                    state_dict = payload.get("model_state", payload) if isinstance(payload, dict) else payload
                    audio_enabled = any(key.startswith("audio_fc.") for key in state_dict.keys())
                    model = build_video_model(self.device, audio=audio_enabled)
                    model.load_state_dict(state_dict, strict=False)
                    model = model.to(self.device).eval()
                    self.model = model
                    self.model_audio_enabled = audio_enabled
                    self.model_name = candidate.name
                    break
                except Exception:
                    continue

            # base CNN backbone for embeddings
            backbone = models.efficientnet_b0(pretrained=True)
            # remove classifier

            backbone.classifier = nn.Identity()
            backbone = backbone.to(self.device).eval()
            self.cnn = backbone

            # load temporal model if present
            cnn_path = Path(settings.video_auth_cnn_path)
            if cnn_path.exists():
                payload = torch.load(str(cnn_path), map_location=self.device)
                # payload expected to be dict with 'temporal_state' and model meta
                # build a small LSTM classifier compatible with saved state
                hidden = payload.get('hidden_size', 256)
                lstm = nn.LSTM(input_size=backbone.classifier.in_features if hasattr(backbone, 'classifier') else 1280,
                               hidden_size=hidden, batch_first=True, bidirectional=False)
                classifier = nn.Sequential(nn.Linear(hidden, 2))
                # attach
                self.temporal_model = (lstm, classifier)
                try:
                    lstm.load_state_dict(payload.get('lstm_state', {}))
                    classifier.load_state_dict(payload.get('classifier_state', {}))
                    lstm = lstm.to(self.device).eval()
                    classifier = classifier.to(self.device).eval()
                    self.temporal_model = (lstm, classifier)
                except Exception:
                    # ignore load errors; keep None
                    self.temporal_model = None
        except Exception:
            self.device = None
            self.cnn = None
            self.temporal_model = None
            self.model = None
            self.model_audio_enabled = False
            self.model_name = None
            self._transform = None

    def _extract_frames(self, video_path: str | bytes, sample_rate: int | None = None) -> List[np.ndarray]:
        """Extract RGB frames from video. `sample_rate` defines sampling every N-th frame."""
        sample_rate = sample_rate or settings.video_frame_sample_rate
        frames: List[np.ndarray] = []
        if isinstance(video_path, (bytes, bytearray)):
            # OpenCV expects a filesystem path, so persist to a temporary file first.
            suffix = ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(video_path)
                tmp_path = Path(tmp_file.name)
            cap = cv2.VideoCapture(str(tmp_path))
            remove_tmp = True
        else:
            cap = cv2.VideoCapture(str(video_path))
            remove_tmp = False

        if not cap.isOpened():
            return frames

        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % sample_rate == 0:
                # convert BGR->RGB and resize small side to 256 keeping aspect
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]
                if min(h, w) > 256:
                    scale = 256.0 / min(h, w)
                    rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)))
                frames.append(rgb)
            idx += 1

        cap.release()
        if remove_tmp and tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
        return frames

    def _frames_to_embeddings(self, frames: List[np.ndarray]) -> np.ndarray:
        """Compute CNN embeddings for frames; falls back to simple color histograms if CNN missing."""
        if self.cnn is None:
            # fallback: per-frame color histograms as features
            feats = []
            for f in frames:
                hist = cv2.calcHist([f], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
                feats.append(hist.flatten())
            return np.array(feats)

        # use torch to compute embeddings
        import torch
        from torchvision import transforms
        t = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        imgs = torch.stack([t(f) for f in frames]).to(self.device)
        with torch.no_grad():
            emb = self.cnn(imgs)
            if isinstance(emb, torch.Tensor):
                emb = emb.cpu().numpy()
        return np.asarray(emb)

    @staticmethod
    def _motion_features(frames: List[np.ndarray]) -> tuple[float, float, float]:
        """Return conservative motion features from a frame sequence.

        The fallback classifier should only flag a video when there are both strong
        frame-to-frame changes and abrupt spikes. Normal motion is common in real
        videos, so we bias toward low false positives.
        """
        if len(frames) < 2:
            return 0.0, 0.0, 0.0

        diffs: List[float] = []
        hist_corrs: List[float] = []
        prev_gray = None
        for frame in frames:
            small = cv2.resize(frame, (96, 96), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
            if prev_gray is not None:
                diff = cv2.absdiff(prev_gray, gray)
                diffs.append(float(diff.mean()))
                hist_prev = cv2.calcHist([prev_gray], [0], None, [32], [0, 256])
                hist_curr = cv2.calcHist([gray], [0], None, [32], [0, 256])
                cv2.normalize(hist_prev, hist_prev)
                cv2.normalize(hist_curr, hist_curr)
                hist_corrs.append(float(cv2.compareHist(hist_prev, hist_curr, cv2.HISTCMP_CORREL)))
            prev_gray = gray

        if not diffs:
            return 0.0, 0.0, 0.0

        diffs_arr = np.asarray(diffs, dtype=np.float32)
        median_diff = float(np.median(diffs_arr))
        mad = float(np.median(np.abs(diffs_arr - median_diff))) + 1e-6
        spike_ratio = float(np.mean(diffs_arr > (median_diff + 2.5 * mad)))
        hist_corr = float(np.clip(np.mean(hist_corrs), -1.0, 1.0)) if hist_corrs else 1.0
        return median_diff, spike_ratio, hist_corr

    @staticmethod
    def _fallback_fake_score(median_diff: float, spike_ratio: float, hist_corr: float) -> float:
        """Map motion features to a conservative fake probability.

        Real videos often have smooth motion, scene changes, and compression noise.
        We only raise the fake score when the video is both choppy and inconsistent.
        """
        motion_component = np.clip((median_diff - 6.0) / 16.0, 0.0, 1.0)
        spike_component = np.clip(spike_ratio * 2.5, 0.0, 1.0)
        hist_component = np.clip((1.0 - max(hist_corr, 0.0)) * 1.3, 0.0, 1.0)
        fake_score = 0.55 * motion_component + 0.30 * spike_component + 0.15 * hist_component
        return float(np.clip(fake_score, 0.0, 1.0))

    def predict_video(self, video_bytes: bytes | str, max_frames: int = 64) -> VideoAuthenticityPrediction:
        frames = self._extract_frames(video_bytes)
        if len(frames) == 0:
            return VideoAuthenticityPrediction(label="Unknown", confidence=0.5, frame_scores=[], suspicious_frames=[])

        if len(frames) < 4:
            # Too few frames for a stable verdict; keep the model conservative.
            return VideoAuthenticityPrediction(label="Real Video", confidence=0.55, frame_scores=[0.0] * len(frames), suspicious_frames=[])

        # limit frames
        if len(frames) > max_frames:
            # uniform sampling
            idxs = np.linspace(0, len(frames) - 1, max_frames).astype(int)
            frames = [frames[i] for i in idxs]

        embeddings = self._frames_to_embeddings(frames)

        # Prefer the trained multimodal/video checkpoint when present.
        if self.model is not None and self._transform is not None:
            try:
                import torch

                tensor_frames = torch.stack([self._transform(frame) for frame in frames[:max_frames]]).unsqueeze(0).to(self.device)
                audio_tensor = None
                if self.model_audio_enabled:
                    audio_tensor = torch.zeros((1, 13, 10), dtype=torch.float32, device=self.device)

                with torch.no_grad():
                    logits = self.model(tensor_frames, audio_tensor)
                    probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()[0]

                ai_prob = float(probs[1]) if probs.shape[0] > 1 else float(probs.max())
                threshold = float(getattr(settings, "video_model_fake_threshold", 0.45))
                label = "Fake Video" if ai_prob >= threshold else "Real Video"
                confidence = ai_prob if label == "Fake Video" else 1.0 - ai_prob
                frame_scores = [ai_prob for _ in range(len(frames))]
                suspicious = list(range(len(frames))) if label == "Fake Video" and ai_prob >= max(threshold, 0.55) else []
                return VideoAuthenticityPrediction(
                    label=label,
                    confidence=float(np.clip(confidence, 0.02, 0.98)),
                    frame_scores=frame_scores,
                    suspicious_frames=suspicious,
                )
            except Exception:
                # fall through to conservative heuristic
                pass

        # temporal model path
        if self.temporal_model is not None:
            try:
                import torch
                lstm, classifier = self.temporal_model
                x = torch.from_numpy(embeddings).unsqueeze(0).float().to(self.device)
                with torch.no_grad():
                    out, _ = lstm(x)
                    last = out[:, -1, :]
                    logits = classifier(last)
                    probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()[0]
                    ai_prob = float(probs[1]) if probs.shape[0] > 1 else float(probs.max())
                    frame_scores = [float(p) for p in probs.repeat(len(frames), 0)[:len(frames)]]
                    label = "Fake Video" if ai_prob > 0.5 else "Real Video"
                    suspicious = [i for i, s in enumerate(frame_scores) if s > 0.6]
                    return VideoAuthenticityPrediction(label=label, confidence=ai_prob, frame_scores=frame_scores, suspicious_frames=suspicious)
            except Exception:
                pass

        # fallback heuristic: compute similarity between consecutive embeddings; abrupt changes may indicate manipulation
        try:
            median_diff, spike_ratio, hist_corr = self._motion_features(frames)
            fake_score = self._fallback_fake_score(median_diff, spike_ratio, hist_corr)
            threshold = float(getattr(settings, "video_fake_threshold", 0.66))
            label = "Fake Video" if fake_score >= threshold else "Real Video"
            confidence = fake_score if label == "Fake Video" else 1.0 - fake_score

            # Use the per-frame motion deltas for the UI timeline.
            frame_scores = [0.0]
            prev_small = cv2.resize(frames[0], (96, 96), interpolation=cv2.INTER_AREA)
            prev_gray = cv2.cvtColor(prev_small, cv2.COLOR_RGB2GRAY)
            for frame in frames[1:]:
                small = cv2.resize(frame, (96, 96), interpolation=cv2.INTER_AREA)
                gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
                frame_scores.append(float(cv2.absdiff(prev_gray, gray).mean()))
                prev_gray = gray

            frame_scores_arr = np.asarray(frame_scores, dtype=np.float32)
            if label == "Fake Video" and len(frame_scores_arr) > 1:
                suspicious_cutoff = float(np.percentile(frame_scores_arr[1:], 85))
                suspicious = [int(i) for i, value in enumerate(frame_scores_arr) if value >= suspicious_cutoff and i > 0]
            else:
                suspicious = []
            return VideoAuthenticityPrediction(label=label, confidence=float(np.clip(confidence, 0.02, 0.98)), frame_scores=frame_scores_arr.tolist(), suspicious_frames=suspicious)
        except Exception:
            return VideoAuthenticityPrediction(label="Real Video", confidence=0.5, frame_scores=[], suspicious_frames=[])


video_authenticity_singleton = VideoAuthenticityDetector()
