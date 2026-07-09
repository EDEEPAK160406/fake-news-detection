from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.core.config import settings


@dataclass
class ImageAuthenticityPrediction:
    label: str
    confidence: float
    probabilities: Dict[str, float]
    reasons: List[str]
    suspicious_regions: List[Dict[str, object]]
    overlay_image: Optional[str]
    model_name: str


def _clamp(value: float) -> float:
    return float(min(max(value, 0.02), 0.98))


def _prob_confidence(probability: float) -> float:
    return float(max(probability, 1.0 - probability))


def _jpeg_block_boundary_score(gray: np.ndarray) -> float:
    if gray.size == 0:
        return 0.0
    row_diff = float(np.mean(np.abs(np.diff(gray.astype(float), axis=0)))) if gray.shape[0] > 1 else 0.0
    col_diff = float(np.mean(np.abs(np.diff(gray.astype(float), axis=1)))) if gray.shape[1] > 1 else 0.0
    return (row_diff + col_diff) / 255.0


class ImageAuthenticityDetector:
    def __init__(self, model_path: str | None = None):
        self.model_path = model_path or settings.image_auth_model_path
        self.pipeline: Optional[Pipeline] = None
        self.cnn_model = None
        self.cnn_available = False
        self.cnn_transforms = None
        self.cnn_device = None
        self.model_name = "artifact-feature-transfer-detector"
        self.feature_names = [
            "red_mean",
            "green_mean",
            "blue_mean",
            "red_std",
            "green_std",
            "blue_std",
            "gray_mean",
            "gray_std",
            "saturation_mean",
            "saturation_std",
            "laplacian_var",
            "edge_density",
            "entropy",
            "dct_high_freq_ratio",
            "fft_peak_ratio",
            "block_variance_mean",
            "block_variance_std",
            "noise_residual_mean",
            "noise_residual_std",
            "chroma_variance_gap",
            "jpeg_block_boundary_score",
            "channel_correlation_gap",
        ]
        self._load_if_available()
        # Attempt to load optional CNN if available
        try:
            import torch
            from torchvision import transforms, models

            self.cnn_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.cnn_transforms = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            # load if path exists
            cnn_path = Path(settings.image_auth_cnn_path)
            if cnn_path.exists():
                payload = torch.load(str(cnn_path), map_location=self.cnn_device)
                # build model
                model = models.efficientnet_b0(pretrained=False)
                in_features = model.classifier[1].in_features if hasattr(model, "classifier") else model.classifier.in_features
                import torch.nn as nn

                model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, 2))
                model.load_state_dict(payload.get("model_state_dict", payload))
                model = model.to(self.cnn_device)
                model.eval()
                self.cnn_model = model
                self.cnn_available = True
                self.model_name = str(payload.get("model_name", self.model_name)) if isinstance(payload, dict) else self.model_name
        except Exception:
            # CNN is optional; if imports or files missing, continue with feature-model only
            self.cnn_available = False

    def _load_if_available(self) -> None:
        path = Path(self.model_path)
        if not path.exists():
            return
        try:
            payload = joblib.load(path)
            self.pipeline = payload.get("pipeline") if isinstance(payload, dict) else payload
            self.model_name = str(payload.get("model_name", self.model_name)) if isinstance(payload, dict) else self.model_name
        except Exception:
            self.pipeline = None

    @staticmethod
    def _decode_image(image_bytes: bytes) -> Tuple[np.ndarray, np.ndarray]:
        array = np.frombuffer(image_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Unsupported image format. Please upload a JPG, JPEG, or PNG image.")

        denoised = cv2.fastNlMeansDenoisingColored(bgr, None, 3, 3, 7, 21)
        resized = cv2.resize(denoised, (224, 224), interpolation=cv2.INTER_AREA)
        return resized, cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _entropy(gray: np.ndarray) -> float:
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
        hist = hist / max(hist.sum(), 1.0)
        hist = hist[hist > 0]
        return float(-(hist * np.log2(hist)).sum()) if hist.size else 0.0

    @staticmethod
    def _channel_correlation_gap(image_bgr: np.ndarray) -> float:
        channels = cv2.split(image_bgr.astype(np.float32))
        if len(channels) != 3:
            return 0.0
        correlations = []
        for first, second in ((0, 1), (1, 2), (0, 2)):
            a = channels[first].ravel()
            b = channels[second].ravel()
            if np.std(a) < 1e-6 or np.std(b) < 1e-6:
                correlations.append(0.0)
                continue
            corr = float(np.corrcoef(a, b)[0, 1])
            correlations.append(abs(corr))
        return float(1.0 - np.mean(correlations))

    @staticmethod
    def _fft_peak_ratio(gray: np.ndarray) -> float:
        spectrum = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))
        magnitude = np.log1p(np.abs(spectrum))
        center = magnitude[magnitude.shape[0] // 4 : 3 * magnitude.shape[0] // 4, magnitude.shape[1] // 4 : 3 * magnitude.shape[1] // 4]
        if center.size == 0:
            return 0.0
        peak = float(np.percentile(center, 99))
        mean = float(np.mean(center)) + 1e-6
        return float(min(max((peak - mean) / mean, 0.0), 10.0) / 10.0)

    @staticmethod
    def _dct_high_frequency_ratio(gray: np.ndarray) -> float:
        block = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA).astype(np.float32)
        dct = cv2.dct(block)
        abs_dct = np.abs(dct)
        high_freq = abs_dct[24:, 24:]
        return float(np.sum(high_freq) / max(np.sum(abs_dct), 1e-6))

    @staticmethod
    def _block_variances(gray: np.ndarray, grid: int = 8) -> Tuple[np.ndarray, List[Dict[str, object]]]:
        h, w = gray.shape[:2]
        block_h = max(h // grid, 1)
        block_w = max(w // grid, 1)
        values: List[float] = []
        scores: List[Dict[str, object]] = []

        for row in range(grid):
            for col in range(grid):
                y0 = row * block_h
                x0 = col * block_w
                y1 = h if row == grid - 1 else min((row + 1) * block_h, h)
                x1 = w if col == grid - 1 else min((col + 1) * block_w, w)
                block = gray[y0:y1, x0:x1]
                if block.size == 0:
                    continue
                variance = float(np.var(block))
                values.append(variance)
                scores.append(
                    {
                        "x": int(x0),
                        "y": int(y0),
                        "width": int(max(x1 - x0, 1)),
                        "height": int(max(y1 - y0, 1)),
                        "score": variance,
                    }
                )

        if not values:
            return np.zeros(2, dtype=float), []

        return np.array([float(np.mean(values)), float(np.std(values))], dtype=float), scores

    @staticmethod
    def _noise_residual(gray: np.ndarray) -> np.ndarray:
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        residual = cv2.absdiff(gray, blur)
        return residual.astype(np.float32)

    def _extract_features(self, image_bgr: np.ndarray, gray: np.ndarray) -> Tuple[np.ndarray, Dict[str, float], List[Dict[str, object]]]:
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1].astype(np.float32)
        b, g, r = cv2.split(image_bgr.astype(np.float32))

        gray_mean = float(np.mean(gray))
        gray_std = float(np.std(gray))
        sat_mean = float(np.mean(saturation))
        sat_std = float(np.std(saturation))
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        edges = cv2.Canny(gray, 100, 200)
        edge_density = float(np.count_nonzero(edges)) / float(edges.size)
        entropy = self._entropy(gray)
        dct_ratio = self._dct_high_frequency_ratio(gray)
        fft_ratio = self._fft_peak_ratio(gray)
        block_stats, blocks = self._block_variances(gray)
        noise = self._noise_residual(gray)
        noise_mean = float(np.mean(noise))
        noise_std = float(np.std(noise))
        chroma_gap = float(np.std(saturation) / max(np.std(gray), 1.0))
        jpeg_score = _jpeg_block_boundary_score(gray)
        channel_gap = self._channel_correlation_gap(image_bgr)

        feature_vector = np.array(
            [
                float(np.mean(r)),
                float(np.mean(g)),
                float(np.mean(b)),
                float(np.std(r)),
                float(np.std(g)),
                float(np.std(b)),
                gray_mean,
                gray_std,
                sat_mean,
                sat_std,
                lap_var,
                edge_density,
                entropy,
                dct_ratio,
                fft_ratio,
                float(block_stats[0]),
                float(block_stats[1]),
                noise_mean,
                noise_std,
                chroma_gap,
                jpeg_score,
                channel_gap,
            ],
            dtype=float,
        )

        stats = {
            "laplacian_var": lap_var,
            "edge_density": edge_density,
            "entropy": entropy,
            "dct_high_frequency_ratio": dct_ratio,
            "fft_peak_ratio": fft_ratio,
            "noise_residual_mean": noise_mean,
            "noise_residual_std": noise_std,
            "jpeg_block_boundary_score": jpeg_score,
            "channel_correlation_gap": channel_gap,
        }
        return feature_vector, stats, blocks

    def _heuristic_probability(self, stats: Dict[str, float]) -> float:
        score = 0.42
        if stats["dct_high_frequency_ratio"] > 0.27:
            score += 0.12
        if stats["fft_peak_ratio"] > 0.22:
            score += 0.08
        if stats["jpeg_block_boundary_score"] > 4.4:
            score += 0.08
        if stats["channel_correlation_gap"] > 0.18:
            score += 0.09
        if stats["entropy"] < 5.7:
            score += 0.08
        if stats["laplacian_var"] < 65:
            score += 0.04
        if stats["edge_density"] < 0.03:
            score += 0.05
        if stats["noise_residual_std"] > 22:
            score += 0.05

        return _clamp(score)

    def _predict_probability(self, features: np.ndarray) -> float:
        if self.pipeline is None:
            return 0.5

        proba = self.pipeline.predict_proba(features.reshape(1, -1))[0]
        classes = list(self.pipeline.classes_)
        if "AI_GENERATED" in classes:
            return float(proba[classes.index("AI_GENERATED")])
        if 1 in classes:
            return float(proba[classes.index(1)])
        return float(proba[-1])

    def _predict_cnn_probability(self, image_bytes: bytes) -> float:
        """Return predicted probability for AI_GENERATED from CNN model if available."""
        if not self.cnn_available or self.cnn_model is None:
            return 0.5
        try:
            from PIL import Image
            import torch

            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            inp = self.cnn_transforms(img).unsqueeze(0).to(self.cnn_device)
            with torch.no_grad():
                out = self.cnn_model(inp)
                probs = torch.nn.functional.softmax(out, dim=1)[0]
                # classes order assumed same as training: index 0,1 corresponds to training dataset classes
                # We assume 0==ai_generated or class ordering from training; provide safe mapping by checking trained classes is not implemented here
                ai_prob = float(probs.max().item()) if probs.size(0) == 1 else float(probs[0].item())
                return float(min(max(ai_prob, 0.0), 1.0))
        except Exception:
            return 0.5

    def _build_overlay(self, image_bgr: np.ndarray, blocks: Sequence[Dict[str, object]]) -> Optional[str]:
        if not blocks:
            return None

        scores = np.array([float(block["score"]) for block in blocks], dtype=np.float32)
        if scores.size == 0 or float(scores.max()) <= 0:
            return None

        normalized = (scores - scores.min()) / max(scores.max() - scores.min(), 1e-6)
        heatmap = np.zeros(image_bgr.shape[:2], dtype=np.float32)
        for block, value in zip(blocks, normalized):
            x = int(block["x"])
            y = int(block["y"])
            width = int(block["width"])
            height = int(block["height"])
            heatmap[y : y + height, x : x + width] = np.maximum(heatmap[y : y + height, x : x + width], value)

        heatmap = cv2.GaussianBlur(heatmap, (0, 0), 8)
        colored = cv2.applyColorMap(np.uint8(255 * np.clip(heatmap, 0.0, 1.0)), cv2.COLORMAP_JET)
        blended = cv2.addWeighted(image_bgr, 0.62, colored, 0.38, 0)
        success, encoded = cv2.imencode(".png", blended)
        if not success:
            return None
        return "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")

    def _top_regions(self, blocks: Sequence[Dict[str, object]], limit: int = 4) -> List[Dict[str, object]]:
        if not blocks:
            return []

        ordered = sorted(blocks, key=lambda item: float(item["score"]), reverse=True)[:limit]
        top_score = max(float(block["score"]) for block in ordered) if ordered else 1.0
        regions: List[Dict[str, object]] = []
        for block in ordered:
            regions.append(
                {
                    "x": int(block["x"]),
                    "y": int(block["y"]),
                    "width": int(block["width"]),
                    "height": int(block["height"]),
                    "score": float(block["score"] / max(top_score, 1e-6)),
                }
            )
        return regions

    def predict_single(self, image_bytes: bytes) -> ImageAuthenticityPrediction:
        image_bgr, gray = self._decode_image(image_bytes)
        features, stats, blocks = self._extract_features(image_bgr, gray)

        # classical feature-based model
        model_probability = self._predict_probability(features)
        heuristic_probability = self._heuristic_probability(stats)

        # optional CNN probability
        cnn_probability = self._predict_cnn_probability(image_bytes) if self.cnn_available else None

        if cnn_probability is not None and self.cnn_available:
            # combine CNN and classical signals; prioritize CNN but keep heuristic
            ai_probability = float(min(max(0.72 * cnn_probability + 0.18 * model_probability + 0.10 * heuristic_probability, 0.02), 0.98))
        else:
            if self.pipeline is None:
                ai_probability = heuristic_probability
            else:
                ai_probability = float(min(max(0.68 * model_probability + 0.32 * heuristic_probability, 0.02), 0.98))

        label = "AI Generated" if ai_probability >= 0.5 else "Original"
        confidence = _prob_confidence(ai_probability)
        reasons: List[str] = []

        if stats["dct_high_frequency_ratio"] > 0.27:
            reasons.append("High-frequency texture energy suggests synthetic detail synthesis.")
        if stats["fft_peak_ratio"] > 0.22:
            reasons.append("Frequency-domain peaks indicate a possible GAN fingerprint.")
        if stats["jpeg_block_boundary_score"] > 4.4:
            reasons.append("Block boundary variation looks inconsistent with a clean camera capture.")
        if stats["channel_correlation_gap"] > 0.18:
            reasons.append("Color channel relationships look atypical for a natural photograph.")
        if stats["entropy"] < 5.7:
            reasons.append("Low entropy and smooth regions can indicate synthetic generation.")
        if stats["laplacian_var"] < 65:
            reasons.append("Reduced edge sharpness is common in AI-generated images.")
        if stats["edge_density"] < 0.03:
            reasons.append("Sparse edge structure suggests an overly smooth rendering.")

        if not reasons:
            reasons.append("No strong artifact pattern was detected in the uploaded image.")

        overlay = self._build_overlay(image_bgr, blocks)
        suspicious_regions = self._top_regions(blocks)

        return ImageAuthenticityPrediction(
            label=label,
            confidence=confidence,
            probabilities={"AI_GENERATED": float(ai_probability), "ORIGINAL": float(1.0 - ai_probability)},
            reasons=reasons[:5],
            suspicious_regions=suspicious_regions,
            overlay_image=overlay,
            model_name=self.model_name,
        )

    def fit_from_directories(self, dataset_root: str) -> Tuple[float, int]:
        root = Path(dataset_root)
        ai_dir = root / "ai_generated"
        real_dir = root / "real"
        image_paths: List[Path] = []
        labels: List[str] = []

        for path in sorted(ai_dir.glob("**/*")):
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                image_paths.append(path)
                labels.append("AI_GENERATED")

        for path in sorted(real_dir.glob("**/*")):
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                image_paths.append(path)
                labels.append("ORIGINAL")

        if not image_paths:
            raise ValueError("No training images found. Expected data/image_authenticity/ai_generated and real folders.")

        feature_rows: List[np.ndarray] = []
        for path in image_paths:
            image_bytes = path.read_bytes()
            image_bgr, gray = self._decode_image(image_bytes)
            features, _, _ = self._extract_features(image_bgr, gray)
            feature_rows.append(features)

        features_matrix = np.vstack(feature_rows)
        y = np.array(labels)

        pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(max_iter=2000, class_weight="balanced", C=1.1, random_state=42),
                ),
            ]
        )
        pipeline.fit(features_matrix, y)
        self.pipeline = pipeline

        training_accuracy = float(pipeline.score(features_matrix, y))
        Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model_name": self.model_name,
                "feature_names": self.feature_names,
                "pipeline": pipeline,
            },
            self.model_path,
        )
        return training_accuracy, len(image_paths)

    def _augment_bgr_variants(self, image_bgr: np.ndarray) -> List[np.ndarray]:
        """Return a small set of augmented BGR images for data expansion."""
        variants: List[np.ndarray] = []
        try:
            h, w = image_bgr.shape[:2]
            # horizontal flip
            variants.append(cv2.flip(image_bgr, 1))
            # small rotations
            for angle in (-10, 10):
                M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
                variants.append(cv2.warpAffine(image_bgr, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT))
            # random crops + resize
            ch, cw = int(h * 0.9), int(w * 0.9)
            y0 = (h - ch) // 2
            x0 = (w - cw) // 2
            crop = image_bgr[y0 : y0 + ch, x0 : x0 + cw]
            variants.append(cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA))
            # color jitter via HSV shift
            hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.int32)
            for delta in (-10, 10):
                mod = hsv.copy()
                mod[:, :, 1] = np.clip(mod[:, :, 1] + delta, 0, 255)
                variants.append(cv2.cvtColor(mod.astype(np.uint8), cv2.COLOR_HSV2BGR))
        except Exception:
            return []
        return variants

    def fit_from_directories(self, dataset_root: str, augment: bool = False, min_samples_per_class: int = 200) -> Tuple[float, int]:
        """Train from directories; optionally augment small classes up to min_samples_per_class."""
        root = Path(dataset_root)
        ai_dir = root / "ai_generated"
        real_dir = root / "real"
        image_paths: List[Path] = []
        labels: List[str] = []

        for path in sorted(ai_dir.glob("**/*")):
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                image_paths.append(path)
                labels.append("AI_GENERATED")

        for path in sorted(real_dir.glob("**/*")):
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                image_paths.append(path)
                labels.append("ORIGINAL")

        if not image_paths:
            raise ValueError("No training images found. Expected data/image_authenticity/ai_generated and real folders.")

        # Organize by class for optional augmentation
        class_to_paths = {"AI_GENERATED": [], "ORIGINAL": []}
        for p, l in zip(image_paths, labels):
            class_to_paths[l].append(p)

        feature_rows: List[np.ndarray] = []
        y_labels: List[str] = []

        for cls, paths in class_to_paths.items():
            for path in paths:
                image_bytes = path.read_bytes()
                image_bgr, gray = self._decode_image(image_bytes)
                feats, _, _ = self._extract_features(image_bgr, gray)
                feature_rows.append(feats)
                y_labels.append(cls)

            if augment and len(paths) < min_samples_per_class:
                needed = min_samples_per_class - len(paths)
                # cycle through existing images and create simple variants
                idx = 0
                while needed > 0 and paths:
                    src_path = paths[idx % len(paths)]
                    b = src_path.read_bytes()
                    try:
                        bgr, g = self._decode_image(b)
                    except Exception:
                        idx += 1
                        continue
                    variants = self._augment_bgr_variants(bgr)
                    for var in variants:
                        if needed <= 0:
                            break
                        feats, _, _ = self._extract_features(var, cv2.cvtColor(var, cv2.COLOR_BGR2GRAY))
                        feature_rows.append(feats)
                        y_labels.append(cls)
                        needed -= 1
                    idx += 1

        X = np.vstack(feature_rows)
        y = np.array(y_labels)

        pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(max_iter=2000, class_weight="balanced", C=1.1, random_state=42),
                ),
            ]
        )
        pipeline.fit(X, y)
        self.pipeline = pipeline

        training_accuracy = float(pipeline.score(X, y))
        Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model_name": self.model_name,
                "feature_names": self.feature_names,
                "pipeline": pipeline,
            },
            self.model_path,
        )
        return training_accuracy, len(y)

    def fit_with_cv(self, dataset_root: str, cv: int = 5) -> Tuple[float, int]:
        """Train with simple cross-validation and return mean CV accuracy and sample count."""
        from sklearn.model_selection import StratifiedKFold

        root = Path(dataset_root)
        ai_dir = root / "ai_generated"
        real_dir = root / "real"
        image_paths: List[Path] = []
        labels: List[str] = []

        for path in sorted(ai_dir.glob("**/*")):
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                image_paths.append(path)
                labels.append("AI_GENERATED")

        for path in sorted(real_dir.glob("**/*")):
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                image_paths.append(path)
                labels.append("ORIGINAL")

        if not image_paths:
            raise ValueError("No training images found. Expected data/image_authenticity/ai_generated and real folders.")

        feature_rows: List[np.ndarray] = []
        for path in image_paths:
            image_bytes = path.read_bytes()
            image_bgr, gray = self._decode_image(image_bytes)
            features, _, _ = self._extract_features(image_bgr, gray)
            feature_rows.append(features)

        X = np.vstack(feature_rows)
        y = np.array(labels)

        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores = []
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            pipeline = Pipeline(
                steps=[("scaler", StandardScaler()), ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", C=1.1))]
            )
            pipeline.fit(X_train, y_train)
            scores.append(float(pipeline.score(X_test, y_test)))

        mean_score = float(np.mean(scores)) if scores else 0.0

        # Fit final model on all data and persist
        final_pipeline = Pipeline(steps=[("scaler", StandardScaler()), ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", C=1.1))])
        final_pipeline.fit(X, y)
        self.pipeline = final_pipeline
        Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model_name": self.model_name, "feature_names": self.feature_names, "pipeline": final_pipeline}, self.model_path)
        return mean_score, len(image_paths)


image_authenticity_singleton = ImageAuthenticityDetector()