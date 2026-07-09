from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from app.services.modality_analyzers import (
    analyze_image_signal,
    analyze_text_signal,
    analyze_url_signal,
)
from app.services.fact_check import FactCheckSignal, analyze_fact_check_signal
from app.services.preprocessing import preprocess_text


@dataclass
class PredictionArtifacts:
    label: str
    confidence: float
    probabilities: Dict[str, float]
    risk_score: float
    module_scores: Dict[str, float]
    module_reasons: Dict[str, List[str]]
    module_details: Dict[str, Dict[str, object]]
    reasons: List[str]


class MultimodalFakeNewsClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=7000,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.97,
            sublinear_tf=True,
        )
        self.scaler = StandardScaler(with_mean=False)
        self.model = LogisticRegression(max_iter=1500, class_weight="balanced", C=1.2)
        self.is_fitted = False
        self.decision_threshold = 0.58

    @staticmethod
    def _text_statistics(text: str) -> np.ndarray:
        raw_text = text or ""
        stripped = raw_text.strip()
        if not stripped:
            return np.zeros(10, dtype=float)

        words = re.findall(r"\b\w+\b", stripped)
        word_count = len(words)
        char_count = len(stripped)
        avg_word_length = float(sum(len(word) for word in words) / max(word_count, 1))
        sentence_count = max(len(re.findall(r"[.!?]+", stripped)), 1)
        exclamation_count = float(stripped.count("!"))
        question_count = float(stripped.count("?"))
        uppercase_ratio = float(sum(1 for char in stripped if char.isupper()) / max(char_count, 1))
        digit_ratio = float(sum(1 for char in stripped if char.isdigit()) / max(char_count, 1))
        url_count = float(len(re.findall(r"https?://\S+|www\.\S+", stripped.lower())))
        unique_word_ratio = float(len(set(word.lower() for word in words)) / max(word_count, 1))

        return np.array(
            [
                word_count,
                char_count,
                avg_word_length,
                sentence_count,
                exclamation_count,
                question_count,
                uppercase_ratio,
                digit_ratio,
                url_count,
                unique_word_ratio,
            ],
            dtype=float,
        )

    def _build_feature_matrix(
        self,
        texts: List[str],
        urls: List[Optional[str]],
        image_features: np.ndarray,
        fit: bool,
    ) -> np.ndarray:
        processed_texts = [preprocess_text(t or "") for t in texts]
        if fit:
            text_vec = self.vectorizer.fit_transform(processed_texts)
        else:
            text_vec = self.vectorizer.transform(processed_texts)

        stats_vec = np.vstack([self._text_statistics(text) for text in texts])
        # URL feature vector length can change as we evolve heuristics. Build raw url_vec,
        # then at prediction time adjust (pad/truncate) to match the scaler's expected input size
        url_vec = np.vstack([analyze_url_signal(url).meta.get("vector", np.zeros(9, dtype=float)) for url in urls])

        dense_text = text_vec.toarray()
        combined = np.hstack([dense_text, stats_vec, image_features, url_vec])

        # If we're transforming (not fitting) and a scaler exists from training,
        # ensure feature dimensionality matches what the scaler expects.
        if not fit and hasattr(self.scaler, "n_features_in_"):
            expected = int(self.scaler.n_features_in_)
            current = combined.shape[1]
            if current != expected:
                diff = expected - current
                # Prefer padding zeros for new features; if too many features, truncate URL columns last.
                if diff > 0:
                    pad = np.zeros((combined.shape[0], diff), dtype=float)
                    combined = np.hstack([combined, pad])
                else:
                    # truncate from the end (safest: drop extra URL-derived columns)
                    combined = combined[:, :expected]

        if fit:
            return self.scaler.fit_transform(combined)
        return self.scaler.transform(combined)

    def fit(
        self,
        texts: List[str],
        labels: List[str],
        urls: Optional[List[Optional[str]]] = None,
        image_features: Optional[np.ndarray] = None,
    ) -> None:
        if not texts or not labels:
            raise ValueError("Training data cannot be empty")

        normalized_labels = [str(label).upper().strip() for label in labels]
        if urls is None:
            urls = [None] * len(texts)
        if image_features is None:
            image_features = np.zeros((len(texts), 8), dtype=float)

        x = self._build_feature_matrix(texts, urls, image_features, fit=True)
        y = np.array(normalized_labels)
        self.model.fit(x, y)
        self.is_fitted = True

    def predict_single(
        self,
        text: Optional[str],
        url: Optional[str],
        image_bytes: Optional[bytes],
        enable_fact_check: bool = True,
    ) -> PredictionArtifacts:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Train or load artifacts first.")

        image_signal, img_vec, ocr_text, image_flags = analyze_image_signal(image_bytes)
        merged_text = (text or "").strip()
        if ocr_text:
            merged_text = f"{merged_text} {ocr_text}".strip()

        text_signal = analyze_text_signal(merged_text)
        url_signal = analyze_url_signal(url)
        fact_check_signal = analyze_fact_check_signal(merged_text) if enable_fact_check else None
        if fact_check_signal is None:
            fact_check_signal = FactCheckSignal(
                used=False,
                fake_probability=0.5,
                confidence=0.0,
                reasons=["Internet fact-check disabled for this run."],
                meta={"claims_checked": 0, "avg_overlap": 0.0, "support_hits": 0, "contradiction_hits": 0, "sources_used": [], "evidence": []},
            )

        x = self._build_feature_matrix([merged_text], [url], img_vec.reshape(1, -1), fit=False)
        probs_arr = self.model.predict_proba(x)[0]
        classes = list(self.model.classes_)
        ml_probabilities = {cls: float(prob) for cls, prob in zip(classes, probs_arr)}
        ml_fake_prob = float(ml_probabilities.get("FAKE", 0.5))

        module_heuristic_fake_prob = self._fuse_module_probabilities(
            text_signal=text_signal,
            url_signal=url_signal,
            image_signal=image_signal,
            fact_check_signal=fact_check_signal,
        )
        blended_fake_prob = self._blend_fake_probability(
            ml_fake_prob,
            module_heuristic_fake_prob,
            text_signal,
            url_signal,
            image_signal,
            fact_check_signal,
        )
        blended_fake_prob = self._apply_final_calibration(
            blended_fake_prob,
            ml_fake_prob,
            module_heuristic_fake_prob,
            bool(text_signal.meta.get("brief_factual_claim", False)),
            merged_text,
            list(url_signal.meta.get("flags", [])),
            int(text_signal.meta.get("word_count", 0)),
            float(url_signal.meta.get("credibility_score", 0.5)),
        )

        fact_contradictions = int(fact_check_signal.meta.get("contradiction_hits", 0))
        fact_support = int(fact_check_signal.meta.get("support_hits", 0))
        lowered_text = (merged_text or "").lower()
        has_sensational = any(term in lowered_text for term in ["shocking", "secret", "miracle", "100%", "must read", "urgent"]) 
        text_only_mode = bool(merged_text) and not bool(url)
        sports_result_claim = bool(text_signal.meta.get("sports_result_claim", False))

        if bool(text_signal.meta.get("brief_factual_claim", False)) and not has_sensational and fact_contradictions == 0:
            if text_signal.fake_probability <= 0.52:
                blended_fake_prob -= 0.2
            else:
                blended_fake_prob -= 0.1
        if fact_support > 0 and fact_contradictions == 0:
            blended_fake_prob -= 0.08
        if fact_contradictions > 0:
            blended_fake_prob += 0.1

        if (
            text_only_mode
            and sports_result_claim
            and fact_support == 0
            and fact_contradictions == 0
            and text_signal.fake_probability >= 0.55
        ):
            blended_fake_prob += 0.02

        # For text-only news where internet evidence is inconclusive, avoid over-flagging real headlines.
        if text_only_mode and fact_contradictions == 0 and not has_sensational and text_signal.fake_probability <= 0.52 and not sports_result_claim:
            blended_fake_prob -= 0.06
        if text_only_mode and fact_contradictions == 0 and int(text_signal.meta.get("word_count", 0)) >= 8 and text_signal.fake_probability <= 0.52 and not sports_result_claim:
            blended_fake_prob -= 0.02

        blended_fake_prob = float(min(max(blended_fake_prob, 0.02), 0.98))

        effective_threshold = self.decision_threshold
        word_count = int(text_signal.meta.get("word_count", 0))
        url_credibility = float(url_signal.meta.get("credibility_score", 0.5))
        url_flags = list(url_signal.meta.get("flags", []))
        has_high_risk_url_flag = any(flag in {"ip_based_url", "url_shortener_domain", "suspicious_top_level_domain"} for flag in url_flags)
        if word_count >= 160 and url_credibility >= 0.75 and not has_high_risk_url_flag:
            effective_threshold = min(self.decision_threshold + 0.08, 0.9)
        if not url and word_count >= 8 and word_count <= 90 and fact_contradictions == 0 and text_signal.fake_probability <= 0.52 and not sports_result_claim:
            effective_threshold = min(effective_threshold + 0.04, 0.88)

        label = "FAKE" if blended_fake_prob >= effective_threshold else "REAL"
        confidence = float(blended_fake_prob if label == "FAKE" else (1.0 - blended_fake_prob))
        probabilities = {
            "FAKE": blended_fake_prob,
            "REAL": float(1.0 - blended_fake_prob),
        }

        module_scores = {
            "text": text_signal.fake_probability,
            "url": url_signal.fake_probability,
            "image": image_signal.fake_probability,
            "fact_check": fact_check_signal.fake_probability,
        }
        module_reasons = {
            "text": text_signal.reasons,
            "url": url_signal.reasons,
            "image": image_signal.reasons,
            "fact_check": fact_check_signal.reasons,
        }
        module_details = {
            "text": {
                "confidence": text_signal.confidence,
                "word_count": int(text_signal.meta.get("word_count", 0)),
                "suspicious_terms": list(text_signal.meta.get("suspicious_terms", [])),
            },
            "url": {
                "confidence": url_signal.confidence,
                "domain": str(url_signal.meta.get("domain", "")),
                "flags": list(url_signal.meta.get("flags", [])),
                "credibility_score": float(url_signal.meta.get("credibility_score", 0.5)),
            },
            "image": {
                "confidence": image_signal.confidence,
                "ocr_text": str(image_signal.meta.get("ocr_text", "")),
                "ocr_words": int(image_signal.meta.get("ocr_words", 0)),
            },
            "fact_check": {
                "confidence": fact_check_signal.confidence,
                "claims_checked": int(fact_check_signal.meta.get("claims_checked", 0)),
                "avg_overlap": float(fact_check_signal.meta.get("avg_overlap", 0.0)),
                "support_hits": int(fact_check_signal.meta.get("support_hits", 0)),
                "contradiction_hits": int(fact_check_signal.meta.get("contradiction_hits", 0)),
                "sources_used": list(fact_check_signal.meta.get("sources_used", [])),
                "evidence": list(fact_check_signal.meta.get("evidence", [])),
            },
        }

        reasons: List[str] = []
        reasons.extend(fact_check_signal.reasons[:2])
        reasons.extend(text_signal.reasons[:2])
        reasons.extend(url_signal.reasons[:2])
        reasons.extend(image_signal.reasons[:2])
        if bool(text_signal.meta.get("brief_factual_claim", False)) and label == "REAL":
            reasons.append("Detected concise factual event-style wording, lowering false-positive risk.")
        if fact_support > 0 and fact_contradictions == 0:
            reasons.append("Internet fact-check found support cues without contradiction, lowering fake risk.")
        if (
            text_only_mode
            and sports_result_claim
            and fact_support == 0
            and fact_contradictions == 0
            and text_signal.fake_probability >= 0.55
        ):
            reasons.append("Sports result claim lacked corroboration and had elevated text risk cues.")
        if effective_threshold > self.decision_threshold:
            reasons.append("Applied stricter fake threshold for this context to reduce false positives.")
        reasons.append(
            f"Hybrid score combines model confidence ({ml_fake_prob:.2f} fake-prob) with modular heuristics ({module_heuristic_fake_prob:.2f})."
        )

        if not reasons:
            reasons.append("No strong risk indicators found across text, image, and URL inputs.")

        return PredictionArtifacts(
            label=label,
            confidence=confidence,
            probabilities=probabilities,
            risk_score=blended_fake_prob,
            module_scores=module_scores,
            module_reasons=module_reasons,
            module_details=module_details,
            reasons=reasons[:6],
        )

    @staticmethod
    def _fuse_module_probabilities(
        text_signal,
        url_signal,
        image_signal,
        fact_check_signal,
    ) -> float:
        weights = {
            "text": 0.42,
            "url": 0.22,
            "image": 0.1,
            "fact_check": 0.26,
        }
        weighted = (
            text_signal.fake_probability * weights["text"]
            + url_signal.fake_probability * weights["url"]
            + image_signal.fake_probability * weights["image"]
            + fact_check_signal.fake_probability * weights["fact_check"]
        )
        return float(min(max(weighted, 0.02), 0.98))

    @staticmethod
    def _blend_fake_probability(
        ml_fake_prob: float,
        heuristic_fake_prob: float,
        text_signal,
        url_signal,
        image_signal,
        fact_check_signal,
    ) -> float:
        word_count = int(text_signal.meta.get("word_count", 0))
        heuristic_weight = 0.3

        if word_count < 15 and text_signal.used:
            heuristic_weight += 0.08
        if not url_signal.used:
            heuristic_weight += 0.05
        if not image_signal.used:
            heuristic_weight += 0.03
        if abs(ml_fake_prob - 0.5) < 0.12:
            heuristic_weight += 0.1
        if bool(text_signal.meta.get("brief_factual_claim", False)):
            heuristic_weight -= 0.08
        if fact_check_signal.used and fact_check_signal.confidence >= 0.25:
            heuristic_weight += 0.08

        heuristic_weight = min(max(heuristic_weight, 0.2), 0.55)
        ml_weight = 1.0 - heuristic_weight
        blended = (ml_fake_prob * ml_weight) + (heuristic_fake_prob * heuristic_weight)
        return float(min(max(blended, 0.02), 0.98))

    @staticmethod
    def _apply_final_calibration(
        blended_fake_prob: float,
        ml_fake_prob: float,
        heuristic_fake_prob: float,
        brief_factual_claim: bool,
        text: str,
        url_flags: List[str],
        word_count: int,
        url_credibility: float,
    ) -> float:
        lowered = (text or "").lower()
        has_sensational = any(term in lowered for term in ["shocking", "secret", "miracle", "100%", "must read"])
        high_risk_url = any(flag in url_flags for flag in ["ip_based_url", "url_shortener_domain", "suspicious_top_level_domain"])

        if (
            brief_factual_claim
            and not has_sensational
            and not high_risk_url
            and heuristic_fake_prob < 0.5
            and ml_fake_prob < 0.78
        ):
            blended_fake_prob -= 0.14

        # Favor REAL for long-form reporting from credible, non-risky domains.
        if word_count >= 180 and url_credibility >= 0.75 and not has_sensational and not high_risk_url:
            blended_fake_prob -= 0.15
        if word_count >= 320 and url_credibility >= 0.85 and not has_sensational and not high_risk_url:
            blended_fake_prob -= 0.06

        if has_sensational and high_risk_url:
            blended_fake_prob += 0.08

        return float(min(max(blended_fake_prob, 0.02), 0.98))

    def save(self, model_path: str, vectorizer_path: str) -> None:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(os.path.dirname(vectorizer_path), exist_ok=True)

        joblib.dump(
            {
                "model": self.model,
                "scaler": self.scaler,
                "is_fitted": self.is_fitted,
                "decision_threshold": self.decision_threshold,
                "version": "hybrid-v2",
            },
            model_path,
        )
        joblib.dump(self.vectorizer, vectorizer_path)

    def load(self, model_path: str, vectorizer_path: str) -> None:
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.scaler = bundle["scaler"]
        self.is_fitted = bool(bundle.get("is_fitted", True))
        loaded_threshold = float(bundle.get("decision_threshold", 0.58))
        self.decision_threshold = float(min(max(loaded_threshold, 0.56), 0.75))
        self.vectorizer = joblib.load(vectorizer_path)


classifier_singleton = MultimodalFakeNewsClassifier()
