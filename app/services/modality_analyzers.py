from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.services.feature_extraction import extract_image_features, extract_ocr_text
from app.services.preprocessing import extract_url_features, preprocess_image


@dataclass
class ModalitySignal:
    used: bool
    fake_probability: float
    confidence: float
    reasons: List[str]
    meta: Dict[str, object]


def _clamp_prob(value: float) -> float:
    return float(min(max(value, 0.02), 0.98))


def _confidence_from_prob(fake_probability: float) -> float:
    return float(abs(fake_probability - 0.5) * 2.0)


def is_brief_factual_claim(text: str) -> bool:
    lowered = (text or "").lower().strip()
    if not lowered:
        return False

    tokens = lowered.split()
    if len(tokens) > 20:
        return False

    factual_markers = ["won", "beat", "defeated", "vs", "against", "in", "on", "at"]
    sports_markers = ["ipl", "match", "t20", "odi", "test", "league", "srh", "rcb", "mi", "csk"]
    has_number = bool(re.search(r"\b\d+(st|nd|rd|th)?\b", lowered))

    factual_hits = sum(marker in lowered for marker in factual_markers)
    sports_hits = sum(marker in lowered for marker in sports_markers)
    sensational = any(term in lowered for term in ["shocking", "miracle", "secret", "100%", "must read"])

    return (factual_hits >= 2 and (sports_hits >= 1 or has_number)) and not sensational


def is_sports_result_claim(text: str) -> bool:
    lowered = (text or "").lower().strip()
    if not lowered:
        return False

    sport_terms = ["ipl", "match", "wickets", "runs", "t20", "odi", "league", "beat", "defeated", "won"]
    hits = sum(term in lowered for term in sport_terms)
    has_score_pattern = bool(re.search(r"\b\d+\s*(wickets?|runs?)\b", lowered))
    return hits >= 2 and has_score_pattern


def analyze_text_signal(text: str) -> ModalitySignal:
    lowered = (text or "").lower().strip()
    if not lowered:
        return ModalitySignal(
            used=False,
            fake_probability=0.5,
            confidence=0.0,
            reasons=["No text provided."],
            meta={"word_count": 0, "brief_factual_claim": False, "suspicious_terms": [], "input_text": ""},
        )

    word_count = len(lowered.split())
    brief_factual = is_brief_factual_claim(lowered)
    sports_result_claim = is_sports_result_claim(lowered)

    score = 0.5
    reasons: List[str] = []

    suspicious_terms = [
        "shocking",
        "must read",
        "secret",
        "miracle",
        "guaranteed",
        "100%",
        "urgent",
        "doctors hate",
        "hidden truth",
        "go dark",
        "rare cosmic event",
    ]
    trust_terms = ["according to", "report", "official statement", "reuters", "ap"]

    hit_suspicious = [term for term in suspicious_terms if term in lowered]
    hit_trust = [term for term in trust_terms if term in lowered]

    score += min(len(hit_suspicious) * 0.055, 0.24)
    score -= min(len(hit_trust) * 0.035, 0.14)

    if word_count < 10:
        score += 0.08
        reasons.append("Text is very short, so standalone verification is harder.")
    elif word_count > 120:
        score -= 0.05

    if hit_suspicious:
        reasons.append(f"Sensational text cues found: {', '.join(hit_suspicious[:3])}.")
    if hit_trust:
        reasons.append("Contains source-style language often seen in factual reporting.")

    if brief_factual:
        score -= 0.06
        reasons.append("Detected concise factual event statement pattern.")

    fake_prob = _clamp_prob(score)
    if not reasons:
        reasons.append("Text module found no strong risk indicators.")

    return ModalitySignal(
        used=True,
        fake_probability=fake_prob,
        confidence=_confidence_from_prob(fake_prob),
        reasons=reasons[:4],
        meta={
            "word_count": word_count,
            "brief_factual_claim": brief_factual,
            "sports_result_claim": sports_result_claim,
            "suspicious_terms": hit_suspicious,
            "input_text": text,
        },
    )


def analyze_url_signal(url: Optional[str]) -> ModalitySignal:
    url_meta = extract_url_features(url)
    flags = url_meta["flags"]
    vector = url_meta["vector"]

    if not url:
        return ModalitySignal(
            used=False,
            fake_probability=0.5,
            confidence=0.0,
            reasons=["No URL provided."],
            meta={
                "flags": flags,
                "credibility_score": float(url_meta["credibility_score"]),
                "vector": vector,
                "domain": str(url_meta.get("domain", "")),
            },
        )

    credibility = float(url_meta["credibility_score"])
    score = 1.0 - credibility

    if "known_reputable_domain" in flags:
        score -= 0.12
    if "url_not_https" in flags:
        score += 0.05
    if "url_shortener_domain" in flags:
        score += 0.16
    if "ip_based_url" in flags:
        score += 0.18
    if "suspicious_top_level_domain" in flags:
        score += 0.14

    score = _clamp_prob(score)
    reasons: List[str] = []
    risk_flags = [f for f in flags if f not in {"known_reputable_domain"}]

    if risk_flags:
        reasons.append(f"URL flags: {', '.join(risk_flags[:3])}.")
    if credibility >= 0.7:
        reasons.append("URL structure appears relatively credible.")
    elif credibility <= 0.45:
        reasons.append("URL structure appears suspicious.")

    if not reasons:
        reasons.append("URL module found mixed signals.")

    return ModalitySignal(
        used=True,
        fake_probability=score,
        confidence=_confidence_from_prob(score),
        reasons=reasons[:4],
        meta={
            "flags": flags,
            "credibility_score": credibility,
            "vector": vector,
            "domain": str(url_meta.get("domain", "")),
        },
    )


def analyze_image_signal(image_bytes: Optional[bytes]) -> Tuple[ModalitySignal, np.ndarray, str, List[str]]:
    if image_bytes is None:
        signal = ModalitySignal(
            used=False,
            fake_probability=0.5,
            confidence=0.0,
            reasons=["No image provided."],
            meta={"ocr_words": 0, "ocr_text": ""},
        )
        return signal, np.zeros(8, dtype=float), "", ["no_image_provided"]

    gray_image = preprocess_image(image_bytes)

    # OCR on the original image resolution is more reliable than using a downscaled tensor.
    ocr_gray = None
    try:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        ocr_gray = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    except Exception:
        ocr_gray = None

    img_vec, img_flags = extract_image_features(gray_image)
    ocr_text = extract_ocr_text(ocr_gray if ocr_gray is not None else gray_image)

    score = 0.5
    reasons: List[str] = []

    if "low_image_sharpness" in img_flags:
        score += 0.06
    if "high_edge_density_screenshot_like" in img_flags:
        score += 0.04
    if "high_lighting_inconsistency" in img_flags:
        score += 0.05

    if img_flags and "no_image_provided" not in img_flags:
        reasons.append(f"Image quality/manipulation cues: {', '.join(img_flags[:3])}.")

    ocr_words = len((ocr_text or "").split())
    if ocr_words >= 6:
        reasons.append("OCR text extracted and used as additional evidence.")

    score = _clamp_prob(score)
    if not reasons:
        reasons.append("Image module found no strong risk indicators.")

    signal = ModalitySignal(
        used=True,
        fake_probability=score,
        confidence=_confidence_from_prob(score),
        reasons=reasons[:4],
        meta={"ocr_words": ocr_words, "ocr_text": ocr_text[:1200] if ocr_text else ""},
    )
    return signal, img_vec, ocr_text, img_flags
