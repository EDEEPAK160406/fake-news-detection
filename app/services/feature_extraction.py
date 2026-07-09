import os
import re
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    import pytesseract
except Exception:  # pragma: no cover
    pytesseract = None


def _configure_tesseract() -> None:
    if pytesseract is None:
        return

    configured_cmd = os.getenv("TESSERACT_CMD", "").strip()
    if configured_cmd:
        pytesseract.pytesseract.tesseract_cmd = configured_cmd
        return

    if os.name != "nt":
        return

    windows_candidates = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for candidate in windows_candidates:
        if os.path.isfile(candidate):
            pytesseract.pytesseract.tesseract_cmd = candidate
            return


def _normalize_ocr_text(text: str) -> str:
    compact = re.sub(r"\s+", " ", text or " ").strip()
    return re.sub(r"[^\x20-\x7E]", "", compact)


def _cleanup_ocr_text(text: str) -> str:
    text = _normalize_ocr_text(text)
    if not text:
        return ""

    # Remove runs of OCR noise symbols while preserving normal punctuation.
    text = re.sub(r"[^A-Za-z0-9\s.,:;!?%$()'\"/-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""

    kept_words: List[str] = []
    for word in text.split():
        alnum_count = sum(ch.isalnum() for ch in word)
        if alnum_count == 0:
            continue
        symbol_ratio = 1.0 - (alnum_count / max(len(word), 1))
        if symbol_ratio > 0.55:
            continue
        kept_words.append(word)

    cleaned = " ".join(kept_words)
    return re.sub(r"\s+", " ", cleaned).strip()


def _ocr_text_quality(text: str) -> float:
    if not text:
        return 0.0

    words = text.split()
    if not words:
        return 0.0

    alpha_num_ratio = sum(ch.isalnum() for ch in text) / max(len(text), 1)
    plausible_words = sum(1 for word in words if re.search(r"[A-Za-z0-9]", word))
    plausible_ratio = plausible_words / max(len(words), 1)
    avg_word_len = sum(len(word) for word in words) / max(len(words), 1)
    length_score = min(len(text) / 80.0, 1.0)

    shape_score = 1.0
    if avg_word_len > 18:
        shape_score -= 0.3
    if avg_word_len < 2:
        shape_score -= 0.3

    quality = 0.45 * alpha_num_ratio + 0.3 * plausible_ratio + 0.15 * length_score + 0.1 * max(shape_score, 0.0)
    return float(min(max(quality, 0.0), 1.0))


def _deskew(gray_image: np.ndarray) -> np.ndarray:
    try:
        inv = cv2.bitwise_not(gray_image)
        thresh = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        coords = np.column_stack(np.where(thresh > 0))
        if coords.shape[0] < 50:
            return gray_image

        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        if abs(angle) < 0.5 or abs(angle) > 20:
            return gray_image

        h, w = gray_image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(gray_image, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    except Exception:
        return gray_image


def _ocr_variants(gray_image: np.ndarray) -> List[np.ndarray]:
    variants: List[np.ndarray] = [gray_image, _deskew(gray_image)]

    h, w = gray_image.shape[:2]
    if min(h, w) < 900:
        enlarged = cv2.resize(gray_image, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        variants.append(enlarged)

    blurred = cv2.GaussianBlur(gray_image, (3, 3), 0)
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(otsu)

    adaptive = cv2.adaptiveThreshold(
        gray_image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )
    variants.append(adaptive)

    return variants


def extract_ocr_text(gray_image: Optional[np.ndarray]) -> str:
    if gray_image is None or pytesseract is None:
        return ""

    _configure_tesseract()
    ocr_configs = [
        "--oem 3 --psm 6 -l eng",
        "--oem 3 --psm 11 -l eng",
        "--oem 3 --psm 3 -l eng",
    ]
    best_text = ""
    best_score = 0.0

    try:
        for image_variant in _ocr_variants(gray_image):
            for config in ocr_configs:
                try:
                    data = pytesseract.image_to_data(
                        image_variant,
                        config=config,
                        output_type=pytesseract.Output.DICT,
                    )

                    words: List[str] = []
                    conf_values: List[float] = []
                    for raw_word, raw_conf in zip(data.get("text", []), data.get("conf", [])):
                        word = str(raw_word or "").strip()
                        if not word:
                            continue
                        try:
                            conf = float(raw_conf)
                        except Exception:
                            conf = -1.0
                        if conf >= 35:
                            words.append(word)
                            conf_values.append(conf)

                    if words:
                        candidate = " ".join(words)
                    else:
                        candidate = pytesseract.image_to_string(image_variant, config=config)

                    candidate = _cleanup_ocr_text(candidate)
                except Exception:
                    continue

                confidence_score = 0.0
                if conf_values:
                    confidence_score = min(max(sum(conf_values) / (len(conf_values) * 100.0), 0.0), 1.0)
                quality_score = _ocr_text_quality(candidate)
                final_score = 0.65 * quality_score + 0.35 * confidence_score

                if final_score > best_score or (final_score == best_score and len(candidate) > len(best_text)):
                    best_score = final_score
                    best_text = candidate

        if best_score < 0.2 and len(best_text) < 5:
            return ""
        return best_text
    except Exception:
        return ""


def extract_image_features(gray_image: Optional[np.ndarray]) -> Tuple[np.ndarray, List[str]]:
    """Create robust handcrafted image features and quality/manipulation hints."""
    if gray_image is None:
        return np.zeros(8, dtype=float), ["no_image_provided"]

    mean_intensity = float(np.mean(gray_image))
    std_intensity = float(np.std(gray_image))
    lap_var = float(cv2.Laplacian(gray_image, cv2.CV_64F).var())

    edges = cv2.Canny(gray_image, 100, 200)
    edge_density = float(np.count_nonzero(edges)) / float(edges.size)

    h, w = gray_image.shape[:2]
    center_crop = gray_image[h // 4 : (3 * h) // 4, w // 4 : (3 * w) // 4]
    center_mean = float(np.mean(center_crop)) if center_crop.size else mean_intensity

    top_half = gray_image[: h // 2, :]
    bottom_half = gray_image[h // 2 :, :]
    half_mean_diff = float(abs(np.mean(top_half) - np.mean(bottom_half)))

    features = np.array(
        [
            mean_intensity,
            std_intensity,
            lap_var,
            edge_density,
            center_mean,
            half_mean_diff,
            h,
            w,
        ],
        dtype=float,
    )

    flags: List[str] = []
    if lap_var < 55:
        flags.append("low_image_sharpness")
    if edge_density > 0.25:
        flags.append("high_edge_density_screenshot_like")
    if half_mean_diff > 35:
        flags.append("high_lighting_inconsistency")

    return features, flags
