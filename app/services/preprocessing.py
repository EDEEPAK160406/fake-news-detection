import re
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse
from datetime import datetime, timezone
import socket
from typing import Any

import cv2
import numpy as np

from app.core.config import settings

try:
    import whois
except Exception:  # pragma: no cover
    whois = None
try:
    import requests
    from bs4 import BeautifulSoup
except Exception:  # pragma: no cover
    requests = None
    BeautifulSoup = None

STOPWORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "and",
    "or",
    "but",
    "if",
    "then",
    "than",
    "to",
    "for",
    "of",
    "in",
    "on",
    "at",
    "by",
    "with",
    "about",
    "from",
    "as",
    "that",
    "this",
    "it",
    "its",
    "into",
    "over",
    "under",
    "after",
    "before",
}

SHORTENER_DOMAINS = {
    "bit.ly",
    "tinyurl.com",
    "t.co",
    "goo.gl",
    "ow.ly",
}

SUSPICIOUS_TLDS = {"zip", "review", "click", "gq", "country", "kim"}


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http[s]?://\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_text(text: str) -> List[str]:
    tokens = text.split()
    return [tok for tok in tokens if tok not in STOPWORDS and len(tok) > 2]


def preprocess_text(text: str) -> str:
    cleaned = clean_text(text)
    return " ".join(tokenize_text(cleaned))


def preprocess_image(image_bytes: bytes, size: Tuple[int, int] = (224, 224)) -> Optional[np.ndarray]:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        return None
    resized = cv2.resize(image, size)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return gray


def _is_ip_like(host: str) -> bool:
    return bool(re.fullmatch(r"\d{1,3}(?:\.\d{1,3}){3}", host))


def extract_url_features(url: Optional[str]) -> Dict[str, object]:
    if not url:
        # Return a vector matching the full feature length used below (16 features)
        return {
            "vector": np.zeros(16, dtype=float),
            "flags": ["no_url_provided"],
            "domain": "",
            "credibility_score": 0.5,
        }

    parsed = urlparse(url)
    domain = parsed.netloc.lower().replace("www.", "")
    path = parsed.path or ""

    is_https = 1 if parsed.scheme == "https" else 0
    url_len = len(url)
    domain_len = len(domain)
    has_at = 1 if "@" in url else 0
    dash_count = domain.count("-")
    digit_ratio = sum(c.isdigit() for c in domain) / max(domain_len, 1)
    path_depth = len([p for p in path.split("/") if p])
    is_ip = 1 if _is_ip_like(domain.split(":")[0]) else 0
    trusted_domains = {d.strip().lower() for d in settings.trusted_domains.split(",") if d.strip()}
    is_trusted_domain = 1 if any(domain == td or domain.endswith(f".{td}") for td in trusted_domains) else 0

    domain_age_days = 0.0
    if whois is not None and domain:
        try:
            rec = whois.whois(domain)
            creation = rec.creation_date
            if isinstance(creation, list) and creation:
                creation = creation[0]
            if creation is not None:
                if creation.tzinfo is None:
                    creation = creation.replace(tzinfo=timezone.utc)
                domain_age_days = max((datetime.now(timezone.utc) - creation).days, 0)
        except Exception:
            domain_age_days = 0.0

    flags: List[str] = []
    if not is_https:
        flags.append("url_not_https")
    if url_len > 90:
        flags.append("url_too_long")
    if has_at:
        flags.append("url_contains_at_symbol")
    if dash_count >= 2:
        flags.append("many_hyphens_in_domain")
    if digit_ratio > 0.25:
        flags.append("high_numeric_domain_pattern")
    if domain in SHORTENER_DOMAINS:
        flags.append("url_shortener_domain")
    if is_ip:
        flags.append("ip_based_url")
    if is_trusted_domain:
        flags.append("known_reputable_domain")
    if domain_age_days and domain_age_days < 180:
        flags.append("newly_registered_domain")

    tld = domain.rsplit(".", 1)[-1] if "." in domain else ""
    if tld in SUSPICIOUS_TLDS:
        flags.append("suspicious_top_level_domain")

    # DNS resolution / IP count
    resolves = 0
    num_ips = 0
    try:
        host = domain.split(":")[0]
        addrs = socket.gethostbyname_ex(host)[2]
        if addrs:
            resolves = 1
            num_ips = len(addrs)
    except Exception:
        resolves = 0
        num_ips = 0

    # Basic HTML fetch features (best-effort; may be slow)
    has_html = 0
    has_forms = 0
    external_links_ratio = 0.0
    num_scripts = 0
    contains_login = 0
    if requests is not None and BeautifulSoup is not None and domain:
        try:
            resp = requests.get(url, timeout=2.5)
            if resp.status_code == 200 and resp.headers.get("content-type", "").lower().startswith("text/html"):
                has_html = 1
                soup = BeautifulSoup(resp.text, "html.parser")
                forms = soup.find_all("form")
                has_forms = 1 if forms else 0
                scripts = soup.find_all("script")
                num_scripts = len(scripts)
                links = soup.find_all("a", href=True)
                total_links = len(links)
                external = 0
                for a in links:
                    href = a.get("href", "")
                    if href.startswith("http") and domain not in href:
                        external += 1
                external_links_ratio = float(external) / max(total_links, 1)
                page_text = soup.get_text(" ")[:2000].lower()
                if any(k in page_text for k in ("login", "sign in", "signin", "password", "otp")):
                    contains_login = 1
        except Exception:
            pass

    risk_flags = [flag for flag in flags if flag != "known_reputable_domain"]
    risk = min(len(risk_flags) / 6.0, 1.0)
    credibility_score = float(max(0.0, 1.0 - risk))

    vec = np.array(
        [
            is_https,
            url_len,
            domain_len,
            has_at,
            dash_count,
            digit_ratio,
            path_depth,
            is_ip,
            domain_age_days,
            resolves,
            num_ips,
            has_html,
            has_forms,
            external_links_ratio,
            num_scripts,
            contains_login,
        ],
        dtype=float,
    )

    return {
        "vector": vec,
        "flags": flags,
        "domain": domain,
        "credibility_score": credibility_score,
    }
