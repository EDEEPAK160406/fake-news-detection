from __future__ import annotations

import html
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Dict, List
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup


@dataclass
class FactCheckSignal:
    used: bool
    fake_probability: float
    confidence: float
    reasons: List[str]
    meta: Dict[str, object]


def _clamp(value: float) -> float:
    return float(min(max(value, 0.02), 0.98))


def _confidence(fake_probability: float) -> float:
    return float(abs(fake_probability - 0.5) * 2.0)


def _tokenize(text: str) -> List[str]:
    try:
        s = text or ""
        if not isinstance(s, str):
            s = str(s)
        s = s.lower()
        # Replace non-word characters with space to avoid regex surprises
        s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
        tokens = re.findall(r"\w+", s, flags=re.UNICODE)
        # Keep tokens containing ascii letters or digits
        return [t for t in tokens if re.search(r"[a-z0-9]", t)]
    except Exception:
        return []


STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "and",
    "or",
    "by",
    "from",
    "that",
    "this",
    "it",
    "as",
    "at",
    "be",
    "after",
    "before",
    "into",
    "about",
}

TRUSTED_DOMAINS = {
    "reuters.com",
    "apnews.com",
    "bbc.com",
    "nytimes.com",
    "thehindu.com",
    "indianexpress.com",
    "livemint.com",
    "snopes.com",
    "who.int",
    "nasa.gov",
    "noaa.gov",
    "un.org",
    "gov.in",
}

LOW_TRUST_DOMAINS = {
    "beforeitsnews.com",
    "naturalnews.com",
    "infowars.com",
    "blogspot.com",
    "wordpress.com",
}

CONTRADICTION_TERMS = {
    "false",
    "hoax",
    "fake",
    "debunk",
    "misleading",
    "not true",
    "baseless",
    "fabricated",
    "no evidence",
    "satire",
}

SUPPORT_TERMS = {
    "official",
    "confirmed",
    "according to",
    "statement",
    "report",
    "verified",
    "press release",
    "announced",
}


def _extract_claims(text: str, max_claims: int = 2) -> List[str]:
    cleaned = re.sub(r"\s+", " ", (text or "")).strip()
    if not cleaned:
        return []

    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    scored: List[tuple[int, str]] = []
    for part in parts:
        sentence = part.strip()
        if len(sentence) < 45:
            continue
        alpha_count = sum(char.isalpha() for char in sentence)
        digit_count = sum(char.isdigit() for char in sentence)
        score = alpha_count + (digit_count * 2)
        scored.append((score, sentence))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in scored[:max_claims]]


def _domain_weight(url: str) -> float:
    domain = (urlparse(url).netloc or "").lower().replace("www.", "")
    if not domain:
        return 0.9

    if any(domain == trusted or domain.endswith(f".{trusted}") for trusted in TRUSTED_DOMAINS):
        return 1.25
    if any(domain == low or domain.endswith(f".{low}") for low in LOW_TRUST_DOMAINS):
        return 0.65
    return 0.95


def _token_overlap(a: str, b: str) -> float:
    ta = [t for t in _tokenize(a) if t not in STOPWORDS]
    tb = [t for t in _tokenize(b) if t not in STOPWORDS]
    if not ta or not tb:
        return 0.0

    sa = set(ta)
    sb = set(tb)
    common = len(sa & sb)
    return float(common / max(len(sa), 1))


def _text_similarity(a: str, b: str) -> float:
    return float(SequenceMatcher(None, (a or "").lower(), (b or "").lower()).ratio())


def _relevance_score(claim: str, snippet: str) -> float:
    overlap = _token_overlap(claim, snippet)
    similarity = _text_similarity(claim, snippet)
    return float((0.7 * overlap) + (0.3 * similarity))


def _stance(snippet: str) -> str:
    lowered = (snippet or "").lower()
    has_contra = any(term in lowered for term in CONTRADICTION_TERMS)
    has_support = any(term in lowered for term in SUPPORT_TERMS)
    if has_contra and not has_support:
        return "contradiction"
    if has_support and not has_contra:
        return "support"
    return "neutral"


@lru_cache(maxsize=128)
def _duckduckgo_api_snippets(query: str) -> List[str]:
    try:
        resp = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
            timeout=8,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        if resp.status_code != 200:
            return []

        payload = resp.json()
        snippets: List[str] = []

        abstract = str(payload.get("AbstractText", "")).strip()
        if abstract:
            snippets.append(abstract)

        heading = str(payload.get("Heading", "")).strip()
        if heading:
            snippets.append(heading)

        related = payload.get("RelatedTopics", [])
        if isinstance(related, list):
            for item in related[:6]:
                if isinstance(item, dict) and "Text" in item:
                    text = str(item.get("Text", "")).strip()
                    if text:
                        snippets.append(text)
                elif isinstance(item, dict) and "Topics" in item:
                    for sub in item.get("Topics", [])[:3]:
                        text = str(sub.get("Text", "")).strip()
                        if text:
                            snippets.append(text)

        return snippets[:8]
    except Exception:
        return []


@lru_cache(maxsize=128)
def _duckduckgo_html_results(query: str) -> List[Dict[str, str]]:
    try:
        resp = requests.get(
            "https://duckduckgo.com/html/",
            params={"q": query},
            timeout=8,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        if resp.status_code != 200:
            return []

        soup = BeautifulSoup(resp.text, "html.parser")
        results: List[Dict[str, str]] = []

        for node in soup.select("div.result")[:12]:
            link_el = node.select_one("a.result__a")
            snippet_el = node.select_one("a.result__snippet, div.result__snippet")

            href = link_el.get("href", "").strip() if link_el else ""
            snippet = snippet_el.get_text(" ", strip=True) if snippet_el else ""
            title = link_el.get_text(" ", strip=True) if link_el else ""

            text = " ".join(part for part in [title, snippet] if part).strip()
            if len(text) < 30:
                continue

            results.append({"text": text, "url": href, "source": "duckduckgo_html"})

        return results
    except Exception:
        return []


@lru_cache(maxsize=128)
def _google_news_rss_results(query: str) -> List[Dict[str, str]]:
    try:
        resp = requests.get(
            "https://news.google.com/rss/search",
            params={"q": query, "hl": "en-IN", "gl": "IN", "ceid": "IN:en"},
            timeout=8,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        if resp.status_code != 200:
            return []

        soup = BeautifulSoup(resp.text, "xml")
        items = soup.find_all("item")[:10]
        results: List[Dict[str, str]] = []

        for item in items:
            title = (item.title.text if item.title else "").strip()
            link = (item.link.text if item.link else "").strip()
            description = (item.description.text if item.description else "").strip()
            description = BeautifulSoup(html.unescape(description), "html.parser").get_text(" ", strip=True)

            text = " ".join(part for part in [title, description] if part).strip()
            if len(text) < 30:
                continue

            results.append({"text": text, "url": link, "source": "google_news_rss"})

        return results
    except Exception:
        return []


@lru_cache(maxsize=128)
def _wikipedia_snippets(query: str) -> List[str]:
    try:
        resp = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "opensearch",
                "search": query,
                "limit": 5,
                "namespace": 0,
                "format": "json",
            },
            timeout=8,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        if resp.status_code != 200:
            return []

        payload = resp.json()
        if not isinstance(payload, list) or len(payload) < 3:
            return []

        descriptions = payload[2] if isinstance(payload[2], list) else []
        return [str(item).strip() for item in descriptions if str(item).strip()][:5]
    except Exception:
        return []


def analyze_fact_check_signal(text: str) -> FactCheckSignal:
    claims = _extract_claims(text)
    if not claims:
        return FactCheckSignal(
            used=False,
            fake_probability=0.5,
            confidence=0.0,
            reasons=["Internet fact-check skipped due to insufficient claim text."],
            meta={"claims_checked": 0, "sources_used": [], "evidence": []},
        )

    trusted_entity_terms = {
        "nasa",
        "noaa",
        "who",
        "cdc",
        "reuters",
        "ap",
        "bbc",
        "government",
        "official",
        "university",
        "research",
        "study",
        "police",
        "court",
        "ministry",
        "hospital",
        "election commission",
        "central bank",
    }

    lowered_text = (text or "").lower()
    has_trusted_entity = any(term in lowered_text for term in trusted_entity_terms)

    evidence_rows: List[Dict[str, object]] = []
    support_hits = 0
    contradiction_hits = 0
    overlap_scores: List[float] = []
    support_score = 0.0
    contradiction_score = 0.0

    for claim in claims:
        query = " ".join(_tokenize(claim)[:10])

        enriched: List[Dict[str, str]] = []
        enriched.extend(_duckduckgo_html_results(query)[:12])
        enriched.extend(_google_news_rss_results(query)[:10])
        enriched.extend({"text": s, "url": "", "source": "duckduckgo_api"} for s in _duckduckgo_api_snippets(query)[:6])
        enriched.extend({"text": s, "url": "https://wikipedia.org", "source": "wikipedia"} for s in _wikipedia_snippets(query)[:4])

        best_overlap = 0.0
        snippet_count = 0

        for ev in enriched:
            snippet = str(ev.get("text", "")).strip()
            if len(snippet) < 25:
                continue

            relevance = _relevance_score(claim, snippet)
            if relevance < 0.08:
                continue

            snippet_count += 1
            if relevance > best_overlap:
                best_overlap = relevance

            stance = _stance(snippet)
            weight = _domain_weight(str(ev.get("url", "")))

            if stance == "contradiction":
                contradiction_hits += 1
                contradiction_score += relevance * weight
            elif stance == "support":
                support_hits += 1
                support_score += relevance * weight
            elif weight >= 1.2 and relevance >= 0.28:
                support_score += relevance * 0.2

            evidence_rows.append(
                {
                    "claim": claim[:240],
                    "snippet": snippet[:320],
                    "source": str(ev.get("source", "web")),
                    "url": str(ev.get("url", ""))[:200],
                    "relevance": round(relevance, 3),
                    "stance": stance,
                }
            )

        overlap_scores.append(best_overlap)
        if snippet_count == 0:
            evidence_rows.append(
                {
                    "claim": claim[:240],
                    "snippet": "",
                    "source": "none",
                    "url": "",
                    "relevance": 0.0,
                    "stance": "neutral",
                }
            )

    avg_overlap = float(sum(overlap_scores) / max(len(overlap_scores), 1))
    total_snippet_count = int(sum(1 for row in evidence_rows if str(row.get("snippet", "")).strip()))

    if total_snippet_count == 0:
        return FactCheckSignal(
            used=False,
            fake_probability=0.5,
            confidence=0.0,
            reasons=["No reliable internet snippets retrieved; fact-check remains inconclusive."],
            meta={
                "claims_checked": len(claims),
                "sources_used": ["duckduckgo_api", "duckduckgo_html", "google_news_rss", "wikipedia"],
                "avg_overlap": 0.0,
                "support_hits": 0,
                "contradiction_hits": 0,
                "evidence": evidence_rows,
                "total_snippets": 0,
            },
        )

    fake_score = 0.5
    fake_score += min(contradiction_score * 0.28, 0.34)
    fake_score -= min(support_score * 0.26, 0.28)

    if contradiction_score > 0 and contradiction_score >= (support_score * 1.35):
        fake_score += 0.08
    if support_score > 0 and support_score >= (contradiction_score * 1.2):
        fake_score -= 0.06
    if avg_overlap >= 0.45 and support_score >= contradiction_score:
        fake_score -= 0.06

    if has_trusted_entity and contradiction_hits == 0:
        fake_score -= 0.08

    fake_score = _clamp(fake_score)

    reasons: List[str] = []
    if contradiction_hits > 0:
        reasons.append("Internet evidence includes contradiction/debunk signals.")
    if support_hits > 0:
        reasons.append("Internet evidence includes support cues from reporting-style sources.")
    if avg_overlap >= 0.45:
        reasons.append("Claim wording aligns with externally indexed references.")
    elif avg_overlap < 0.18:
        reasons.append("Limited external alignment found for extracted claims; evidence remains inconclusive.")
    if has_trusted_entity and contradiction_hits == 0:
        reasons.append("Claim mentions a trusted institution or source, which lowers fake risk.")
    if not reasons:
        reasons.append("Internet fact-check found mixed evidence.")

    return FactCheckSignal(
        used=True,
        fake_probability=fake_score,
        confidence=_confidence(fake_score),
        reasons=reasons[:4],
        meta={
            "claims_checked": len(claims),
            "sources_used": ["duckduckgo_api", "duckduckgo_html", "google_news_rss", "wikipedia"],
            "avg_overlap": round(avg_overlap, 3),
            "support_hits": support_hits,
            "contradiction_hits": contradiction_hits,
            "evidence": evidence_rows,
            "total_snippets": total_snippet_count,
        },
    )
