import re
from typing import Optional

import requests
from bs4 import BeautifulSoup


def _clean_chunks(chunks: list[str], max_chars: int = 1800) -> Optional[str]:
    normalized: list[str] = []
    seen: set[str] = set()

    for chunk in chunks:
        text = re.sub(r"\s+", " ", (chunk or "")).strip()
        if len(text) < 20 or text in seen:
            continue
        seen.add(text)
        normalized.append(text)

    if not normalized:
        return None

    joined = " ".join(normalized)
    return joined[:max_chars].strip()


def summarize_news_text(text: str, max_sentences: int = 3, max_chars: int = 420) -> str:
    """Return a concise article summary from extracted page text."""
    cleaned = re.sub(r"\s+", " ", (text or "")).strip()
    if not cleaned:
        return ""

    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    selected: list[str] = []
    total_chars = 0

    for sentence in sentences:
        compact = sentence.strip()
        if len(compact) < 30:
            continue
        selected.append(compact)
        total_chars += len(compact) + 1
        if len(selected) >= max_sentences or total_chars >= max_chars:
            break

    if not selected:
        return cleaned[:max_chars].strip()

    summary = " ".join(selected).strip()
    return summary[:max_chars].strip()


def fetch_text_from_url(url: str, timeout: int = 8) -> Optional[str]:
    """Fetch a news URL and extract the most useful article text available."""
    try:
        response = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        if response.status_code != 200:
            return None

        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup.find_all(["script", "style", "noscript"]):
            tag.decompose()

        title = soup.title.get_text(" ", strip=True) if soup.title else ""
        meta_title = ""
        meta_description = ""

        title_tag = soup.find("meta", attrs={"property": "og:title"}) or soup.find("meta", attrs={"name": "title"})
        if title_tag and title_tag.get("content"):
            meta_title = title_tag.get("content", "")

        description_tag = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", attrs={"property": "og:description"})
        if description_tag and description_tag.get("content"):
            meta_description = description_tag.get("content", "")

        article_chunks = [title, meta_title, meta_description]

        article_tags = soup.find_all(["article", "main"])
        for article in article_tags[:2]:
            article_chunks.extend([p.get_text(" ", strip=True) for p in article.find_all(["h1", "h2", "h3", "p"])[:12]])

        if len(article_chunks) < 4:
            article_chunks.extend([p.get_text(" ", strip=True) for p in soup.find_all("p")[:8]])
            article_chunks.extend([h.get_text(" ", strip=True) for h in soup.find_all(["h1", "h2"])[:4]])

        return _clean_chunks(article_chunks)
    except requests.RequestException:
        return None
