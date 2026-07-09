from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
import requests
from bs4 import BeautifulSoup


OUTPUT = Path("data/web_current_real_news.csv")

TOPIC_QUERIES = [
    "india election commission latest update",
    "reserve bank inflation policy statement",
    "tamil nadu assembly news",
    "supreme court india latest judgement",
    "parliament delimitation bill debate",
    "world bank economic outlook report",
    "imf global economy update",
    "un climate summit official statement",
    "nasa official mission update",
    "who public health advisory",
    "reuters world news today",
    "bbc asia latest report",
    "associated press international news",
    "hindu national news india",
    "financial times market update",
    "bloomberg economy briefing",
    "ukraine war official update",
    "middle east ceasefire negotiation news",
    "g20 summit declaration",
    "technology regulation policy update",
]


def _search_snippets(query: str) -> List[str]:
    try:
        resp = requests.get(
            "https://duckduckgo.com/html/",
            params={"q": query},
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        if resp.status_code != 200:
            return []

        soup = BeautifulSoup(resp.text, "html.parser")
        snippets: List[str] = []

        for node in soup.select("a.result__snippet, div.result__snippet")[:30]:
            text = " ".join(node.get_text(" ", strip=True).split())
            if len(text) >= 40:
                snippets.append(text)

        # Fallback for alternate markup.
        if not snippets:
            for node in soup.select("div.result, div.web-result")[:20]:
                text = " ".join(node.get_text(" ", strip=True).split())
                if len(text) >= 40:
                    snippets.append(text[:360])

        return snippets
    except Exception:
        return []


def main() -> None:
    rows = []
    for query in TOPIC_QUERIES:
        snippets = _search_snippets(query)
        for snippet in snippets:
            rows.append({"text": snippet, "label": "REAL", "url": ""})

    if not rows:
        raise RuntimeError("No snippets fetched from web search")

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)

    print(f"Saved {len(df)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
