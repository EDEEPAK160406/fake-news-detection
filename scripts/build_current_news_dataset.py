from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
import requests
from bs4 import BeautifulSoup


RSS_FEEDS = [
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://feeds.bbci.co.uk/news/world/asia/rss.xml",
    "https://feeds.bbci.co.uk/news/business/rss.xml",
    "https://feeds.bbci.co.uk/news/technology/rss.xml",
    "https://feeds.bbci.co.uk/news/india/rss.xml",
    "https://www.thehindu.com/news/national/?service=rss",
    "https://www.thehindu.com/news/international/?service=rss",
    "https://www.thehindu.com/business/?service=rss",
    "https://www.livemint.com/rss/news",
    "https://www.aljazeera.com/xml/rss/all.xml",
]

OUTPUT = Path("data/current_real_news.csv")


def _fetch_feed(url: str, timeout: int = 10) -> List[dict]:
    try:
        response = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        if response.status_code != 200:
            return []

        soup = BeautifulSoup(response.text, "xml")
        rows: List[dict] = []
        for item in soup.find_all("item")[:120]:
            title = (item.title.text if item.title else "").strip()
            description = (item.description.text if item.description else "").strip()
            link = (item.link.text if item.link else "").strip()
            if len(title) < 15:
                continue

            text = f"{title} {description}".strip()
            text = " ".join(text.split())
            if len(text) < 40:
                continue

            rows.append(
                {
                    "text": text,
                    "label": "REAL",
                    "url": link,
                    "source_feed": url,
                }
            )

        return rows
    except Exception:
        return []


def main() -> None:
    all_rows: List[dict] = []
    for feed in RSS_FEEDS:
        all_rows.extend(_fetch_feed(feed))

    if not all_rows:
        raise RuntimeError("No rows fetched from RSS feeds")

    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates(subset=["text"])
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)

    print(f"Saved {len(df)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
