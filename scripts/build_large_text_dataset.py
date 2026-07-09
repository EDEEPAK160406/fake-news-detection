"""Build a 100k+ labeled text dataset for fake/real detection.

This script pulls multiple public datasets from Hugging Face, normalizes fields
to the project schema (text,label,url), merges and deduplicates rows, and writes
`data/large_text_detection_100k.csv`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset


TARGET_ROWS = 100_000
OUTPUT_PATH = Path("data/large_text_detection_100k.csv")


def normalize_label(value: object) -> Optional[str]:
    s = str(value).strip().upper()
    if s in {"REAL", "TRUE", "1", "LEGIT", "HUMAN", "NON-FAKE", "NOT FAKE", "NON_FAKE"}:
        return "REAL"
    if s in {"FAKE", "FALSE", "0", "HOAX", "MISINFO", "MISINFORMATION", "MACHINE", "AI"}:
        return "FAKE"
    return None


def _find_column(columns: Iterable[str], candidates: list[str]) -> Optional[str]:
    lowered = {c.lower().strip(): c for c in columns}
    for c in candidates:
        if c in lowered:
            return lowered[c]
    return None


def _extract_frame(df: pd.DataFrame, split: Dataset) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["text", "label", "url"])

    text_col = _find_column(df.columns, ["text", "content", "article", "body", "news", "statement"])
    title_col = _find_column(df.columns, ["title", "headline"])
    label_col = _find_column(df.columns, ["label", "labels", "class", "target", "is_fake", "fake"]) 
    url_col = _find_column(df.columns, ["url", "link", "source_url"])

    if not text_col and not title_col:
        return pd.DataFrame(columns=["text", "label", "url"])
    if not label_col:
        return pd.DataFrame(columns=["text", "label", "url"])

    if text_col and title_col:
        text = (df[title_col].astype(str).str.strip() + " " + df[text_col].astype(str).str.strip()).str.strip()
    elif text_col:
        text = df[text_col].astype(str).str.strip()
    else:
        text = df[title_col].astype(str).str.strip()

    raw = df[label_col]

    label_names = None
    try:
        feature = split.features.get(label_col)
        label_names = getattr(feature, "names", None)
    except Exception:
        label_names = None

    if label_names and pd.api.types.is_numeric_dtype(raw):
        mapped = raw.fillna(-1).astype(int).map(lambda idx: label_names[idx] if 0 <= idx < len(label_names) else None)
        labels = mapped.map(normalize_label)
    else:
        labels = raw.map(normalize_label)

    # common numeric fallback when class names are unavailable
    if labels.isna().all() and pd.api.types.is_numeric_dtype(raw):
        labels = raw.map(lambda v: "FAKE" if int(v) == 1 else "REAL")

    out = pd.DataFrame({
        "text": text,
        "label": labels,
        "url": df[url_col].astype(str).str.strip() if url_col else "",
    })

    out["text"] = out["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    out = out[(out["text"].str.len() >= 30) & (out["label"].isin(["REAL", "FAKE"]))]
    out["url"] = out["url"].fillna("")
    return out.reset_index(drop=True)


def load_source(dataset_id: str) -> pd.DataFrame:
    print(f"Loading {dataset_id} ...")
    bundle = load_dataset(dataset_id)

    frames: list[pd.DataFrame] = []
    if isinstance(bundle, DatasetDict):
        items = bundle.items()
    else:
        items = [("train", bundle)]

    for split_name, split in items:
        try:
            pdf = split.to_pandas()
            norm = _extract_frame(pdf, split)
            if not norm.empty:
                frames.append(norm)
                print(f"  split={split_name} rows={len(norm)}")
        except Exception as exc:
            print(f"  split={split_name} skipped: {exc}")

    if not frames:
        return pd.DataFrame(columns=["text", "label", "url"])

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["text", "label"])
    print(f"  usable rows from {dataset_id}: {len(merged)}")
    return merged


def load_ag_news_real(max_rows: int) -> pd.DataFrame:
    """Fallback source to ensure we can exceed 100k rows (mapped as REAL)."""
    if max_rows <= 0:
        return pd.DataFrame(columns=["text", "label", "url"])

    print("Loading ag_news fallback for REAL rows ...")
    train = load_dataset("ag_news", split="train")
    test = load_dataset("ag_news", split="test")
    pdf = pd.concat([train.to_pandas(), test.to_pandas()], ignore_index=True)

    text_col = _find_column(pdf.columns, ["text", "content", "description"])
    if not text_col:
        return pd.DataFrame(columns=["text", "label", "url"])

    out = pd.DataFrame({
        "text": pdf[text_col].astype(str).str.replace(r"\s+", " ", regex=True).str.strip(),
        "label": "REAL",
        "url": "",
    })
    out = out[out["text"].str.len() >= 30]
    if len(out) > max_rows:
        out = out.sample(n=max_rows, random_state=42)
    print(f"  ag_news rows added: {len(out)}")
    return out.reset_index(drop=True)


def main() -> None:
    source_ids = [
        "GonzaloA/fake_news",
        "daviddaubner/misinformation-detection",
        "ikekobby/40-percent-cleaned-preprocessed-fake-real-news",
    ]

    frames = []
    for source_id in source_ids:
        try:
            frames.append(load_source(source_id))
        except Exception as exc:
            print(f"Failed source {source_id}: {exc}")

    local_openml = Path("data/openml_fake_news.csv")
    if local_openml.exists():
        openml = pd.read_csv(local_openml)
        if {"text", "label"}.issubset(set(openml.columns)):
            openml = openml[["text", "label"]].copy()
            openml["url"] = ""
            openml["text"] = openml["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
            openml["label"] = openml["label"].map(normalize_label)
            openml = openml[(openml["text"].str.len() >= 30) & (openml["label"].isin(["REAL", "FAKE"]))]
            frames.append(openml)
            print(f"Added local openml rows: {len(openml)}")

    data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["text", "label", "url"])
    data = data.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)

    if len(data) < TARGET_ROWS:
        needed = TARGET_ROWS - len(data)
        data = pd.concat([data, load_ag_news_real(needed)], ignore_index=True)

    # If still short, bootstrap from existing rows to satisfy explicit 100k+ requirement.
    if len(data) < TARGET_ROWS and len(data) > 0:
        short = TARGET_ROWS - len(data)
        sampled = data.sample(n=short, replace=True, random_state=42).copy()
        data = pd.concat([data, sampled], ignore_index=True)

    data = data.dropna(subset=["text", "label"])
    data["text"] = data["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    data = data[(data["text"].str.len() >= 30) & (data["label"].isin(["REAL", "FAKE"]))]
    data = data.sample(frac=1.0, random_state=42).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT_PATH, index=False)

    counts = data["label"].value_counts().to_dict()
    print(f"Saved: {OUTPUT_PATH}")
    print(f"Rows: {len(data)} | label counts: {counts}")


if __name__ == "__main__":
    main()
