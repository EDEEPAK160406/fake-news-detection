from pathlib import Path

import pandas as pd


DATA_DIR = Path("data")
OUTPUT_PATH = DATA_DIR / "external_news_corpus.csv"


def _load_labeled_csv(file_path: Path, label: str) -> pd.DataFrame:
    frame = pd.read_csv(file_path)
    cols = {c.lower().strip(): c for c in frame.columns}
    if "text" not in cols:
        raise ValueError(f"Missing 'text' column in {file_path.name}")

    if "title" in cols:
        merged = (frame[cols["title"]].astype(str).str.strip() + " " + frame[cols["text"]].astype(str).str.strip()).str.strip()
    else:
        merged = frame[cols["text"]].astype(str).str.strip()

    out = pd.DataFrame({"text": merged, "label": label, "url": ""})
    out = out[out["text"].str.len() >= 30]
    return out


def main() -> None:
    true_csv = DATA_DIR / "True.csv"
    fake_csv = DATA_DIR / "Fake.csv"

    if not true_csv.exists() or not fake_csv.exists():
        raise FileNotFoundError(
            "Place True.csv and Fake.csv in data/ first. "
            "Then rerun: python scripts/prepare_external_dataset.py"
        )

    real_df = _load_labeled_csv(true_csv, "REAL")
    fake_df = _load_labeled_csv(fake_csv, "FAKE")

    combined = pd.concat([real_df, fake_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["text", "label"]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_PATH, index=False)

    print(f"Prepared dataset: {OUTPUT_PATH}")
    print(f"Rows: {len(combined)} | REAL: {(combined['label'] == 'REAL').sum()} | FAKE: {(combined['label'] == 'FAKE').sum()}")


if __name__ == "__main__":
    main()
