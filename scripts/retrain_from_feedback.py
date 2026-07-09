import pandas as pd

from app.db.mongo import feedback_collection
from app.services.training import train_from_csv


def build_feedback_dataset(output_csv: str = "data/feedback_dataset.csv") -> int:
    docs = list(feedback_collection().find({}, {"_id": 0}))
    if not docs:
        return 0

    rows = []
    for doc in docs:
        text = doc.get("input_text") or ""
        if not text:
            continue
        rows.append(
            {
                "text": text,
                "url": doc.get("input_url"),
                "label": str(doc.get("corrected_label", "REAL")).upper(),
            }
        )

    if not rows:
        return 0

    pd.DataFrame(rows).to_csv(output_csv, index=False)
    return len(rows)


if __name__ == "__main__":
    count = build_feedback_dataset()
    if count == 0:
        print("No feedback data available for retraining.")
    else:
        acc, samples = train_from_csv("data/feedback_dataset.csv")
        print(f"Feedback retraining complete | samples={samples} | validation_accuracy={acc:.4f}")
