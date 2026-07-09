from app.services.training import train_from_csv


if __name__ == "__main__":
    accuracy, samples = train_from_csv("data/sample_news.csv")
    print(f"Training complete | samples={samples} | validation_accuracy={accuracy:.4f}")
    if samples < 1000:
        print("WARNING: Training data is small. Add large True/Fake CSV files in data/ and run scripts/prepare_external_dataset.py for stronger real-world accuracy.")
