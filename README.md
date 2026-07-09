# Multimodal Fake News Detection System (Text + Image + URL)

A complete, real-time fake news detection system using a multimodal pipeline:
- NLP text analysis
- Image feature analysis + OCR
- URL credibility/risk analysis
- Feature fusion + classification
- Explainable AI reasons
- Feedback capture + retraining
- Dashboard metrics

## 1) Project Structure

```text
fnd/
├── app/
│   ├── api/routes.py
│   ├── core/config.py
│   ├── db/mongo.py
│   ├── models/schemas.py
│   ├── services/
│   │   ├── ingestion.py
│   │   ├── preprocessing.py
│   │   ├── feature_extraction.py
│   │   ├── classifier.py
│   │   └── training.py
│   └── main.py
├── frontend/
│   ├── index.html
│   ├── styles.css
│   └── app.js
├── scripts/
│   ├── train_model.py
│   └── retrain_from_feedback.py
├── data/sample_news.csv
├── artifacts/                  # trained model files generated at runtime
├── requirements.txt
└── .env.example
```

## 2) Core Capabilities Implemented

1. Data ingestion for text/image/url input.
2. Text preprocessing: cleaning, stopword removal, tokenization.
3. Image preprocessing: decode, resize, grayscale.
4. URL feature extraction: HTTPS, length, symbols, IP-like host, suspicious TLD.
5. Text embeddings with TF-IDF.
6. Image feature extraction with OpenCV + OCR support (Tesseract).
7. Multimodal fusion: concatenate text + image + URL vectors.
8. Classification with Logistic Regression.
9. Explainable output reasons from text/url/image heuristics.
10. Real-time prediction API + browser UI.
11. Feedback storage and retraining endpoints/scripts.
12. Dashboard with prediction and feedback statistics.

## 3) Dataset

Included starter dataset: `data/sample_news.csv`.

Recommended Kaggle datasets for production-quality training:
1. Fake and Real News Dataset: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
2. WELFake Dataset: https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification
3. Fakeddit (multimodal): https://www.kaggle.com/datasets/akshaybabloo/fakeddit

## 4) Local Setup

### Prerequisites
1. Python 3.10+
2. MongoDB running locally (`mongodb://localhost:27017`)
3. Tesseract OCR installed and available in PATH (optional but recommended)

### Install and run

```powershell
cd C:\Users\HP\Desktop\fnd
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
python scripts/train_model.py
uvicorn app.main:app --reload
```

Open: `http://127.0.0.1:8000`

### Strong Accuracy Setup (Recommended)

The built-in sample dataset is only for demo and is too small for robust real-world accuracy.

1. Download a large labeled dataset (for example `True.csv` and `Fake.csv`) and place both files in `data/`.
2. Build normalized training corpus:

```powershell
python scripts/prepare_external_dataset.py
```

3. Retrain model artifacts:

```powershell
python scripts/train_model.py
```

The trainer now auto-merges compatible CSVs inside `data/` (sample data, feedback data, and external corpus files).

## 5) API Endpoints

1. `POST /api/v1/predict` (multipart form)
   - Fields: `text` (optional), `url` (optional), `image` (optional file)
   - Returns label (`REAL`/`FAKE`), confidence, and explanation reasons.
2. `POST /api/v1/feedback`
   - JSON payload with predicted and corrected label.
3. `POST /api/v1/retrain`
   - Retrains model from local dataset.
4. `GET /api/v1/dashboard`
   - Returns aggregate metrics.
5. `GET /api/v1/health`
   - Health check.

## 6) How Explainability Works

The system returns reasons such as:
- Suspicious URL features (e.g., non-HTTPS, URL shortener, suspicious TLD)
- Image consistency/manipulation indicators (e.g., low sharpness, unusual edge density)
- Text risk cues (e.g., sensational terms, very short unverifiable claims)

## 7) Retraining Workflow

1. Collect user corrections through `POST /api/v1/feedback`.
2. Build a feedback dataset and retrain:

```powershell
python scripts/retrain_from_feedback.py
```

3. The updated model artifacts are saved in `artifacts/`.

## 8) Notes for Scaling

1. Replace handcrafted image features with a CNN embedding model (e.g., EfficientNet).
2. Replace TF-IDF with transformer embeddings (e.g., BERT from Hugging Face).
3. Add domain reputation APIs and WHOIS/domain-age enrichment.
4. Use background workers for asynchronous retraining.
5. Add authentication and role-based moderation for enterprise use.
