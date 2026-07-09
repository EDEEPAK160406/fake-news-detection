from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.config import settings
from app.services.classifier import classifier_singleton
from app.services.training import train_from_csv

app = FastAPI(title=settings.app_name, version=settings.app_version)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")


@app.on_event("startup")
def startup_event():
    model_exists = Path(settings.model_path).exists() and Path(settings.vectorizer_path).exists()
    if model_exists:
        classifier_singleton.load(settings.model_path, settings.vectorizer_path)
    else:
        train_from_csv("data/sample_news.csv")
