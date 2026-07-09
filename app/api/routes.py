from typing import Optional
from pathlib import Path
import time
from datetime import datetime

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.core.config import settings
from app.db.mongo import feedback_collection, image_authenticity_collection, prediction_collection
from app.models.schemas import (
    DashboardStats,
    FeedbackRequest,
    ImageAuthenticityLog,
    ImageAuthenticityResponse,
    PredictionLog,
    PredictionResponse,
)
from app.services.classifier import classifier_singleton
from app.services.image_authenticity import image_authenticity_singleton
from app.services.ingestion import fetch_text_from_url, summarize_news_text
from app.services.training import train_from_csv
from app.video_auth.router import router as video_auth_router

router = APIRouter(prefix="/api/v1", tags=["fake-news"])
router.include_router(video_auth_router)


def _build_verdict_reason(label: str, reasons: list[str]) -> str:
    if not reasons:
        if label == "NEEDS_VERIFICATION":
            return "Needs verification due to limited corroborating internet evidence."
        return f"Classified as {label} based on combined content and URL signals."

    lead = reasons[0]
    if label == "NEEDS_VERIFICATION":
        if len(reasons) > 1:
            return f"Needs verification because {lead} Also, {reasons[1].rstrip('.')}."
        return f"Needs verification because {lead}"
    if len(reasons) > 1:
        return f"Classified as {label} because {lead} Also, {reasons[1].rstrip('.')}."
    return f"Classified as {label} because {lead}"


@router.post("/predict", response_model=PredictionResponse)
async def predict_news(
    text: Optional[str] = Form(default=None),
    url: Optional[str] = Form(default=None),
    image: Optional[UploadFile] = File(default=None),
):
    """Real-time multimodal prediction endpoint."""
    final_text = text or ""
    extracted: Optional[str] = None

    if url:
        extracted = fetch_text_from_url(url)
        if not extracted:
            raise HTTPException(
                status_code=422,
                detail="Could not read article content from URL. Please use a publicly accessible article URL.",
            )
        final_text = f"{final_text} {extracted}".strip() if final_text else extracted

    if not final_text and not image and not url:
        raise HTTPException(status_code=400, detail="Provide at least one input: text, image, or url")

    image_bytes = await image.read() if image else None
    # Ensure model is available; attempt to load artifacts if not fitted
    if not classifier_singleton.is_fitted:
        try:
            classifier_singleton.load(settings.model_path, settings.vectorizer_path)
        except Exception:
            raise HTTPException(
                status_code=503,
                detail="Model not available. Please retrain or ensure artifacts exist on disk.",
            )

    try:
        prediction = classifier_singleton.predict_single(text=final_text, url=url, image_bytes=image_bytes)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    summary = summarize_news_text(extracted or final_text) if (url and final_text) else None
    verdict_reason = _build_verdict_reason(prediction.label, prediction.reasons)

    log_entry = PredictionLog(
        input_text=final_text[:1000] if final_text else None,
        input_url=url,
        predicted_label=prediction.label,
        confidence=prediction.confidence,
        reasons=prediction.reasons,
        summary=summary,
        verdict_reason=verdict_reason,
    )
    prediction_collection().insert_one(log_entry.model_dump())

    return PredictionResponse(
        label=prediction.label,
        confidence=prediction.confidence,
        probabilities=prediction.probabilities,
        risk_score=prediction.risk_score,
        module_scores=prediction.module_scores,
        module_reasons=prediction.module_reasons,
        module_details=prediction.module_details,
        reasons=prediction.reasons,
        summary=summary,
        verdict_reason=verdict_reason,
        text_used=bool(final_text),
        image_used=bool(image_bytes),
        url_used=bool(url),
    )


@router.post("/feedback")
def submit_feedback(payload: FeedbackRequest):
    feedback_collection().insert_one(payload.model_dump())
    return {"message": "Feedback stored successfully"}


@router.post("/image-authenticity", response_model=ImageAuthenticityResponse)
async def analyze_image_authenticity(image: UploadFile = File(...)):
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Please upload a valid image file")

    try:
        result = image_authenticity_singleton.predict_single(image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    log_entry = ImageAuthenticityLog(
        predicted_label=result.label,
        confidence=result.confidence,
        probabilities=result.probabilities,
        reasons=result.reasons,
        suspicious_regions=result.suspicious_regions,
        model_name=result.model_name,
    )
    image_authenticity_collection().insert_one(log_entry.model_dump())

    return ImageAuthenticityResponse(
        label=result.label,
        confidence=result.confidence,
        probabilities=result.probabilities,
        reasons=result.reasons,
        suspicious_regions=result.suspicious_regions,
        overlay_image=result.overlay_image,
        model_name=result.model_name,
        image_used=True,
    )


@router.post("/image-authenticity/feedback")
async def image_authenticity_feedback(image: UploadFile = File(...), corrected_label: str = Form(...)):
    """Store user-corrected label and save the image into the feedback dataset for future retraining."""
    img_bytes = await image.read()
    if not img_bytes:
        raise HTTPException(status_code=400, detail="Please upload a valid image file")

    cleaned = str(corrected_label or "").strip().upper()
    if cleaned not in {"AI_GENERATED", "ORIGINAL", "AI", "REAL"}:
        raise HTTPException(status_code=400, detail="corrected_label must be either 'AI_GENERATED' or 'ORIGINAL'")

    label_folder = "ai_generated" if cleaned.startswith("AI") else "real"
    dest_root = Path(settings.image_auth_dataset_dir)
    feedback_folder = dest_root / "feedback" / label_folder
    feedback_folder.mkdir(parents=True, exist_ok=True)
    fname = f"fb_{int(time.time())}_{image.filename or 'upload'}.png"
    dest_path = feedback_folder / fname
    try:
        dest_path.write_bytes(img_bytes)
    except Exception:
        raise HTTPException(status_code=500, detail="Could not save feedback image")

    log = {
        "predicted_label": None,
        "corrected_label": cleaned,
        "filename": str(dest_path),
        "created_at": datetime.utcnow(),
    }
    image_authenticity_collection().insert_one(log)
    return {"message": "Feedback saved", "path": str(dest_path)}


@router.post("/image-authenticity/retrain")
def retrain_image_authenticity(cv: int = 5):
    """Retrain the image-authenticity detector using images under settings.image_auth_dataset_dir.

    Returns cross-validation accuracy and sample count.
    """
    try:
        accuracy, samples = image_authenticity_singleton.fit_with_cv(settings.image_auth_dataset_dir, cv=cv)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"message": "Image authenticity model retrained", "cv_accuracy": round(accuracy, 4), "samples": samples}


@router.post("/retrain")
def retrain_model():
    # Retrain from local CSV. Feedback-aware training can extend this step.
    accuracy, samples = train_from_csv("data/sample_news.csv")
    return {
        "message": "Model retrained",
        "trained_samples": samples,
        "validation_accuracy": round(accuracy, 4),
    }


@router.get("/dashboard", response_model=DashboardStats)
def get_dashboard_stats():
    total_predictions = prediction_collection().count_documents({})
    fake_predictions = prediction_collection().count_documents({"predicted_label": "FAKE"})
    real_predictions = prediction_collection().count_documents({"predicted_label": "REAL"})
    total_feedback = feedback_collection().count_documents({})
    corrected_cases = 0
    for doc in feedback_collection().find({}, {"predicted_label": 1, "corrected_label": 1}):
        if str(doc.get("predicted_label", "")).upper() != str(doc.get("corrected_label", "")).upper():
            corrected_cases += 1

    return DashboardStats(
        total_predictions=total_predictions,
        fake_predictions=fake_predictions,
        real_predictions=real_predictions,
        total_feedback=total_feedback,
        corrected_cases=corrected_cases,
    )


@router.get("/health")
def health_check():
    return {"status": "ok", "app": settings.app_name}
