from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.core.config import settings
from app.db.mongo import video_authenticity_collection
from app.video_auth.schemas import VideoAuthenticityLog, VideoAuthenticityResponse
from app.video_auth.service import video_authenticity_singleton

router = APIRouter(prefix="/video-authenticity", tags=["video-authenticity"])


@router.post("", response_model=VideoAuthenticityResponse)
async def analyze_video_authenticity(video: UploadFile = File(...)):
    video_bytes = await video.read()
    if not video_bytes:
        raise HTTPException(status_code=400, detail="Please upload a valid video file")

    try:
        result = video_authenticity_singleton.predict_video(video_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        video_authenticity_collection().insert_one(
            VideoAuthenticityLog(
                predicted_label=result.label,
                confidence=result.confidence,
                frame_scores=result.frame_scores,
                suspicious_frames=result.suspicious_frames,
                model_name=getattr(video_authenticity_singleton, "model_name", None),
            ).model_dump()
        )
    except Exception:
        pass

    return VideoAuthenticityResponse(
        label=result.label,
        confidence=result.confidence,
        frame_scores=result.frame_scores,
        suspicious_frames=result.suspicious_frames,
        model_name=getattr(video_authenticity_singleton, "model_name", None),
    )


@router.post("/feedback")
async def video_feedback(video: UploadFile = File(...), corrected_label: str = Form(...)):
    """Persist a corrected video sample into the module dataset for retraining."""
    video_bytes = await video.read()
    if not video_bytes:
        raise HTTPException(status_code=400, detail="Please upload a valid video file")

    cleaned = str(corrected_label or "").strip().upper()
    if cleaned not in {"REAL", "FAKE", "AI_GENERATED", "MANIPULATED"}:
        raise HTTPException(status_code=400, detail="corrected_label must be REAL or FAKE")

    label_folder = "real" if cleaned == "REAL" else "ai_generated"
    dest_root = Path(settings.video_auth_dataset_dir)
    feedback_folder = dest_root / "feedback" / label_folder
    feedback_folder.mkdir(parents=True, exist_ok=True)
    filename = f"fb_{video.filename or 'upload'}.mp4"
    dest_path = feedback_folder / filename
    dest_path.write_bytes(video_bytes)

    try:
        video_authenticity_collection().insert_one(
            {
                "predicted_label": None,
                "corrected_label": cleaned,
                "filename": str(dest_path),
            }
        )
    except Exception:
        pass

    return {"message": "Feedback saved", "path": str(dest_path)}
