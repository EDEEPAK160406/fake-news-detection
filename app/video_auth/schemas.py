from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class VideoAuthenticityResponse(BaseModel):
    label: str
    confidence: float
    frame_scores: List[float]
    suspicious_frames: List[int]
    model_name: Optional[str] = None


class VideoAuthenticityLog(BaseModel):
    predicted_label: str
    confidence: float
    frame_scores: List[float]
    suspicious_frames: List[int]
    model_name: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
