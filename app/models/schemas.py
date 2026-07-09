from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    label: str
    confidence: float
    probabilities: Dict[str, float]
    risk_score: float
    module_scores: Dict[str, float]
    module_reasons: Dict[str, List[str]]
    module_details: Dict[str, Dict[str, object]]
    reasons: List[str]
    summary: Optional[str] = None
    verdict_reason: Optional[str] = None
    text_used: bool
    image_used: bool
    url_used: bool


class ImageAuthenticityResponse(BaseModel):
    label: str
    confidence: float
    probabilities: Dict[str, float]
    reasons: List[str]
    suspicious_regions: List[Dict[str, object]]
    overlay_image: Optional[str] = None
    model_name: str
    image_used: bool = True


class ImageAuthenticityLog(BaseModel):
    predicted_label: str
    confidence: float
    probabilities: Dict[str, float]
    reasons: List[str]
    suspicious_regions: List[Dict[str, object]]
    model_name: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class FeedbackRequest(BaseModel):
    input_text: Optional[str] = None
    input_url: Optional[str] = None
    predicted_label: str
    corrected_label: str
    confidence: float
    created_at: datetime = Field(default_factory=datetime.utcnow)


class DashboardStats(BaseModel):
    total_predictions: int
    fake_predictions: int
    real_predictions: int
    total_feedback: int
    corrected_cases: int


class PredictionLog(BaseModel):
    input_text: Optional[str] = None
    input_url: Optional[str] = None
    predicted_label: str
    confidence: float
    reasons: List[str]
    summary: Optional[str] = None
    verdict_reason: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


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
