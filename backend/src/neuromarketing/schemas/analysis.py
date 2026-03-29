from datetime import datetime

from pydantic import BaseModel, Field

from neuromarketing.schemas.enums import AnalysisStatus, ProcessingStage


# --- Request/Response Models ---


class AnalysisCreateResponse(BaseModel):
    analysis_id: str
    status: AnalysisStatus
    created_at: datetime


class AnalysisStatusResponse(BaseModel):
    analysis_id: str
    status: AnalysisStatus
    stage: ProcessingStage | None = None
    stage_label: str | None = None
    progress_percent: int = 0


class VideoMetadataSchema(BaseModel):
    filename: str
    duration_seconds: float
    resolution: str
    file_size_bytes: int


# --- Report Models ---


class TimelineSegment(BaseModel):
    start_seconds: float
    end_seconds: float
    label: str
    dominant_regions: list[str]
    cognitive_state: str
    engagement_level: str = Field(pattern="^(high|medium|low)$")
    insight: str


class RegionInsight(BaseModel):
    roi_name: str
    full_name: str
    cognitive_function: str
    activation_rank: int
    marketing_implication: str


class Recommendation(BaseModel):
    category: str
    title: str
    description: str
    priority: str = Field(pattern="^(high|medium|low)$")


class AnalysisReport(BaseModel):
    executive_summary: str
    overall_score: int = Field(ge=1, le=100)
    timeline: list[TimelineSegment]
    top_regions: list[RegionInsight]
    recommendations: list[Recommendation]
    strengths: list[str]
    weaknesses: list[str]


class AnalysisResultResponse(BaseModel):
    analysis_id: str
    status: AnalysisStatus
    created_at: datetime
    completed_at: datetime | None = None
    video_metadata: VideoMetadataSchema
    report: AnalysisReport
    brain_images: list[str]


# --- API Envelope ---


class ApiResponse(BaseModel):
    success: bool = True
    data: dict | AnalysisCreateResponse | AnalysisStatusResponse | AnalysisResultResponse | None = None
    error: str | None = None
