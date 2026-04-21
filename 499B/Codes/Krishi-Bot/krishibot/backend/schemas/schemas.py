from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Chat schemas
# ---------------------------------------------------------------------------


class ChatMessageRequest(BaseModel):
    session_id: str = Field(..., description="UUID or any unique identifier for the conversation")
    message: str = Field(..., min_length=1, description="User's message text")
    language: str = Field(default="en", description="BCP-47 language code, e.g. 'en', 'bn', 'hi'")


class ChatMessageResponse(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    timestamp: datetime

    model_config = {"from_attributes": True}


class ChatHistoryResponse(BaseModel):
    session_id: str
    messages: list[ChatMessageResponse]

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Disease analysis schemas
# ---------------------------------------------------------------------------


class AnalysisResponse(BaseModel):
    is_diseased: bool
    disease_name: str
    # Qualitative confidence returned by the LLM rather than a raw float
    confidence: Literal["High", "Medium", "Low"]
    symptoms: list[str] = Field(default_factory=list)
    treatment: list[str] = Field(default_factory=list)
    prevention: list[str] = Field(default_factory=list)
    # The raw LLM output — useful for debugging and logging
    raw_response: str


# ---------------------------------------------------------------------------
# Advisory schemas
# ---------------------------------------------------------------------------


class AdvisoryRequest(BaseModel):
    crop: str = Field(..., description="Crop name, e.g. 'rice', 'wheat', 'tomato'")
    topic: str = Field(
        ...,
        description="Advisory topic, e.g. 'irrigation', 'fertilizer', 'pest control'",
    )
    question: str | None = Field(
        default=None,
        description="Optional specific question to narrow the advisory response",
    )


class AdvisoryResponse(BaseModel):
    crop: str
    topic: str
    content: str


# ---------------------------------------------------------------------------
# Community disease board schemas
# ---------------------------------------------------------------------------


class CommunityReportCreate(BaseModel):
    disease_name:  str = Field(..., min_length=1, max_length=128)
    crop_name:     str = Field(..., min_length=1, max_length=64)
    location_name: str = Field(..., min_length=1, max_length=128)
    division:      str = Field(default="Dhaka", max_length=64)
    severity:      Literal["Low", "Medium", "High"] = Field(default="Medium")
    notes:         str | None = Field(default=None, max_length=200)


class CommunityReportResponse(BaseModel):
    id:            int
    disease_name:  str
    crop_name:     str
    location_name: str
    division:      str | None
    severity:      str
    notes:         str | None
    created_at:    datetime
    report_date:   str

    model_config = {"from_attributes": True}


class DiseaseCount(BaseModel):
    disease_name: str
    count: int


class LocationCount(BaseModel):
    location_name: str
    count: int


class DivisionCount(BaseModel):
    division: str
    count: int


class CommunityStatsResponse(BaseModel):
    total_reports:        int
    top_diseases:         list[DiseaseCount]
    top_locations:        list[LocationCount]
    recent_reports:       list[CommunityReportResponse]
    reports_by_division:  list[DivisionCount]


# ---------------------------------------------------------------------------
# PDF report schema
# ---------------------------------------------------------------------------


class PDFReportRequest(BaseModel):
    disease_name: str
    is_diseased:  bool
    confidence:   str
    symptoms:     list[str] = Field(default_factory=list)
    treatment:    list[str] = Field(default_factory=list)
    prevention:   list[str] = Field(default_factory=list)
    filename:     str = ""


# ---------------------------------------------------------------------------
# Generic error schema
# ---------------------------------------------------------------------------


class ErrorResponse(BaseModel):
    error: str = Field(..., description="Short error type, e.g. 'ValidationError'")
    detail: str = Field(..., description="Human-readable explanation of what went wrong")
